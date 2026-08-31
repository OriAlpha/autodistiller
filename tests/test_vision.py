"""Vision models: the arithmetic, the subset, and the refusals.

The three things that fail quietly rather than loudly. A ViT's shape is read
from field names an encoder spells the same way, so a wrong answer looks like a
right one. An ImageNet split arrives sorted by class, so a subset taken from
the front scores a handful of classes and reports it as a number about a
thousand. And a runtime with no server for a model still accepts the command,
so the refusal has to come from here.
"""

from __future__ import annotations

import io

import pytest
import torch
from PIL import Image

from autodistiller.architecture import DECODER, ENCODER, VISION, model_kind
from autodistiller.config import ImageClassificationTask
from autodistiller.evaluation.datasets import _evenly_spaced, load_image_classification
from autodistiller.evaluation.image_classification import evaluate_image_classification
from autodistiller.models.loader import LoadedModel, auto_class_for


class _Config:
    """A stand-in for a Hugging Face config: attributes and nothing else."""

    def __init__(self, **fields):
        self.__dict__.update(fields)


def _vit_config(**overrides):
    fields = dict(
        architectures=["ViTForImageClassification"],
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        image_size=224,
        patch_size=16,
        num_channels=3,
        id2label={i: f"class-{i}" for i in range(1000)},
    )
    fields.update(overrides)
    return _Config(**fields)


# --- which kind of model is this ----------------------------------------


def test_image_classifier_is_its_own_kind():
    assert model_kind(["ViTForImageClassification"]) == VISION
    assert model_kind(["SwinForImageClassification"]) == VISION
    # A VLM generates text and is read by its decoder, not as a vision tower.
    assert model_kind(["Gemma3ForConditionalGeneration"]) == DECODER
    assert model_kind(["BertForMaskedLM"]) == ENCODER


def test_vision_loads_with_its_classifier():
    """``AutoModel`` would drop the head, and a dropped head scores like noise."""
    from transformers import AutoModelForImageClassification

    assert auto_class_for(_vit_config()) is AutoModelForImageClassification


# --- the subset ----------------------------------------------------------


def test_a_limit_spans_the_split_rather_than_its_front():
    """ImageNet val holds 50 images of class 0, then 50 of class 1, ..."""
    indices = _evenly_spaced(50_000, 4)
    assert indices == [0, 12_500, 25_000, 37_500]
    assert _evenly_spaced(100, None) is None
    assert _evenly_spaced(100, 500) is None  # asking for more than there is


def _write_image_set(directory, labels):
    """A tiny JSONL image set on disk, one solid-colour PNG per label."""
    import json

    rows = []
    for index, label in enumerate(labels):
        path = directory / f"{index}.png"
        Image.new("RGB", (32, 32), color=(index * 7 % 256, 0, 0)).save(path)
        rows.append({"image": path.name, "label": label})
    manifest = directory / "images.jsonl"
    manifest.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    return manifest


def test_local_image_sets_load_and_fingerprint(tmp_path):
    from autodistiller.config import DatasetSpec

    manifest = _write_image_set(tmp_path, [0, 1, 2, 3])
    spec = DatasetSpec(source="jsonl", path=str(manifest))
    dataset = load_image_classification(spec)

    assert dataset.n_examples == 4
    assert dataset.labels == [0, 1, 2, 3]
    assert all(isinstance(blob, bytes) for blob in dataset.images)
    # The same files hash the same way; different labels do not.
    assert load_image_classification(spec).fingerprint == dataset.fingerprint


# --- scoring -------------------------------------------------------------


class _FakeProcessor:
    """Turns PIL images into a batch, the way an image processor does."""

    def __call__(self, images, return_tensors=None):
        return {"pixel_values": torch.zeros(len(images), 3, 8, 8)}


class _FakeClassifier(torch.nn.Module):
    """Predicts each image's label from the red channel it was drawn with."""

    def __init__(self, n_labels: int, wrong: set[int] | None = None):
        super().__init__()
        self.config = _Config(num_labels=n_labels, architectures=["ViTForImageClassification"])
        self.calls = 0
        self.wrong = wrong or set()

    def forward(self, pixel_values=None, **_):
        rows = []
        for _ in range(pixel_values.shape[0]):
            logits = torch.zeros(self.config.num_labels)
            index = self.calls
            # Rank the true label first, unless this one is meant to be wrong,
            # in which case it lands second -- inside top-5, outside top-1.
            true = index % self.config.num_labels
            if index in self.wrong:
                logits[(true + 1) % self.config.num_labels] = 2.0
                logits[true] = 1.0
            else:
                logits[true] = 2.0
            rows.append(logits)
            self.calls += 1
        return _Config(logits=torch.stack(rows))


def _handle(model):
    return LoadedModel(
        model=model,
        tokenizer=_FakeProcessor(),
        info=None,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )


def _image_set(n_labels, n_images):
    from autodistiller.evaluation.datasets import ImageSet

    blobs = []
    for index in range(n_images):
        buffer = io.BytesIO()
        Image.new("RGB", (8, 8), color=(index, 0, 0)).save(buffer, format="PNG")
        blobs.append(buffer.getvalue())
    return ImageSet(
        images=blobs,
        labels=[i % n_labels for i in range(n_images)],
        fingerprint="test",
        source="test:images",
    )


def test_top1_and_top5_are_scored_separately():
    dataset = _image_set(n_labels=10, n_images=8)
    # Two of the eight are ranked second: wrong at top-1, right within top-5.
    model = _FakeClassifier(n_labels=10, wrong={2, 5})
    task = ImageClassificationTask(name="t", dataset={"path": "x"}, batch_size=4)

    result = evaluate_image_classification(_handle(model), task, dataset=dataset)

    assert result.metric("acc").value == pytest.approx(6 / 8)
    assert result.metric("acc_top5").value == pytest.approx(1.0)
    assert result.n_samples == 8


def test_a_label_the_model_cannot_predict_is_refused():
    """Scoring it anyway would report a mismatched pair of things as a bad model."""
    dataset = _image_set(n_labels=10, n_images=4)
    dataset.labels[0] = 999
    task = ImageClassificationTask(name="t", dataset={"path": "x"})

    with pytest.raises(ValueError, match="out of range"):
        evaluate_image_classification(_handle(_FakeClassifier(n_labels=10)), task, dataset=dataset)
