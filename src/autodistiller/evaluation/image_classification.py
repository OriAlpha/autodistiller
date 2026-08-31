"""Top-1 and top-5 accuracy for an image classifier.

The vision counterpart of :mod:`.multiple_choice`: one forward pass per image,
argmax over the label logits, no sampling, so the result is exactly
reproducible. What makes it a different module rather than a branch is the
input side -- an image goes through a processor whose resize, crop and
normalisation are part of the checkpoint, not a formatting detail.

Two metrics, both standard in the literature so the number can be checked
against a published one rather than only against itself:

    acc       top-1: the model's best guess is the label
    acc_top5  the label is somewhere in its best five
"""

from __future__ import annotations

import io
import math
import time
from collections.abc import Callable

import torch

from ..config import ImageClassificationTask
from ..models.loader import LoadedModel
from ..results import MetricValue, TaskResult
from .datasets import ImageSet, load_image_classification

ProgressFn = Callable[[int, int], None]

TOP_K = 5
"""The second metric's cut-off. Five, because that is what ImageNet reports."""


def _decode(blob: bytes):
    from PIL import Image

    # RGB unconditionally: ImageNet holds a few thousand greyscale and CMYK
    # JPEGs, and a processor handed a single-channel image builds a tensor the
    # patch projection cannot read.
    return Image.open(io.BytesIO(blob)).convert("RGB")


@torch.inference_mode()
def _predict_batch(handle: LoadedModel, blobs: list[bytes], top_k: int) -> torch.Tensor:
    """Indices of each image's ``top_k`` highest-scoring classes, best first."""
    inputs = handle.tokenizer(images=[_decode(b) for b in blobs], return_tensors="pt")
    inputs = {key: value.to(handle.device) for key, value in inputs.items()}
    # The processor produces float32; the model may hold bf16 weights.
    if "pixel_values" in inputs:
        inputs["pixel_values"] = inputs["pixel_values"].to(handle.dtype)

    logits = handle.model(**inputs).logits.float()
    return logits.topk(min(top_k, logits.shape[-1]), dim=-1).indices.cpu()


def evaluate_image_classification(
    handle: LoadedModel,
    task: ImageClassificationTask,
    *,
    dataset: ImageSet | None = None,
    progress: ProgressFn | None = None,
) -> TaskResult:
    started = time.perf_counter()
    dataset = dataset or load_image_classification(
        task.dataset,
        image_column=task.image_column,
        label_column=task.label_column,
    )

    n_model_classes = int(getattr(handle.model.config, "num_labels", 0) or 0)
    highest_label = max(dataset.labels)
    if n_model_classes and highest_label >= n_model_classes:
        # The dataset's label space is not the model's. Scored anyway, every
        # out-of-range image counts wrong, and the accuracy would look like a
        # broken model rather than a mismatched pair of them.
        raise ValueError(
            f"{dataset.source}: label {highest_label} is out of range for a model with "
            f"{n_model_classes} classes. The dataset and the checkpoint disagree about "
            f"what the labels mean, so the accuracy would be measuring the mismatch."
        )

    n_correct = 0
    n_correct_top5 = 0
    n_images = 0

    for start in range(0, dataset.n_examples, task.batch_size):
        blobs = dataset.images[start : start + task.batch_size]
        labels = dataset.labels[start : start + task.batch_size]
        predictions = _predict_batch(handle, blobs, TOP_K)

        for row, label in enumerate(labels):
            ranked = predictions[row].tolist()
            n_correct += int(ranked[0] == label)
            n_correct_top5 += int(label in ranked)

        n_images += len(blobs)
        if progress is not None:
            progress(min(start + len(blobs), dataset.n_examples), dataset.n_examples)

    n = dataset.n_examples

    def _binomial_stderr(correct: int) -> float:
        p = correct / n
        return math.sqrt(max(p * (1 - p), 0.0) / n)

    metrics = [
        MetricValue(
            name="acc",
            value=n_correct / n,
            higher_is_better=True,
            stderr=_binomial_stderr(n_correct),
        ),
        MetricValue(
            name=f"acc_top{TOP_K}",
            value=n_correct_top5 / n,
            higher_is_better=True,
            stderr=_binomial_stderr(n_correct_top5),
        ),
    ]

    return TaskResult(
        name=task.name,
        kind=task.kind,
        metrics=metrics,
        n_samples=n,
        # Images, not tokens. The field counts work done and every image is the
        # same fixed number of patches, so the two are one multiplication apart.
        n_tokens=n_images,
        duration_s=time.perf_counter() - started,
        dataset_fingerprint=dataset.fingerprint,
        details={
            "source": dataset.source,
            "n_correct": n_correct,
            f"n_correct_top{TOP_K}": n_correct_top5,
            "n_classes_in_data": dataset.metadata.get("n_classes"),
            "n_classes_in_model": n_model_classes,
            "batch_size": task.batch_size,
        },
    )


__all__ = ["evaluate_image_classification"]
