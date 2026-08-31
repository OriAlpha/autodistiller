"""Embedding evaluation.

The correlation is arithmetic and easy to check exactly. The part that fails
silently is pooling: a padding leak does not raise, it just makes a sentence's
vector depend on what was batched beside it, and the score drifts with batch
size for no visible reason.
"""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

import numpy as np
import pytest
import torch

from autodistiller.config import EmbeddingTask, ModelSpec
from autodistiller.evaluation.datasets import SentencePair, SentencePairSet
from autodistiller.evaluation.embedding import (
    correlation_stderr,
    evaluate_embedding,
    pool,
    spearman,
)

PAIRS = [
    SentencePair(text_a="a man is playing a guitar", text_b="a man plays a guitar", score=4.8),
    SentencePair(text_a="a dog runs in the park", text_b="a puppy is running outside", score=3.6),
    SentencePair(text_a="the stock market fell", text_b="a cat sat on the mat", score=0.2),
    SentencePair(text_a="she is cooking dinner", text_b="a woman prepares a meal", score=4.1),
    SentencePair(text_a="rain is falling", text_b="the engine needs oil", score=0.4),
]


def _pair_set() -> SentencePairSet:
    return SentencePairSet(examples=PAIRS, fingerprint="test", source="test:pairs")


# --- correlation --------------------------------------------------------


def test_spearman_is_one_for_a_monotonic_relationship():
    """Monotonic but not linear, which is exactly why it is not Pearson."""
    a = np.array([1.0, 2.0, 3.0, 4.0])
    b = np.array([1.0, 10.0, 100.0, 1000.0])

    assert spearman(a, b) == pytest.approx(1.0)
    assert spearman(a, -b) == pytest.approx(-1.0)


def test_spearman_averages_ties():
    """Human similarity scores land on a coarse grid, so ties are the norm.

    Breaking them by position would rank equal scores arbitrarily and quietly
    deflate the correlation.
    """
    a = np.array([1.0, 2.0, 2.0, 3.0])
    b = np.array([5.0, 6.0, 6.0, 7.0])

    assert spearman(a, b) == pytest.approx(1.0)


def test_spearman_of_constant_input_is_zero_not_a_crash():
    constant = np.array([2.0, 2.0, 2.0, 2.0])

    assert spearman(np.array([1.0, 2.0, 3.0, 4.0]), constant) == 0.0


def test_correlation_stderr_shrinks_with_samples():
    wide = correlation_stderr(0.9, 10)
    narrow = correlation_stderr(0.9, 1500)

    assert wide is not None and narrow is not None
    assert narrow < wide
    assert correlation_stderr(0.9, 1) is None


# --- pooling ------------------------------------------------------------


def test_mean_pooling_ignores_padding():
    """Otherwise a vector depends on the longest sentence batched with it.

    Nothing raises when this is wrong. The score just moves when batch size
    does, which is the hardest kind of measurement bug to notice.
    """
    hidden = torch.tensor([[[1.0, 1.0], [3.0, 3.0], [99.0, 99.0]]])
    mask = torch.tensor([[1, 1, 0]])

    pooled = pool(hidden, mask, "mean")

    assert torch.allclose(pooled, torch.tensor([[2.0, 2.0]]))


def test_cls_pooling_takes_the_first_token():
    hidden = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
    mask = torch.tensor([[1, 1]])

    assert torch.allclose(pool(hidden, mask, "cls"), torch.tensor([[1.0, 2.0]]))


# --- end to end ---------------------------------------------------------


def test_evaluate_embedding_scores_a_real_model(tiny_model_dir: Path):
    """Also covers the decoder path.

    The tiny fixture is a causal LM, which returns logits and no hidden states
    unless they are asked for -- so this is the check that a decoder-based
    embedder does not fall over on a missing ``last_hidden_state``.
    """
    from autodistiller.models.loader import loaded_model

    task = EmbeddingTask(name="pairs", dataset={"source": "jsonl", "path": "unused"}, batch_size=2)

    with loaded_model(ModelSpec(id=str(tiny_model_dir), device="cpu")) as handle:
        result = evaluate_embedding(handle, task, dataset=_pair_set())

    assert result.n_samples == len(PAIRS)
    assert result.n_tokens > 0
    assert {m.name for m in result.metrics} == {"spearman", "pearson"}

    for metric in result.metrics:
        assert -1.0 <= metric.value <= 1.0
        assert metric.stderr is not None
        assert metric.higher_is_better

    assert result.details["embedding_dim"] == 32


def test_encoders_load_without_a_causal_head():
    """AutoModelForCausalLM on BERT either refuses or attaches a random head."""
    from transformers import AutoModel, AutoModelForCausalLM

    from autodistiller.models.loader import auto_class_for

    class Encoder:
        architectures: ClassVar[list[str]] = ["BertModel"]

    class Decoder:
        architectures: ClassVar[list[str]] = ["Qwen3ForCausalLM"]

    class Unlabelled:
        architectures = None

    assert auto_class_for(Encoder()) is AutoModel
    assert auto_class_for(Decoder()) is AutoModelForCausalLM
    # No architectures at all is a local or hand-written config. Reading that
    # silence as "encoder" would change how every such checkpoint has loaded
    # since before embeddings existed, so the causal LM stays the default.
    assert auto_class_for(Unlabelled()) is AutoModelForCausalLM


def test_pooling_is_read_from_the_model_not_defaulted(tmp_path: Path):
    """The server reads this file; the screen has to read the same one.

    vLLM's pooling server reported CLS for bge-small while this defaulted to
    mean. That measured within noise on that model, but a screening number and
    a deployment that pool differently are not describing the same model.
    """
    import json

    from autodistiller.evaluation.embedding import DEFAULT_POOLING, detect_pooling

    cls_model = tmp_path / "cls-model"
    (cls_model / "1_Pooling").mkdir(parents=True)
    (cls_model / "1_Pooling" / "config.json").write_text(
        json.dumps({"pooling_mode_cls_token": True, "pooling_mode_mean_tokens": False})
    )

    mean_model = tmp_path / "mean-model"
    (mean_model / "1_Pooling").mkdir(parents=True)
    (mean_model / "1_Pooling" / "config.json").write_text(
        json.dumps({"pooling_mode_cls_token": False, "pooling_mode_mean_tokens": True})
    )

    assert detect_pooling(str(cls_model)) == "cls"
    assert detect_pooling(str(mean_model)) == "mean"
    # A model that says nothing gets the convention, not an error.
    assert detect_pooling(str(tmp_path / "nothing-here")) == DEFAULT_POOLING


def test_the_resolved_pooling_is_recorded(tiny_model_dir: Path):
    """An explicit choice overrides detection, and the record says which was used."""
    from autodistiller.models.loader import loaded_model

    task = EmbeddingTask(
        name="pairs",
        dataset={"source": "jsonl", "path": "unused"},
        batch_size=2,
        pooling="cls",
    )

    with loaded_model(ModelSpec(id=str(tiny_model_dir), device="cpu")) as handle:
        result = evaluate_embedding(handle, task, dataset=_pair_set())

    assert result.details["pooling"] == "cls"
