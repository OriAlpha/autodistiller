"""Sentence-embedding quality.

An encoder has no next-token distribution, so perplexity and log-likelihood
scoring have nothing to work with. What it does have is a vector, and the
question worth asking of a vector is whether it puts things a human called
similar close together.

That is a rank correlation between cosine similarity and human scores. Two
properties make it the right screening metric here:

* **Absolute.** A score for one model, not a comparison between two, so records
  stay independently cacheable and retention is a ratio like every other task's.
* **Scale-free.** Correlating ranks means the human scale never matters, and
  quantization shifting every cosine slightly in the same direction -- which it
  does -- costs nothing it should not.

Spearman rather than Pearson because the relationship between cosine and human
judgement is monotonic but not linear; Pearson is reported alongside because the
embedding literature quotes both.
"""

from __future__ import annotations

import json
import logging
import math
import time
from pathlib import Path

import numpy as np
import torch

from ..config import EmbeddingTask
from ..results import MetricValue, TaskResult
from .datasets import SentencePairSet, load_sentence_pairs

logger = logging.getLogger(__name__)

DEFAULT_POOLING = "mean"

POOLING_CONFIG = "1_Pooling/config.json"
"""Where a sentence-transformers model records how it was trained to pool.

Worth reading rather than defaulting, because the serving runtime reads it too:
vLLM's pooling server reported ``seq_pooling_type='CLS'`` for bge-small while
this defaulted to mean. The gap measured within noise on that model -- 0.8954
against 0.8893 on STS-B, against a combined standard error of 0.0074 -- but a
screening number and a deployment that pool differently are not describing the
same model, and there is no reason to leave that to luck.
"""


def detect_pooling(model_id: str) -> str:
    """How this model was trained to pool, or the default when it does not say.

    Checks a local directory first, then the hub. Any failure is an absent
    answer rather than an error: an unreachable hub should not stop an
    evaluation that can proceed on the conventional default.
    """
    local = Path(model_id) / POOLING_CONFIG
    payload: dict | None = None

    if local.is_file():
        try:
            payload = json.loads(local.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            payload = None
    else:
        try:
            from huggingface_hub import hf_hub_download

            downloaded = hf_hub_download(model_id, POOLING_CONFIG)
            payload = json.loads(Path(downloaded).read_text(encoding="utf-8"))
        except Exception as exc:
            logger.debug("no pooling config for %s: %s", model_id, exc)

    if not payload:
        return DEFAULT_POOLING
    if payload.get("pooling_mode_cls_token"):
        return "cls"
    if payload.get("pooling_mode_mean_tokens"):
        return "mean"
    return DEFAULT_POOLING


def _average_ranks(values: np.ndarray) -> np.ndarray:
    """Ranks with ties averaged.

    Ties are not an edge case here: human similarity scores land on a coarse
    grid, so a dataset of 1500 pairs holds only a few dozen distinct values.
    Breaking ties by position would rank those arbitrarily and quietly deflate
    the correlation.
    """
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=np.float64)
    sorted_values = values[order]

    start = 0
    for index in range(1, len(values) + 1):
        if index == len(values) or sorted_values[index] != sorted_values[start]:
            ranks[order[start:index]] = (start + index - 1) / 2.0
            start = index
    return ranks


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = a - a.mean()
    b = b - b.mean()
    denominator = math.sqrt(float((a * a).sum()) * float((b * b).sum()))
    # Constant input: every value identical, so there is no variation to
    # correlate. Zero is the honest answer, not a division by zero.
    return float((a * b).sum() / denominator) if denominator else 0.0


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Rank correlation: Pearson over average ranks."""
    return _pearson(_average_ranks(a), _average_ranks(b))


def correlation_stderr(r: float, n: int) -> float | None:
    """Large-sample standard error of a correlation coefficient.

    ``(1 - r^2) / sqrt(n - 1)``. The whole tool turns on whether a difference
    between two models is real, so a correlation reported without one is a
    number nobody can act on.

    # ponytail: the large-sample approximation, not a bootstrap. It is symmetric
    # and so overstates the interval as r approaches 1; bootstrap it if a
    # decision ever turns on the fourth decimal.
    """
    return (1.0 - r * r) / math.sqrt(n - 1) if n > 1 else None


def pool(hidden: torch.Tensor, attention_mask: torch.Tensor, how: str) -> torch.Tensor:
    """Token vectors to one sentence vector."""
    if how == "cls":
        return hidden[:, 0]
    mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
    # Padding positions carry whatever the model produced for a pad token.
    # Averaging them in makes a sentence's vector depend on the longest sentence
    # batched with it, which is a silent batch-size dependence in the score.
    return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)


def encode(
    handle,
    texts: list[str],
    *,
    batch_size: int,
    max_length: int,
    pooling: str,
) -> tuple[torch.Tensor, int]:
    """Encode texts to L2-normalized vectors. Returns them and the token count."""
    vectors: list[torch.Tensor] = []
    n_tokens = 0

    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        encoded = handle.tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        encoded = {k: v.to(handle.device) for k, v in encoded.items()}
        n_tokens += int(encoded["attention_mask"].sum())

        with torch.no_grad():
            # Hidden states are requested rather than assumed: an encoder loaded
            # through AutoModel returns last_hidden_state, a decoder loaded as a
            # causal LM returns logits and no states at all unless asked. Asking
            # makes the same code path work for a decoder-based embedder.
            output = handle.model(**encoded, output_hidden_states=True)

        hidden = getattr(output, "last_hidden_state", None)
        if hidden is None:
            hidden = output.hidden_states[-1]

        pooled = pool(hidden, encoded["attention_mask"], pooling)
        # Normalizing here makes cosine similarity a dot product, and keeps the
        # comparison independent of vector magnitude -- which quantization does
        # move, without moving direction.
        vectors.append(torch.nn.functional.normalize(pooled.float(), dim=-1).cpu())

    return torch.cat(vectors), n_tokens


def evaluate_embedding(
    handle,
    task: EmbeddingTask,
    *,
    dataset: SentencePairSet | None = None,
    progress=None,
) -> TaskResult:
    """Score an embedding model on sentence-pair similarity."""
    started = time.perf_counter()
    dataset = dataset or load_sentence_pairs(
        task.dataset,
        text_a_column=task.text_a_column,
        text_b_column=task.text_b_column,
        score_column=task.score_column,
    )

    max_length = task.max_length or handle.context_length
    # Unset means "whatever this model was trained for", which is also what the
    # serving runtime will do with it.
    pooling = task.pooling or detect_pooling(handle.info.id)
    examples = dataset.examples

    # Both sides in one pass so every sentence meets the same batching, and so
    # the two halves cannot drift apart through padding.
    texts = [e.text_a for e in examples] + [e.text_b for e in examples]
    vectors, n_tokens = encode(
        handle,
        texts,
        batch_size=task.batch_size,
        max_length=max_length,
        pooling=pooling,
    )
    if progress is not None:
        progress(len(examples), len(examples))

    half = len(examples)
    cosines = (vectors[:half] * vectors[half:]).sum(dim=-1).numpy()
    human = np.array([e.score for e in examples], dtype=np.float64)

    rho = spearman(cosines, human)
    r = _pearson(cosines, human)
    n = len(examples)

    metrics = [
        MetricValue(
            name="spearman",
            value=rho,
            higher_is_better=True,
            stderr=correlation_stderr(rho, n),
        ),
        MetricValue(
            name="pearson",
            value=r,
            higher_is_better=True,
            stderr=correlation_stderr(r, n),
        ),
    ]

    return TaskResult(
        name=task.name,
        kind=task.kind,
        metrics=metrics,
        n_samples=n,
        n_tokens=n_tokens,
        duration_s=time.perf_counter() - started,
        dataset_fingerprint=dataset.fingerprint,
        details={
            "source": dataset.source,
            "pooling": pooling,
            "max_length": max_length,
            "mean_cosine": float(cosines.mean()),
            "embedding_dim": int(vectors.shape[-1]),
        },
    )


__all__ = [
    "DEFAULT_POOLING",
    "correlation_stderr",
    "detect_pooling",
    "encode",
    "evaluate_embedding",
    "pool",
    "spearman",
]
