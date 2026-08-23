"""Strided perplexity -- the cheap screening metric.

Perplexity is the roadmap's first filter: it costs one forward pass per window
and catches a broken quantisation immediately, long before an expensive
deployment benchmark is worth running.

Two implementation details matter for trustworthiness:

* **Strided windows.** Tokens are scored with as much left context as the window
  allows, and each token is scored exactly once. Naive chunking scores the first
  tokens of every chunk with no context and inflates perplexity.
* **Chunked cross-entropy.** Logits for a 2048-token window over a 150k
  vocabulary are >1 GiB. The fp32 upcast for the loss is done in slices so peak
  memory stays bounded on small GPUs.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from ..config import PerplexityTask
from ..models.loader import LoadedModel
from ..results import MetricValue, TaskResult
from .datasets import TextCorpus, load_text_corpus

IGNORE_INDEX = -100
LOSS_CHUNK = 512
"""Positions per cross-entropy slice. Bounds the fp32 logit upcast."""

ProgressFn = Callable[[int, int], None]


@dataclass
class _NLLAccumulator:
    """Running sum/variance of per-token negative log-likelihood.

    Kept in float64 on CPU: summing hundreds of thousands of fp32 values loses
    precision, and the whole point of the metric is that it is stable enough to
    compare across runs.
    """

    total_nll: float = 0.0
    total_sq: float = 0.0
    n_tokens: int = 0

    def update(self, values: torch.Tensor) -> None:
        values = values.detach().to(torch.float64).cpu()
        self.total_nll += float(values.sum())
        self.total_sq += float((values**2).sum())
        self.n_tokens += int(values.numel())

    @property
    def mean(self) -> float:
        return self.total_nll / self.n_tokens if self.n_tokens else float("nan")

    @property
    def stderr(self) -> float | None:
        """Standard error of the mean NLL.

        Tokens are correlated, so this understates the true uncertainty. It is
        still the right order of magnitude for deciding whether a candidate's
        regression is noise.
        """
        if self.n_tokens < 2:
            return None
        variance = (self.total_sq - self.total_nll**2 / self.n_tokens) / (self.n_tokens - 1)
        return math.sqrt(max(variance, 0.0) / self.n_tokens)


def _windows(n_tokens: int, max_length: int, stride: int) -> list[tuple[int, int, int]]:
    """Plan the strided pass.

    Yields ``(begin, end, n_scored)`` where ``n_scored`` is how many trailing
    tokens of the window contribute to the metric. Earlier tokens in the window
    exist only to give those scored tokens context.

    Every token except the very first is scored exactly once, so the plan always
    accounts for ``n_tokens - 1`` predictions. Token 0 is never scored: nothing
    precedes it to predict it from.

    The step is capped at ``max_length - 1`` rather than ``max_length``. A step
    equal to the window would make each window begin exactly where its scored
    region begins, leaving that first token with no context and quietly dropping
    it from the average -- one lost token per window, and an inflated score for
    the ones next to it.
    """
    if n_tokens < 2:
        return []

    step = min(stride, max_length - 1)
    plan: list[tuple[int, int, int]] = []

    end = min(max_length, n_tokens)
    plan.append((0, end, end - 1))
    prev_end = end

    while prev_end < n_tokens:
        end = min(prev_end + step, n_tokens)
        begin = max(0, end - max_length)
        plan.append((begin, end, end - prev_end))
        prev_end = end

    return plan


def _batches(plan: list[tuple[int, int, int]], batch_size: int) -> list[list[tuple[int, int, int]]]:
    """Group windows into equal-length batches.

    Windows of different lengths cannot share a batch without padding, and the
    padded positions would need masking. Since only the first and last windows
    can differ in length, grouping by length costs almost nothing -- and unlike
    filtering, it never drops a window.
    """
    batches: list[list[tuple[int, int, int]]] = []
    current: list[tuple[int, int, int]] = []

    for window in plan:
        length = window[1] - window[0]
        if current and (len(current) >= batch_size or (current[0][1] - current[0][0]) != length):
            batches.append(current)
            current = []
        current.append(window)

    if current:
        batches.append(current)
    return batches


def _chunked_nll(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Per-token NLL for scored positions, computed in memory-bounded slices."""
    shift_logits = logits[:, :-1, :]
    shift_labels = labels[:, 1:]

    collected: list[torch.Tensor] = []
    for start in range(0, shift_labels.size(1), LOSS_CHUNK):
        stop = start + LOSS_CHUNK
        chunk_labels = shift_labels[:, start:stop].reshape(-1)
        if not bool((chunk_labels != IGNORE_INDEX).any()):
            continue
        chunk_logits = shift_logits[:, start:stop, :].reshape(-1, shift_logits.size(-1))
        losses = F.cross_entropy(
            chunk_logits.float(),
            chunk_labels,
            reduction="none",
            ignore_index=IGNORE_INDEX,
        )
        collected.append(losses[chunk_labels != IGNORE_INDEX])

    if not collected:
        return torch.empty(0)
    return torch.cat(collected)


@torch.inference_mode()
def evaluate_perplexity(
    handle: LoadedModel,
    task: PerplexityTask,
    *,
    corpus: TextCorpus | None = None,
    progress: ProgressFn | None = None,
) -> TaskResult:
    """Score a corpus and return token perplexity plus bits-per-byte."""
    started = time.perf_counter()
    corpus = corpus or load_text_corpus(task.dataset)

    max_length = task.max_length or handle.context_length
    stride = task.stride or max_length
    if stride > max_length:
        raise ValueError(f"stride ({stride}) must not exceed max_length ({max_length})")

    text = task.doc_separator.join(corpus.documents)
    encoded = handle.tokenizer(text, return_tensors="pt", add_special_tokens=False)
    input_ids = encoded["input_ids"][0]
    n_tokens = int(input_ids.numel())
    if n_tokens < 2:
        raise ValueError(f"{corpus.source}: corpus tokenises to {n_tokens} tokens, need >= 2")

    plan = _windows(n_tokens, max_length, stride)
    accumulator = _NLLAccumulator()

    n_done = 0
    for batch in _batches(plan, task.batch_size):
        ids = torch.stack([input_ids[b:e] for b, e, _ in batch]).to(handle.device)
        labels = ids.clone()
        for row, (_, _, n_scored) in enumerate(batch):
            # Everything before the scored tail exists only as context.
            labels[row, : labels.size(1) - n_scored] = IGNORE_INDEX

        logits = handle.model(input_ids=ids).logits
        accumulator.update(_chunked_nll(logits, labels))
        del logits

        n_done += len(batch)
        if progress is not None:
            progress(n_done, len(plan))

    mean_nll = accumulator.mean
    perplexity = math.exp(mean_nll)
    nll_stderr = accumulator.stderr

    metrics = [
        MetricValue(
            name="perplexity",
            value=perplexity,
            higher_is_better=False,
            # Delta method: sd(exp(x)) ~= exp(mean) * sd(x)
            stderr=perplexity * nll_stderr if nll_stderr is not None else None,
        ),
        MetricValue(
            name="nll_per_token",
            value=mean_nll,
            higher_is_better=False,
            stderr=nll_stderr,
        ),
    ]

    # Bits per byte is tokenizer-independent, so it stays meaningful when a
    # candidate ships a different tokenizer than the baseline.
    if corpus.n_bytes:
        metrics.append(
            MetricValue(
                name="bits_per_byte",
                value=accumulator.total_nll / (corpus.n_bytes * math.log(2)),
                higher_is_better=False,
                unit="bits/byte",
            )
        )

    return TaskResult(
        name=task.name,
        kind=task.kind,
        metrics=metrics,
        n_samples=corpus.n_documents,
        n_tokens=accumulator.n_tokens,
        duration_s=time.perf_counter() - started,
        dataset_fingerprint=corpus.fingerprint,
        details={
            "source": corpus.source,
            "max_length": max_length,
            "stride": stride,
            "n_windows": len(plan),
            "n_corpus_tokens": n_tokens,
            "n_corpus_bytes": corpus.n_bytes,
        },
    )


def summarise(result: TaskResult) -> str:
    ppl = result.metric("perplexity")
    return f"{result.name}: ppl={ppl.format() if ppl else 'n/a'} over {result.n_tokens} tokens"


__all__ = ["evaluate_perplexity", "summarise"]
