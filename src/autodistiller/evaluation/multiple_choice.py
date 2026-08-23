"""Log-likelihood multiple choice.

Perplexity screens cheaply but it is not what users care about. This evaluator
measures a real downstream signal: for each example every candidate answer is
scored by the log-probability the model assigns it, and the highest-scoring one
is the model's answer. No generation, no sampling, so the result is exactly
reproducible.

Two metrics are reported, both standard in the evaluation literature:

    acc       argmax over summed log-probability
    acc_norm  argmax over log-probability per character, so long answers are
              not penalized simply for having more tokens
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from ..config import MultipleChoiceTask
from ..models.loader import LoadedModel
from ..results import MetricValue, TaskResult
from .datasets import MultipleChoiceSet, load_multiple_choice

ProgressFn = Callable[[int, int], None]


@dataclass
class _Request:
    """One (context, choice) pair to score."""

    example_index: int
    choice_index: int
    input_ids: list[int]
    n_continuation: int
    n_choice_chars: int


def _encode_pair(
    tokenizer, context: str, continuation: str, max_length: int
) -> tuple[list[int], int]:
    """Tokenise ``context + continuation`` and report how many tokens the tail owns.

    Encoding the concatenation (rather than the two halves separately) keeps any
    token that merges across the boundary attributed the way a real prompt would
    tokenize it.
    """
    context_ids = tokenizer(context, add_special_tokens=True)["input_ids"]
    full_ids = tokenizer(context + continuation, add_special_tokens=True)["input_ids"]

    n_continuation = len(full_ids) - len(context_ids)
    if n_continuation <= 0:
        # Degenerate tokenization (e.g. the continuation merged entirely into the
        # last context token). Fall back to scoring the final token.
        n_continuation = 1

    if len(full_ids) > max_length:
        # Truncate from the left: the answer must survive, the context need not.
        full_ids = full_ids[-max_length:]
        n_continuation = min(n_continuation, len(full_ids) - 1)

    return full_ids, max(n_continuation, 1)


@torch.inference_mode()
def _score_batch(handle: LoadedModel, batch: list[_Request]) -> list[float]:
    """Summed log-probability of each request's continuation tokens."""
    max_len = max(len(r.input_ids) for r in batch)
    pad_id = handle.tokenizer.pad_token_id or 0

    input_ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((len(batch), max_len), dtype=torch.long)
    for row, request in enumerate(batch):
        length = len(request.input_ids)
        # Right padding: with a causal mask, padding after the real tokens
        # cannot influence the positions being scored.
        input_ids[row, :length] = torch.tensor(request.input_ids, dtype=torch.long)
        attention_mask[row, :length] = 1

    input_ids = input_ids.to(handle.device)
    logits = handle.model(
        input_ids=input_ids, attention_mask=attention_mask.to(handle.device)
    ).logits

    scores: list[float] = []
    for row, request in enumerate(batch):
        length = len(request.input_ids)
        start = length - request.n_continuation
        # Position i predicts token i+1, so the logits for the continuation
        # start one step earlier.
        window = logits[row, start - 1 : length - 1, :].float()
        targets = input_ids[row, start:length]
        log_probs = F.log_softmax(window, dim=-1)
        scores.append(float(log_probs.gather(-1, targets.unsqueeze(-1)).sum()))

    del logits
    return scores


def evaluate_multiple_choice(
    handle: LoadedModel,
    task: MultipleChoiceTask,
    *,
    dataset: MultipleChoiceSet | None = None,
    progress: ProgressFn | None = None,
) -> TaskResult:
    started = time.perf_counter()
    dataset = dataset or load_multiple_choice(
        task.dataset,
        context_column=task.context_column,
        choices_column=task.choices_column,
        answer_column=task.answer_column,
    )

    max_length = handle.context_length

    requests: list[_Request] = []
    for example_index, example in enumerate(dataset.examples):
        for choice_index, choice in enumerate(example.choices):
            ids, n_continuation = _encode_pair(
                handle.tokenizer, example.context, choice, max_length
            )
            requests.append(
                _Request(
                    example_index=example_index,
                    choice_index=choice_index,
                    input_ids=ids,
                    n_continuation=n_continuation,
                    n_choice_chars=max(len(choice), 1),
                )
            )

    # Longest-first batching keeps padding waste low without changing results.
    order = sorted(range(len(requests)), key=lambda i: -len(requests[i].input_ids))
    scores: list[float] = [0.0] * len(requests)
    n_tokens = 0

    for batch_start in range(0, len(order), task.batch_size):
        indices = order[batch_start : batch_start + task.batch_size]
        batch = [requests[i] for i in indices]
        for position, score in zip(indices, _score_batch(handle, batch), strict=True):
            scores[position] = score
        n_tokens += sum(len(r.input_ids) for r in batch)
        if progress is not None:
            progress(min(batch_start + len(batch), len(order)), len(order))

    # Collect per-example choice scores and pick winners.
    per_example: list[list[tuple[int, float, int]]] = [[] for _ in dataset.examples]
    for request, score in zip(requests, scores, strict=True):
        per_example[request.example_index].append(
            (request.choice_index, score, request.n_choice_chars)
        )

    n_correct = 0
    n_correct_norm = 0
    for example, entries in zip(dataset.examples, per_example, strict=True):
        entries.sort(key=lambda e: e[0])
        best = max(entries, key=lambda e: e[1])[0]
        best_norm = max(entries, key=lambda e: e[1] / e[2])[0]
        n_correct += int(best == example.answer_index)
        n_correct_norm += int(best_norm == example.answer_index)

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
            name="acc_norm",
            value=n_correct_norm / n,
            higher_is_better=True,
            stderr=_binomial_stderr(n_correct_norm),
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
            "n_requests": len(requests),
            "n_correct": n_correct,
            "n_correct_norm": n_correct_norm,
            "max_length": max_length,
        },
    )


__all__ = ["evaluate_multiple_choice"]
