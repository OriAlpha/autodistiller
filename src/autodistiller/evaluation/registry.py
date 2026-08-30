"""Built-in task presets and the ``--task`` mini-syntax.

Presets exist so the common case is one flag. The syntax also accepts local
files, because a user's own eval set is usually more informative about their
deployment than any public benchmark:

===========================  =================================================
``wikitext2``                built-in preset
``ppl:data/domain.txt``      perplexity over a local text file
``ppl:data/docs.jsonl``      perplexity over a local JSONL corpus
``mc:data/my_evals.jsonl``   multiple choice over a local JSONL file
===========================  =================================================
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Literal

from ..config import DatasetSpec, EmbeddingTask, MultipleChoiceTask, PerplexityTask, TaskSpec

PresetFactory = Callable[[int | None], TaskSpec]

DEFAULT_SCREENING_LIMIT = 256
"""Documents used when a preset is asked for without an explicit limit.

Phase 1 is a screening stage. A full wikitext-2 pass is minutes of GPU time that
buys very little extra signal about whether a baseline is sane.
"""


def _stsb(limit: int | None) -> TaskSpec:
    """STS-B: the standard sentence-similarity benchmark.

    1500 validation pairs, so a full pass is seconds even on a small GPU and
    the default limit leaves it whole. Capping it would only widen the error
    bar on a correlation that is already the cheapest task here.
    """
    return EmbeddingTask(
        name="stsb",
        dataset=DatasetSpec(
            source="hub",
            path="nyu-mll/glue",
            name="stsb",
            split="validation",
            limit=limit,
        ),
        score_column="label",
    )


def _wikitext2(limit: int | None) -> TaskSpec:
    return PerplexityTask(
        name="wikitext2",
        dataset=DatasetSpec(
            source="hub",
            path="Salesforce/wikitext",
            name="wikitext-2-raw-v1",
            split="test",
            text_column="text",
            limit=limit or DEFAULT_SCREENING_LIMIT,
        ),
    )


def _wikitext103(limit: int | None) -> TaskSpec:
    return PerplexityTask(
        name="wikitext103",
        dataset=DatasetSpec(
            source="hub",
            path="Salesforce/wikitext",
            name="wikitext-103-raw-v1",
            split="test",
            text_column="text",
            limit=limit or DEFAULT_SCREENING_LIMIT,
        ),
    )


def _arc_easy(limit: int | None) -> TaskSpec:
    return MultipleChoiceTask(
        name="arc_easy",
        dataset=DatasetSpec(
            source="hub",
            path="allenai/ai2_arc",
            name="ARC-Easy",
            split="test",
            limit=limit or DEFAULT_SCREENING_LIMIT,
        ),
        preprocessor="arc",
    )


def _arc_challenge(limit: int | None) -> TaskSpec:
    return MultipleChoiceTask(
        name="arc_challenge",
        dataset=DatasetSpec(
            source="hub",
            path="allenai/ai2_arc",
            name="ARC-Challenge",
            split="test",
            limit=limit or DEFAULT_SCREENING_LIMIT,
        ),
        preprocessor="arc",
    )


def _hellaswag(limit: int | None) -> TaskSpec:
    return MultipleChoiceTask(
        name="hellaswag",
        dataset=DatasetSpec(
            source="hub",
            path="Rowan/hellaswag",
            split="validation",
            limit=limit or DEFAULT_SCREENING_LIMIT,
        ),
        preprocessor="hellaswag",
    )


def _piqa(limit: int | None) -> TaskSpec:
    return MultipleChoiceTask(
        name="piqa",
        dataset=DatasetSpec(
            source="hub",
            path="ybisk/piqa",
            split="validation",
            limit=limit or DEFAULT_SCREENING_LIMIT,
        ),
        preprocessor="piqa",
    )


PRESETS: dict[str, PresetFactory] = {
    "wikitext2": _wikitext2,
    "wikitext103": _wikitext103,
    "arc_easy": _arc_easy,
    "arc_challenge": _arc_challenge,
    "hellaswag": _hellaswag,
    "piqa": _piqa,
    "stsb": _stsb,
}

DEFAULT_TASKS = ("wikitext2",)

_KIND_PREFIXES = {
    "ppl": "perplexity",
    "perplexity": "perplexity",
    "mc": "multiple_choice",
    "multiple_choice": "multiple_choice",
}


def _local_dataset_source(path: Path) -> Literal["jsonl", "text"]:
    return "jsonl" if path.suffix.lower() in {".jsonl", ".ndjson", ".json"} else "text"


def _task_name_from_path(path: Path) -> str:
    return path.stem.replace(" ", "_")


def resolve_task(expression: str, *, limit: int | None = None) -> TaskSpec:
    """Turn one ``--task`` value into a :class:`TaskSpec`."""
    expression = expression.strip()
    if not expression:
        raise ValueError("empty task expression")

    prefix, separator, target = expression.partition(":")
    kind = _KIND_PREFIXES.get(prefix.lower()) if separator else None

    if kind is None:
        if expression in PRESETS:
            return PRESETS[expression](limit)
        raise ValueError(
            f"unknown task {expression!r}. Use a preset ({', '.join(sorted(PRESETS))}) "
            f"or a prefixed path such as 'ppl:corpus.txt' / 'mc:evals.jsonl'."
        )

    path = Path(target)
    if not path.exists():
        raise FileNotFoundError(f"task {expression!r}: file not found: {path}")

    dataset = DatasetSpec(
        source=_local_dataset_source(path),
        path=str(path),
        split="test",
        limit=limit,
    )

    if kind == "perplexity":
        return PerplexityTask(name=_task_name_from_path(path), dataset=dataset)
    return MultipleChoiceTask(name=_task_name_from_path(path), dataset=dataset)


def resolve_tasks(expressions: list[str] | None, *, limit: int | None = None) -> list[TaskSpec]:
    """Resolve several task expressions, de-duplicating names."""
    tasks = [resolve_task(e, limit=limit) for e in (expressions or list(DEFAULT_TASKS))]

    seen: dict[str, int] = {}
    for task in tasks:
        if task.name in seen:
            seen[task.name] += 1
            task.name = f"{task.name}_{seen[task.name]}"
        else:
            seen[task.name] = 0
    return tasks


__all__ = ["DEFAULT_TASKS", "PRESETS", "resolve_task", "resolve_tasks"]
