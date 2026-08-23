"""Named row transforms for hub datasets.

Public multiple-choice datasets each ship their own schema. Rather than teach the
loader about every one, a task names a preprocessor and the loader resolves it
here. Names, not function objects, keep a ``RunConfig`` serialisable and hashable
-- which is what lets Phase 6 cache on it.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

RowTransform = Callable[[dict[str, Any]], dict[str, Any]]


def arc(row: dict[str, Any]) -> dict[str, Any]:
    """AI2 ARC: choices are a dict of parallel text/label lists, answer is a label."""
    choices = row["choices"]
    labels = [str(label) for label in choices["label"]]
    answer = str(row["answerKey"])
    return {
        "id": row.get("id"),
        "context": f"Question: {row['question']}\nAnswer:",
        "choices": [f" {text}" for text in choices["text"]],
        "answer_index": labels.index(answer),
    }


def hellaswag(row: dict[str, Any]) -> dict[str, Any]:
    """HellaSwag: sentence completion; the context is activity label + first sentence."""
    context = f"{row['activity_label']}: {row['ctx_a']} {row['ctx_b'].capitalize()}".strip()
    return {
        "id": row.get("ind"),
        "context": context,
        "choices": [f" {ending}" for ending in row["endings"]],
        "answer_index": int(row["label"]),
    }


def piqa(row: dict[str, Any]) -> dict[str, Any]:
    """PIQA: physical commonsense, two candidate solutions."""
    return {
        "id": row.get("id"),
        "context": f"Question: {row['goal']}\nAnswer:",
        "choices": [f" {row['sol1']}", f" {row['sol2']}"],
        "answer_index": int(row["label"]),
    }


PREPROCESSORS: dict[str, RowTransform] = {
    "arc": arc,
    "hellaswag": hellaswag,
    "piqa": piqa,
}


def get_preprocessor(name: str | None) -> RowTransform | None:
    if name is None:
        return None
    try:
        return PREPROCESSORS[name]
    except KeyError:
        raise KeyError(
            f"unknown preprocessor {name!r}; available: {sorted(PREPROCESSORS)}"
        ) from None


__all__ = ["PREPROCESSORS", "RowTransform", "get_preprocessor"]
