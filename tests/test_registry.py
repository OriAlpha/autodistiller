"""Task resolution, including the Windows-path edge case in the ``kind:path``
syntax (``ppl:D:\\data\\corpus.txt`` has a colon in the path too).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from autodistiller.config import MultipleChoiceTask, PerplexityTask
from autodistiller.evaluation.registry import PRESETS, resolve_task, resolve_tasks


def test_presets_are_all_resolvable():
    for name in PRESETS:
        task = resolve_task(name)
        assert task.name == name
        assert task.kind in {"perplexity", "multiple_choice", "embedding", "retrieval"}


def test_preset_limit_is_applied():
    assert resolve_task("wikitext2", limit=7).dataset.limit == 7


def test_preset_has_a_default_screening_limit():
    """Phase 1 screens; it should not silently spend minutes on a full split."""
    assert resolve_task("wikitext2").dataset.limit is not None


def test_multiple_choice_presets_declare_a_preprocessor():
    for name in ("arc_easy", "arc_challenge", "hellaswag", "piqa"):
        assert resolve_task(name).preprocessor is not None


def test_local_text_corpus(text_corpus_file: Path):
    task = resolve_task(f"ppl:{text_corpus_file}")
    assert isinstance(task, PerplexityTask)
    assert task.dataset.source == "text"
    assert task.name == "corpus"


def test_local_jsonl_corpus(jsonl_corpus_file: Path):
    task = resolve_task(f"ppl:{jsonl_corpus_file}")
    assert task.dataset.source == "jsonl"


def test_local_multiple_choice(mc_dataset_file: Path):
    task = resolve_task(f"mc:{mc_dataset_file}")
    assert isinstance(task, MultipleChoiceTask)
    assert task.dataset.source == "jsonl"


def test_windows_drive_letters_survive_the_prefix_split(text_corpus_file: Path):
    """Only the first colon separates the kind from the path."""
    absolute = str(Path(text_corpus_file).resolve())
    task = resolve_task(f"ppl:{absolute}")
    assert Path(task.dataset.path).resolve() == Path(absolute)


def test_unknown_task_lists_the_presets():
    with pytest.raises(ValueError, match="wikitext2"):
        resolve_task("not_a_real_task")


def test_missing_file_is_reported():
    with pytest.raises(FileNotFoundError):
        resolve_task("ppl:definitely/not/here.txt")


def test_default_tasks_when_none_given():
    tasks = resolve_tasks(None)
    assert [t.name for t in tasks] == ["wikitext2"]


def test_duplicate_names_are_disambiguated(text_corpus_file: Path):
    """Two different files can share a stem; task names must stay unique
    because RunConfig rejects duplicates."""
    tasks = resolve_tasks([f"ppl:{text_corpus_file}", f"mc:{text_corpus_file}"])
    assert len({t.name for t in tasks}) == 2
