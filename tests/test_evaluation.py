"""Evaluation correctness.

The window planner gets its own tests because a subtle off-by-one there silently
scores tokens twice or skips them, which shifts perplexity without any error
being raised -- exactly the failure mode a "trustworthy baseline" must not have.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from autodistiller.config import DatasetSpec, ModelSpec, MultipleChoiceTask, PerplexityTask
from autodistiller.evaluation.datasets import (
    check_dataset_available,
    load_multiple_choice,
    load_text_corpus,
)
from autodistiller.evaluation.multiple_choice import evaluate_multiple_choice
from autodistiller.evaluation.perplexity import _batches, _windows, evaluate_perplexity
from autodistiller.evaluation.preprocessors import get_preprocessor
from autodistiller.models.loader import load_model

# --- window planning ----------------------------------------------------


@pytest.mark.parametrize(
    ("n_tokens", "max_length", "stride"),
    [(100, 10, 10), (100, 10, 5), (100, 32, 8), (7, 10, 10), (10, 10, 10), (101, 10, 10)],
)
def test_windows_score_every_token_exactly_once(n_tokens, max_length, stride):
    """Every token but the first is predicted exactly once."""
    plan = _windows(n_tokens, max_length, stride)
    assert sum(n_scored for _, _, n_scored in plan) == n_tokens - 1


@pytest.mark.parametrize(
    ("n_tokens", "max_length", "stride"),
    [(100, 10, 10), (100, 10, 5), (100, 32, 8), (10, 10, 10), (101, 10, 10)],
)
def test_every_scored_token_has_context(n_tokens, max_length, stride):
    """A scored token with no preceding context would be unpredictable by
    construction and would inflate the score."""
    for begin, end, n_scored in _windows(n_tokens, max_length, stride):
        assert n_scored <= end - begin - 1


def test_windows_cover_a_contiguous_range():
    plan = _windows(100, 32, 8)
    scored_end = 1
    for _, end, n_scored in plan:
        assert end - n_scored == scored_end
        scored_end = end
    assert scored_end == 100


def test_windows_on_a_corpus_shorter_than_the_window():
    assert _windows(5, 64, 64) == [(0, 5, 4)]


def test_windows_needs_at_least_two_tokens():
    assert _windows(1, 64, 64) == []


def test_windows_never_exceed_max_length():
    for begin, end, _ in _windows(1000, 64, 16):
        assert end - begin <= 64


def test_windows_scored_region_is_within_the_window():
    for begin, end, n_scored in _windows(500, 64, 16):
        assert n_scored <= end - begin


def test_smaller_stride_gives_scored_tokens_more_context():
    """A smaller stride buys more context per scored token, at more windows."""
    wide = _windows(100, 20, 19)
    narrow = _windows(100, 20, 5)
    assert len(narrow) > len(wide)

    # Each narrow window scores 5 tokens while seeing 20.
    begin, end, n_scored = narrow[1]
    assert end - begin == 20
    assert n_scored == 5


def test_batches_never_drop_a_window():
    """The last window is usually shorter; it must still be evaluated."""
    plan = _windows(100, 32, 8)
    batched = [w for batch in _batches(plan, 4) for w in batch]
    assert batched == plan


def test_batches_contain_only_equal_length_windows():
    for batch in _batches(_windows(1000, 64, 16), 8):
        assert len({end - begin for begin, end, _ in batch}) == 1
        assert len(batch) <= 8


# --- dataset loading ----------------------------------------------------


def test_load_text_corpus_from_file(text_corpus_file: Path):
    corpus = load_text_corpus(DatasetSpec(source="text", path=str(text_corpus_file)))
    assert corpus.n_documents == 1
    assert corpus.n_bytes > 0
    assert len(corpus.fingerprint) == 16


def test_load_jsonl_corpus_respects_limit(jsonl_corpus_file: Path):
    corpus = load_text_corpus(DatasetSpec(source="jsonl", path=str(jsonl_corpus_file), limit=2))
    assert corpus.n_documents == 2


def test_corpus_fingerprint_is_content_addressed(tmp_path: Path):
    first = tmp_path / "a.txt"
    second = tmp_path / "b.txt"
    first.write_text("identical content", encoding="utf-8")
    second.write_text("identical content", encoding="utf-8")

    a = load_text_corpus(DatasetSpec(source="text", path=str(first)))
    b = load_text_corpus(DatasetSpec(source="text", path=str(second)))
    assert a.fingerprint == b.fingerprint

    second.write_text("different content", encoding="utf-8")
    c = load_text_corpus(DatasetSpec(source="text", path=str(second)))
    assert c.fingerprint != a.fingerprint


def test_missing_column_names_the_available_ones(tmp_path: Path):
    path = tmp_path / "rows.jsonl"
    path.write_text(json.dumps({"body": "hello"}) + "\n", encoding="utf-8")
    with pytest.raises(KeyError, match="body"):
        load_text_corpus(DatasetSpec(source="jsonl", path=str(path), text_column="text"))


def test_blank_documents_are_dropped(tmp_path: Path):
    path = tmp_path / "rows.jsonl"
    path.write_text(
        "\n".join(json.dumps({"text": t}) for t in ["real text", "   ", "", "more text"]) + "\n",
        encoding="utf-8",
    )
    corpus = load_text_corpus(DatasetSpec(source="jsonl", path=str(path)))
    assert corpus.n_documents == 2


def test_load_multiple_choice(mc_dataset_file: Path):
    dataset = load_multiple_choice(DatasetSpec(source="jsonl", path=str(mc_dataset_file)))
    assert dataset.n_examples == 3
    assert dataset.examples[0].answer_index == 0


def test_multiple_choice_rejects_out_of_range_answer(tmp_path: Path):
    path = tmp_path / "bad.jsonl"
    path.write_text(
        json.dumps({"context": "q", "choices": [" a", " b"], "answer_index": 5}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="out of range"):
        load_multiple_choice(DatasetSpec(source="jsonl", path=str(path)))


def test_multiple_choice_rejects_single_choice(tmp_path: Path):
    path = tmp_path / "bad.jsonl"
    path.write_text(
        json.dumps({"context": "q", "choices": [" only"], "answer_index": 0}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match=">= 2 choices"):
        load_multiple_choice(DatasetSpec(source="jsonl", path=str(path)))


def test_invalid_json_reports_the_line_number(tmp_path: Path):
    path = tmp_path / "broken.jsonl"
    path.write_text('{"text": "fine"}\nnot json at all\n', encoding="utf-8")
    with pytest.raises(ValueError, match=":2:"):
        load_text_corpus(DatasetSpec(source="jsonl", path=str(path)))


# --- preprocessors ------------------------------------------------------


def test_arc_preprocessor_maps_label_to_index():
    row = {
        "id": "x",
        "question": "What is 2+2?",
        "choices": {"text": ["3", "4", "5"], "label": ["A", "B", "C"]},
        "answerKey": "B",
    }
    mapped = get_preprocessor("arc")(row)
    assert mapped["answer_index"] == 1
    assert mapped["choices"] == [" 3", " 4", " 5"]
    assert "What is 2+2?" in mapped["context"]


def test_unknown_preprocessor_lists_the_valid_ones():
    with pytest.raises(KeyError, match="available"):
        get_preprocessor("nope")


def test_no_preprocessor_is_a_passthrough():
    assert get_preprocessor(None) is None


# --- end to end on a real (tiny) model ---------------------------------


@pytest.fixture(scope="module")
def loaded_tiny(tiny_model_dir: Path):
    handle = load_model(ModelSpec(id=str(tiny_model_dir), device="cpu", dtype="float32"))
    yield handle
    handle.free()


def test_model_loading_records_provenance(loaded_tiny):
    info = loaded_tiny.info
    assert info.is_local is True
    assert info.n_parameters and info.n_parameters > 0
    assert info.context_length == 128
    assert info.dtype == "float32"
    assert len(info.architecture_fingerprint) == 16


def test_perplexity_is_finite_and_reproducible(loaded_tiny, text_corpus_file: Path):
    task = PerplexityTask(
        name="ppl",
        dataset=DatasetSpec(source="text", path=str(text_corpus_file)),
        max_length=64,
        stride=32,
    )
    first = evaluate_perplexity(loaded_tiny, task)
    second = evaluate_perplexity(loaded_tiny, task)

    ppl = first.metric("perplexity")
    assert ppl is not None
    assert math.isfinite(ppl.value) and ppl.value > 1.0
    assert first.n_tokens > 0
    assert first.details["n_windows"] >= 1

    # Same config, same weights, same data -> byte-identical metric.
    assert ppl.value == pytest.approx(second.metric("perplexity").value, rel=1e-12)


def test_perplexity_scores_every_corpus_token(loaded_tiny, text_corpus_file: Path):
    """Scored tokens = corpus tokens - 1 (the first token has nothing to predict it)."""
    task = PerplexityTask(
        name="ppl",
        dataset=DatasetSpec(source="text", path=str(text_corpus_file)),
        max_length=64,
        stride=64,
    )
    result = evaluate_perplexity(loaded_tiny, task)
    assert result.n_tokens == result.details["n_corpus_tokens"] - 1


def test_perplexity_reports_bits_per_byte(loaded_tiny, text_corpus_file: Path):
    task = PerplexityTask(
        name="ppl", dataset=DatasetSpec(source="text", path=str(text_corpus_file)), max_length=64
    )
    bpb = evaluate_perplexity(loaded_tiny, task).metric("bits_per_byte")
    assert bpb is not None and bpb.value > 0


def test_perplexity_rejects_stride_larger_than_window(loaded_tiny, text_corpus_file: Path):
    task = PerplexityTask(
        name="ppl",
        dataset=DatasetSpec(source="text", path=str(text_corpus_file)),
        max_length=32,
        stride=64,
    )
    with pytest.raises(ValueError, match="must not exceed"):
        evaluate_perplexity(loaded_tiny, task)


def test_batched_perplexity_matches_unbatched(loaded_tiny, text_corpus_file: Path):
    """Batching is an optimisation; it must not move the number."""
    dataset = DatasetSpec(source="text", path=str(text_corpus_file))
    single = evaluate_perplexity(
        loaded_tiny,
        PerplexityTask(name="ppl", dataset=dataset, max_length=32, stride=32, batch_size=1),
    )
    batched = evaluate_perplexity(
        loaded_tiny,
        PerplexityTask(name="ppl", dataset=dataset, max_length=32, stride=32, batch_size=4),
    )
    assert single.n_tokens == batched.n_tokens
    assert single.metric("perplexity").value == pytest.approx(
        batched.metric("perplexity").value, rel=1e-4
    )


def test_multiple_choice_produces_bounded_accuracy(loaded_tiny, mc_dataset_file: Path):
    task = MultipleChoiceTask(
        name="mc", dataset=DatasetSpec(source="jsonl", path=str(mc_dataset_file))
    )
    result = evaluate_multiple_choice(loaded_tiny, task)

    acc = result.metric("acc")
    acc_norm = result.metric("acc_norm")
    assert acc is not None and 0.0 <= acc.value <= 1.0
    assert acc_norm is not None and 0.0 <= acc_norm.value <= 1.0
    assert result.n_samples == 3
    assert result.details["n_requests"] == 6


def test_batched_multiple_choice_matches_unbatched(loaded_tiny, mc_dataset_file: Path):
    """Padding must not leak into the scored positions."""
    dataset = DatasetSpec(source="jsonl", path=str(mc_dataset_file))
    single = evaluate_multiple_choice(
        loaded_tiny, MultipleChoiceTask(name="mc", dataset=dataset, batch_size=1)
    )
    batched = evaluate_multiple_choice(
        loaded_tiny, MultipleChoiceTask(name="mc", dataset=dataset, batch_size=6)
    )
    assert single.metric("acc").value == pytest.approx(batched.metric("acc").value)
    assert single.metric("acc_norm").value == pytest.approx(batched.metric("acc_norm").value)


# --- pre-flight ---------------------------------------------------------


def test_preflight_rejects_bare_hub_dataset_ids():
    """datasets v3+ dropped un-namespaced canonical ids and fails with an
    opaque URI error deep inside the library."""
    with pytest.raises(ValueError, match="namespaced"):
        check_dataset_available(DatasetSpec(source="hub", path="wikitext"))


def test_preflight_accepts_namespaced_hub_ids():
    check_dataset_available(DatasetSpec(source="hub", path="Salesforce/wikitext"))


def test_preflight_rejects_missing_local_files():
    with pytest.raises(FileNotFoundError):
        check_dataset_available(DatasetSpec(source="text", path="no/such/file.txt"))


def test_preflight_accepts_existing_local_files(text_corpus_file: Path):
    check_dataset_available(DatasetSpec(source="text", path=str(text_corpus_file)))
