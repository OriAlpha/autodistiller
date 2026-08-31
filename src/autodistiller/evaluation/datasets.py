"""Evaluation data loading.

Hub datasets and user-supplied local files land in the same in-memory shapes, so
"task/custom evaluation datasets" is one code path rather than two. Every corpus
carries a content fingerprint: comparing a baseline against a candidate scored on
different data is the easiest way to draw a wrong conclusion, and the fingerprint
makes that detectable.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..config import DatasetSpec
from ..metadata.hashing import hash_obj, hash_text_stream


@dataclass
class TextCorpus:
    """Plain documents used for perplexity."""

    documents: list[str]
    fingerprint: str
    source: str

    @property
    def n_documents(self) -> int:
        return len(self.documents)

    @property
    def n_bytes(self) -> int:
        return sum(len(d.encode("utf-8")) for d in self.documents)


@dataclass
class MultipleChoiceExample:
    context: str
    choices: list[str]
    answer_index: int
    id: str | None = None

    def __post_init__(self) -> None:
        if len(self.choices) < 2:
            raise ValueError(f"example {self.id!r}: need >= 2 choices, got {len(self.choices)}")
        if not 0 <= self.answer_index < len(self.choices):
            raise ValueError(
                f"example {self.id!r}: answer_index {self.answer_index} "
                f"out of range for {len(self.choices)} choices"
            )


@dataclass
class MultipleChoiceSet:
    examples: list[MultipleChoiceExample]
    fingerprint: str
    source: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def n_examples(self) -> int:
        return len(self.examples)


@dataclass
class SentencePair:
    """Two sentences a human scored for similarity."""

    text_a: str
    text_b: str
    score: float
    id: str | None = None


@dataclass
class SentencePairSet:
    examples: list[SentencePair]
    fingerprint: str
    source: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def n_examples(self) -> int:
        return len(self.examples)


@dataclass
class RetrievalSet:
    """A corpus, queries, and which documents answer which query."""

    doc_ids: list[str]
    documents: list[str]
    query_ids: list[str]
    queries: list[str]
    relevance: dict[str, dict[str, float]]
    fingerprint: str
    source: str

    @property
    def n_examples(self) -> int:
        return len(self.queries)


def _rows(spec: DatasetSpec) -> list[dict[str, Any]]:
    if spec.source == "jsonl":
        path = Path(spec.path)
        if not path.exists():
            raise FileNotFoundError(f"dataset not found: {path}")
        return _read_jsonl(path, spec.limit)
    if spec.source == "text":
        raise ValueError("retrieval tasks need source 'jsonl' or 'hub', not 'text'")
    return _load_hub_rows(spec)


def load_retrieval(
    corpus: DatasetSpec,
    queries: DatasetSpec,
    qrels: DatasetSpec,
    *,
    doc_id_column: str = "_id",
    doc_text_column: str = "text",
    doc_title_column: str | None = "title",
    query_id_column: str = "_id",
    query_text_column: str = "text",
    qrel_query_column: str = "query-id",
    qrel_doc_column: str = "corpus-id",
    qrel_score_column: str = "score",
) -> RetrievalSet:
    """Load a BEIR-shaped retrieval benchmark.

    Only the queries that have a judgement are kept. A query with no relevant
    document scores zero however good the model is, so leaving them in measures
    the dataset's coverage rather than the model.
    """
    source = f"{corpus.source}:{corpus.path}"

    relevance: dict[str, dict[str, float]] = {}
    for row in _rows(qrels):
        score = float(_require_column(row, qrel_score_column, source))
        if score <= 0:
            continue
        query_id = str(_require_column(row, qrel_query_column, source))
        relevance.setdefault(query_id, {})[str(_require_column(row, qrel_doc_column, source))] = (
            score
        )

    if not relevance:
        raise ValueError(f"{source}: no relevance judgements found")

    doc_ids: list[str] = []
    documents: list[str] = []
    for row in _rows(corpus):
        doc_ids.append(str(_require_column(row, doc_id_column, source)))
        text = str(_require_column(row, doc_text_column, source))
        title = str(row.get(doc_title_column) or "") if doc_title_column else ""
        # Title first, the way BEIR concatenates them: it is often the most
        # retrievable sentence in the document.
        documents.append(f"{title} {text}".strip() if title else text)

    query_ids: list[str] = []
    texts: list[str] = []
    for row in _rows(queries):
        query_id = str(_require_column(row, query_id_column, source))
        if query_id not in relevance:
            continue
        query_ids.append(query_id)
        texts.append(str(_require_column(row, query_text_column, source)))

    if not documents or not texts:
        raise ValueError(f"{source}: corpus or queries came back empty")

    fingerprint = hash_obj(
        {
            "docs": hash_text_stream(documents),
            "queries": hash_text_stream(texts),
            "qrels": sorted((q, sorted(d)) for q, d in relevance.items()),
        }
    )
    return RetrievalSet(
        doc_ids=doc_ids,
        documents=documents,
        query_ids=query_ids,
        queries=texts,
        relevance=relevance,
        fingerprint=fingerprint,
        source=source,
    )


def _read_jsonl(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno}: invalid JSON ({exc.msg})") from exc
            if limit is not None and len(rows) >= limit:
                break
    return rows


def check_dataset_available(spec: DatasetSpec) -> None:
    """Fail fast on a dataset that cannot possibly load.

    Called as a pre-flight before the model is loaded, so a typo costs a second
    rather than a full model download. Deliberately *not* a schema validator:
    stored run records are historical data and must stay readable even when the
    rules tighten.
    """
    if spec.source == "hub":
        # `datasets` v3 dropped un-namespaced canonical ids. Plain 'wikitext'
        # now fails with an opaque URI parsing error deep inside the library.
        if "/" not in spec.path:
            raise ValueError(
                f"hub dataset {spec.path!r} must be namespaced as 'owner/name' "
                f"(for example 'Salesforce/wikitext')"
            )
        return

    if not Path(spec.path).exists():
        raise FileNotFoundError(f"dataset file not found: {spec.path}")


def _load_hub_rows(spec: DatasetSpec) -> list[dict[str, Any]]:
    check_dataset_available(spec)
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - datasets is a hard dependency
        raise RuntimeError("`datasets` is required for hub datasets") from exc

    dataset = load_dataset(spec.path, spec.name, split=spec.split)
    if spec.limit is not None:
        dataset = dataset.select(range(min(spec.limit, len(dataset))))
    return [dict(row) for row in dataset]


def _require_column(row: dict[str, Any], column: str, source: str) -> Any:
    if column not in row:
        raise KeyError(f"{source}: column {column!r} not found. Available columns: {sorted(row)}")
    return row[column]


def load_text_corpus(spec: DatasetSpec) -> TextCorpus:
    """Load documents for a perplexity task."""
    source = f"{spec.source}:{spec.path}"

    if spec.source == "text":
        path = Path(spec.path)
        if not path.exists():
            raise FileNotFoundError(f"text corpus not found: {path}")
        documents = [path.read_text(encoding="utf-8")]
    elif spec.source == "jsonl":
        path = Path(spec.path)
        if not path.exists():
            raise FileNotFoundError(f"jsonl corpus not found: {path}")
        rows = _read_jsonl(path, spec.limit)
        documents = [str(_require_column(r, spec.text_column, source)) for r in rows]
    else:
        rows = _load_hub_rows(spec)
        documents = [str(_require_column(r, spec.text_column, source)) for r in rows]

    # Hub corpora such as wikitext are full of blank separator lines; they add
    # nothing to a perplexity estimate but do skew the document count.
    documents = [d for d in documents if d.strip()]
    if not documents:
        raise ValueError(f"{source}: no non-empty documents found")

    if spec.limit is not None:
        documents = documents[: spec.limit]

    return TextCorpus(
        documents=documents,
        fingerprint=hash_text_stream(documents),
        source=source,
    )


def load_multiple_choice(
    spec: DatasetSpec,
    *,
    context_column: str = "context",
    choices_column: str = "choices",
    answer_column: str = "answer_index",
    transform: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
) -> MultipleChoiceSet:
    """Load multiple-choice examples from a hub dataset or a local JSONL file.

    The expected JSONL schema is one object per line::

        {"id": "q1", "context": "Q: ...\nA:", "choices": [" yes", " no"],
         "answer_index": 0}

    Choices normally carry their own leading space: scoring appends them to the
    context verbatim so tokenization matches what a real prompt would produce.

    ``transform`` adapts a hub dataset's native schema into that shape; see
    :mod:`autodistiller.evaluation.preprocessors`.
    """
    source = f"{spec.source}:{spec.path}"

    if spec.source == "text":
        raise ValueError("multiple_choice tasks need source 'jsonl' or 'hub', not 'text'")
    if spec.source == "jsonl":
        path = Path(spec.path)
        if not path.exists():
            raise FileNotFoundError(f"dataset not found: {path}")
        rows = _read_jsonl(path, spec.limit)
    else:
        rows = _load_hub_rows(spec)

    examples: list[MultipleChoiceExample] = []
    for index, row in enumerate(rows):
        if transform is not None:
            try:
                row = transform(row)
            except (KeyError, ValueError, TypeError) as exc:
                raise ValueError(f"{source}: preprocessor failed on row {index}: {exc}") from exc
        context = str(_require_column(row, context_column, source))
        raw_choices = _require_column(row, choices_column, source)
        if not isinstance(raw_choices, (list, tuple)):
            raise TypeError(f"{source}: {choices_column!r} must be a list, got {type(raw_choices)}")
        answer = _require_column(row, answer_column, source)
        examples.append(
            MultipleChoiceExample(
                id=str(row.get("id", index)),
                context=context,
                choices=[str(c) for c in raw_choices],
                answer_index=int(answer),
            )
        )

    if not examples:
        raise ValueError(f"{source}: no examples found")

    fingerprint = hash_obj(
        [
            {"context": e.context, "choices": e.choices, "answer_index": e.answer_index}
            for e in examples
        ]
    )
    return MultipleChoiceSet(examples=examples, fingerprint=fingerprint, source=source)


def load_sentence_pairs(
    spec: DatasetSpec,
    *,
    text_a_column: str = "sentence1",
    text_b_column: str = "sentence2",
    score_column: str = "label",
) -> SentencePairSet:
    """Load scored sentence pairs from a hub dataset or a local JSONL file.

    The expected JSONL schema is one object per line::

        {"sentence1": "a man is playing a guitar",
         "sentence2": "a man plays a guitar", "label": 4.6}

    The score's scale does not matter. Quality is a rank correlation, so any
    monotonic scale gives the same answer -- 0-5 as STS-B uses, or 0-1.
    """
    source = f"{spec.source}:{spec.path}"

    if spec.source == "text":
        raise ValueError("embedding tasks need source 'jsonl' or 'hub', not 'text'")
    if spec.source == "jsonl":
        path = Path(spec.path)
        if not path.exists():
            raise FileNotFoundError(f"dataset not found: {path}")
        rows = _read_jsonl(path, spec.limit)
    else:
        rows = _load_hub_rows(spec)

    examples: list[SentencePair] = []
    for index, row in enumerate(rows):
        examples.append(
            SentencePair(
                id=str(row.get("id", index)),
                text_a=str(_require_column(row, text_a_column, source)),
                text_b=str(_require_column(row, text_b_column, source)),
                score=float(_require_column(row, score_column, source)),
            )
        )

    if not examples:
        raise ValueError(f"{source}: no examples found")

    # A rank correlation over fewer than three points is not a measurement.
    if len(examples) < 3:
        raise ValueError(f"{source}: need at least 3 pairs to correlate, got {len(examples)}")

    fingerprint = hash_obj([{"a": e.text_a, "b": e.text_b, "score": e.score} for e in examples])
    return SentencePairSet(examples=examples, fingerprint=fingerprint, source=source)
