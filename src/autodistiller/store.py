"""On-disk run storage and the experiment cache.

One directory per run: the record, and the exact config that produced it. Runs
are content-addressed by the Phase 6 cache keys (:mod:`autodistiller.cache`), so
a lookup is "has this exact experiment already been measured on this machine
with this stack", not "is there a run with a similar name".

Alongside the run directories sits ``index.jsonl``: one line per record, holding
the keys and a summary but not the metrics. It exists because the callers that
matter look things up in a loop -- the optimizer checks the cache once per
candidate -- and answering that by parsing every record on disk gets slower with
every run ever done. The index is derived state: delete it and it rebuilds. It
is also, deliberately, the shape a shared benchmark database would want: flat
rows carrying a complete key, rather than a local file layout.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path

from .config import RunConfig
from .results import RunRecord

logger = logging.getLogger(__name__)

RECORD_FILENAME = "record.json"
CONFIG_FILENAME = "config.yaml"
INDEX_FILENAME = "index.jsonl"

_SLUG_RE = re.compile(r"[^a-zA-Z0-9._-]+")


def _slug(text: str, *, max_length: int = 40) -> str:
    return _SLUG_RE.sub("-", text).strip("-")[:max_length] or "model"


def make_run_id(config: RunConfig, *, now: datetime | None = None) -> str:
    """Timestamp + model + config fingerprint.

    Sortable, human-readable, and collision-resistant across configs that differ
    only in a task parameter.
    """
    stamp = (now or datetime.now(timezone.utc)).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}_{_slug(config.model.id.split('/')[-1])}_{config.fingerprint[:8]}"


def _index_row(record: RunRecord) -> dict:
    """The summary the index keeps: enough to answer a lookup, no metrics."""
    return {
        "run_id": record.run_id,
        "created_at": record.created_at.isoformat(),
        "schema_version": record.schema_version,
        "model": record.model.id,
        "status": record.status,
        "candidate_id": record.candidate_id,
        "config_fingerprint": record.config_fingerprint,
        "experiment_key": record.experiment_key,
        "benchmark_key": record.benchmark_key,
        "has_tasks": bool(record.tasks),
        "has_deployment": record.deployment is not None,
        "has_compression": record.compression is not None,
    }


class RunStore:
    """Reads and writes run directories under a root, and indexes them."""

    def __init__(self, root: Path | str = Path("runs")) -> None:
        self.root = Path(root)
        self._index: list[dict] | None = None

    # Paths

    def run_dir(self, run_id: str) -> Path:
        return self.root / run_id

    def new_run_id(self, config: RunConfig) -> str:
        """A run id that is not already taken under this root.

        Run ids are timestamped to the second, which is readable but not unique:
        two runs of the same config within one second collide, and the second
        would then be written straight over the first. That is reachable in
        practice -- ``--refresh`` on a fast evaluation does exactly it -- so the
        store, which owns this namespace, disambiguates rather than losing a
        result.
        """
        base = make_run_id(config)
        run_id, attempt = base, 2
        while self.run_dir(run_id).exists():
            run_id = f"{base}-{attempt}"
            attempt += 1
        return run_id

    @property
    def index_path(self) -> Path:
        return self.root / INDEX_FILENAME

    # Writing

    def save(self, record: RunRecord) -> Path:
        directory = self.run_dir(record.run_id)
        directory.mkdir(parents=True, exist_ok=True)
        record.save(directory / RECORD_FILENAME)
        record.config.save(directory / CONFIG_FILENAME)
        self._append_index(record)
        return directory

    def _append_index(self, record: RunRecord) -> None:
        row = _index_row(record)

        # Re-saving an enriched record (the optimizer attaches a benchmark to a
        # record the evaluator already wrote) must replace the row, not add a
        # second one that answers lookups with a stale key.
        index = self._rows()
        replaced = False
        for position, existing in enumerate(index):
            if existing.get("run_id") == record.run_id:
                index[position] = row
                replaced = True
                break
        if not replaced:
            index.append(row)

        self._write_index(index)

    def _write_index(self, rows: list[dict]) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        text = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
        self.index_path.write_text(text, encoding="utf-8")
        self._index = rows

    # Reading

    def load(self, run_id: str) -> RunRecord:
        path = self.run_dir(run_id) / RECORD_FILENAME
        if not path.exists():
            raise FileNotFoundError(f"no run record at {path}")
        return RunRecord.load(path)

    def _record_paths(self) -> list[Path]:
        if not self.root.exists():
            return []
        return sorted(self.root.glob(f"*/{RECORD_FILENAME}"), reverse=True)

    def _rows(self) -> list[dict]:
        """The index, rebuilt from the records on disk when it cannot be trusted.

        Trust is a count: an index with fewer rows than there are records was
        written by an older version, or lost rows to an interrupted write. Either
        way a full rebuild is cheap next to the runs it saves.
        """
        if self._index is not None:
            return self._index

        rows: list[dict] = []
        if self.index_path.exists():
            for line in self.index_path.read_text(encoding="utf-8").splitlines():
                if not (line := line.strip()):
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:  # a truncated final write
                    continue

        if len(rows) < len(self._record_paths()):
            rows = self.rebuild_index()

        self._index = rows
        return rows

    def rebuild_index(self) -> list[dict]:
        """Re-derive the index by reading every record. Safe to call any time."""
        rows: list[dict] = []
        for path in reversed(self._record_paths()):  # oldest first, so it reads as history
            try:
                rows.append(_index_row(RunRecord.load(path)))
            except Exception as exc:
                # A record we cannot parse still gets a row. Leaving it out
                # would make the index look short of the directory listing, and
                # the staleness check would then rebuild on every single lookup
                # for as long as the bad file sits there. The row matches no
                # key and is never "ok", so it can only ever be skipped.
                logger.debug("indexing unreadable record %s as inert: %s", path, exc)
                rows.append({"run_id": path.parent.name, "status": "unreadable"})
        self._write_index(rows)
        return rows

    def list_records(self, *, limit: int | None = None) -> list[RunRecord]:
        """Newest first. Unreadable records are skipped rather than fatal."""
        records: list[RunRecord] = []
        for path in self._record_paths():
            try:
                records.append(RunRecord.load(path))
            except Exception:  # a partially written or older-schema record
                continue
            if limit is not None and len(records) >= limit:
                break
        return records

    def summaries(self, *, limit: int | None = None) -> list[dict]:
        """Index rows, newest first. Reads no record files."""
        rows = list(reversed(self._rows()))
        return rows[:limit] if limit is not None else rows

    # Cache lookups

    def _find(self, field: str, key: str, requires: str) -> RunRecord | None:
        """Newest successful record whose ``field`` matches and that has ``requires``."""
        if not key:
            # Records predating the cache carry no keys. Matching None against
            # None would hand back an arbitrary old run as a cache hit.
            return None

        for row in reversed(self._rows()):
            if row.get(field) != key or row.get("status") != "ok" or not row.get(requires):
                continue
            try:
                return self.load(row["run_id"])
            except (FileNotFoundError, ValueError):
                # The index outlived the directory, or the record no longer
                # parses. Keep looking; a stale row is not a cache miss for
                # every older run that is still on disk.
                continue
        return None

    def find_experiment(self, key: str) -> RunRecord | None:
        """A completed evaluation for this experiment key."""
        return self._find("experiment_key", key, requires="has_tasks")

    def find_benchmark(self, key: str) -> RunRecord | None:
        """A completed deployment benchmark for this benchmark key."""
        return self._find("benchmark_key", key, requires="has_deployment")

    def resolve(self, reference: str) -> RunRecord:
        """Load a run by path to a record, path to a run dir, or run id."""
        candidate = Path(reference)
        if candidate.is_file():
            return RunRecord.load(candidate)
        if (candidate / RECORD_FILENAME).is_file():
            return RunRecord.load(candidate / RECORD_FILENAME)
        return self.load(reference)


__all__ = [
    "CONFIG_FILENAME",
    "INDEX_FILENAME",
    "RECORD_FILENAME",
    "RunStore",
    "make_run_id",
]
