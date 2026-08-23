"""On-disk run storage.

Phase 1 writes one directory per run: the record, and the exact config that
produced it. Phase 6 turns this into a content-addressed experiment cache, so
runs are already indexed by ``config_fingerprint`` and reads go through this
module rather than through raw globs scattered across the codebase.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path

from .config import RunConfig
from .results import RunRecord

RECORD_FILENAME = "record.json"
CONFIG_FILENAME = "config.yaml"

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


class RunStore:
    """Reads and writes run directories under a root."""

    def __init__(self, root: Path | str = Path("runs")) -> None:
        self.root = Path(root)

    def run_dir(self, run_id: str) -> Path:
        return self.root / run_id

    def save(self, record: RunRecord) -> Path:
        directory = self.run_dir(record.run_id)
        directory.mkdir(parents=True, exist_ok=True)
        record.save(directory / RECORD_FILENAME)
        record.config.save(directory / CONFIG_FILENAME)
        return directory

    def load(self, run_id: str) -> RunRecord:
        path = self.run_dir(run_id) / RECORD_FILENAME
        if not path.exists():
            raise FileNotFoundError(f"no run record at {path}")
        return RunRecord.load(path)

    def list_records(self, *, limit: int | None = None) -> list[RunRecord]:
        """Newest first. Unreadable records are skipped rather than fatal."""
        if not self.root.exists():
            return []

        records: list[RunRecord] = []
        for path in sorted(self.root.glob(f"*/{RECORD_FILENAME}"), reverse=True):
            try:
                records.append(RunRecord.load(path))
            except Exception:  # a partially written or older-schema record
                continue
            if limit is not None and len(records) >= limit:
                break
        return records

    def find_by_fingerprint(self, config_fingerprint: str) -> RunRecord | None:
        """Most recent successful run for a config. The seed of the Phase 6 cache."""
        for record in self.list_records():
            if record.config_fingerprint == config_fingerprint and record.status == "ok":
                return record
        return None

    def resolve(self, reference: str) -> RunRecord:
        """Load a run by path to a record, path to a run dir, or run id."""
        candidate = Path(reference)
        if candidate.is_file():
            return RunRecord.load(candidate)
        if (candidate / RECORD_FILENAME).is_file():
            return RunRecord.load(candidate / RECORD_FILENAME)
        return self.load(reference)


__all__ = ["CONFIG_FILENAME", "RECORD_FILENAME", "RunStore", "make_run_id"]
