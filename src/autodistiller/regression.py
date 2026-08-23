"""Quality regression reporting.

A baseline is only worth establishing if something later gets compared against
it. This module answers one question -- *did quality hold?* -- and, just as
importantly, refuses to answer it when the two runs are not comparable.

Retention is direction-aware, so ``1.0`` always means "as good as the baseline"
whether the metric is accuracy (higher is better) or perplexity (lower is
better). A candidate at ``retention = 0.98`` kept 98% of the baseline's quality
on that metric, and ``--min-retention 0.95`` is the constraint Phase 5 will
optimise under.
"""

from __future__ import annotations

import math
from typing import Literal

from pydantic import BaseModel, Field

from .results import MetricValue, RunRecord

Verdict = Literal["pass", "fail", "not_comparable"]
IssueLevel = Literal["info", "warning", "error"]

DEFAULT_MIN_RETENTION = 0.99
SIGNIFICANCE_SIGMA = 2.0
"""How many combined standard errors a change must exceed to be called real."""


class ComparabilityIssue(BaseModel):
    level: IssueLevel
    message: str


class MetricComparison(BaseModel):
    """One metric, before and after."""

    task: str
    metric: str
    higher_is_better: bool
    baseline: float
    candidate: float
    delta: float
    relative_delta: float | None = None
    retention: float | None = None
    significant: bool | None = None
    verdict: Verdict = "pass"

    @property
    def improved(self) -> bool:
        return self.delta > 0 if self.higher_is_better else self.delta < 0

    def format_retention(self) -> str:
        return "n/a" if self.retention is None else f"{self.retention * 100:.2f}%"


class RegressionReport(BaseModel):
    baseline_run_id: str
    candidate_run_id: str
    min_retention: float
    comparisons: list[MetricComparison] = Field(default_factory=list)
    issues: list[ComparabilityIssue] = Field(default_factory=list)

    @property
    def failures(self) -> list[MetricComparison]:
        return [c for c in self.comparisons if c.verdict == "fail"]

    @property
    def blocking_issues(self) -> list[ComparabilityIssue]:
        return [i for i in self.issues if i.level == "error"]

    @property
    def passed(self) -> bool:
        return not self.failures and not self.blocking_issues

    @property
    def worst(self) -> MetricComparison | None:
        scored = [c for c in self.comparisons if c.retention is not None]
        return min(scored, key=lambda c: c.retention or 0.0) if scored else None


def _retention(baseline: float, candidate: float, higher_is_better: bool) -> float | None:
    """Fraction of baseline quality retained, oriented so >= 1.0 is always good."""
    if baseline == 0 or not math.isfinite(baseline) or not math.isfinite(candidate):
        return None
    if higher_is_better:
        return candidate / baseline
    if candidate == 0:
        return None
    return baseline / candidate


def _significant(baseline: MetricValue, candidate: MetricValue) -> bool | None:
    """True when the change exceeds combined sampling noise."""
    if baseline.stderr is None or candidate.stderr is None:
        return None
    combined = math.sqrt(baseline.stderr**2 + candidate.stderr**2)
    if combined == 0:
        return abs(candidate.value - baseline.value) > 0
    return abs(candidate.value - baseline.value) > SIGNIFICANCE_SIGMA * combined


def _check_comparability(baseline: RunRecord, candidate: RunRecord) -> list[ComparabilityIssue]:
    issues: list[ComparabilityIssue] = []

    if baseline.status != "ok":
        issues.append(ComparabilityIssue(level="error", message="baseline run did not succeed"))
    if candidate.status != "ok":
        issues.append(ComparabilityIssue(level="error", message="candidate run did not succeed"))

    if baseline.hardware.fingerprint != candidate.hardware.fingerprint:
        issues.append(
            ComparabilityIssue(
                level="warning",
                message=(
                    f"different hardware: {baseline.hardware.describe()} vs "
                    f"{candidate.hardware.describe()}. Quality metrics remain valid; "
                    f"timings do not."
                ),
            )
        )

    if baseline.environment.fingerprint != candidate.environment.fingerprint:
        moved = [
            f"{name} {baseline.environment.packages[name]} -> {version}"
            for name, version in candidate.environment.packages.items()
            if baseline.environment.packages.get(name) not in (None, version)
        ]
        issues.append(
            ComparabilityIssue(
                level="warning",
                message="software stack differs" + (f" ({', '.join(moved)})" if moved else ""),
            )
        )

    if (
        baseline.model.architecture_fingerprint
        and candidate.model.architecture_fingerprint
        and baseline.model.architecture_fingerprint != candidate.model.architecture_fingerprint
    ):
        # Expected when comparing a compressed candidate: worth stating, not blocking.
        issues.append(
            ComparabilityIssue(
                level="info",
                message=(
                    "model architecture/dtype differs from baseline (expected after compression)"
                ),
            )
        )

    baseline_tasks = {t.name for t in baseline.tasks}
    candidate_tasks = {t.name for t in candidate.tasks}
    for name in sorted(baseline_tasks - candidate_tasks):
        issues.append(
            ComparabilityIssue(level="warning", message=f"task {name!r} missing from candidate")
        )
    for name in sorted(candidate_tasks - baseline_tasks):
        issues.append(
            ComparabilityIssue(level="warning", message=f"task {name!r} not present in baseline")
        )

    return issues


def compare_runs(
    baseline: RunRecord,
    candidate: RunRecord,
    *,
    min_retention: float = DEFAULT_MIN_RETENTION,
    metrics: list[str] | None = None,
) -> RegressionReport:
    """Compare a candidate run against a baseline run.

    ``metrics`` restricts the check to specific metric names; by default every
    metric the two runs share is compared.
    """
    report = RegressionReport(
        baseline_run_id=baseline.run_id,
        candidate_run_id=candidate.run_id,
        min_retention=min_retention,
        issues=_check_comparability(baseline, candidate),
    )

    for baseline_task in baseline.tasks:
        candidate_task = candidate.task(baseline_task.name)
        if candidate_task is None:
            continue

        # Scoring on different data is the silent way to reach a wrong
        # conclusion, so it invalidates the task rather than warning about it.
        data_mismatch = bool(
            baseline_task.dataset_fingerprint
            and candidate_task.dataset_fingerprint
            and baseline_task.dataset_fingerprint != candidate_task.dataset_fingerprint
        )
        if data_mismatch:
            report.issues.append(
                ComparabilityIssue(
                    level="error",
                    message=(
                        f"task {baseline_task.name!r} was scored on different data "
                        f"({baseline_task.dataset_fingerprint} vs "
                        f"{candidate_task.dataset_fingerprint})"
                    ),
                )
            )

        for baseline_metric in baseline_task.metrics:
            if metrics and baseline_metric.name not in metrics:
                continue
            candidate_metric = candidate_task.metric(baseline_metric.name)
            if candidate_metric is None:
                continue

            retention = _retention(
                baseline_metric.value,
                candidate_metric.value,
                baseline_metric.higher_is_better,
            )
            delta = candidate_metric.value - baseline_metric.value

            if data_mismatch or retention is None:
                verdict: Verdict = "not_comparable"
            else:
                verdict = "pass" if retention >= min_retention else "fail"

            report.comparisons.append(
                MetricComparison(
                    task=baseline_task.name,
                    metric=baseline_metric.name,
                    higher_is_better=baseline_metric.higher_is_better,
                    baseline=baseline_metric.value,
                    candidate=candidate_metric.value,
                    delta=delta,
                    relative_delta=(
                        delta / baseline_metric.value if baseline_metric.value else None
                    ),
                    retention=retention,
                    significant=_significant(baseline_metric, candidate_metric),
                    verdict=verdict,
                )
            )

    return report


__all__ = [
    "DEFAULT_MIN_RETENTION",
    "ComparabilityIssue",
    "MetricComparison",
    "RegressionReport",
    "compare_runs",
]
