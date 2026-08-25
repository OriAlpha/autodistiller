"""The optimizer.

Runs candidates through progressively more expensive stages and drops them as
soon as they fail, so the costly work only ever happens on configurations that
are still plausible:

===============  ===========================  ==========================
memory estimate  free                         arithmetic over the config
compress         1-7 min                      one backend run
quality screen   scales with --limit          perplexity vs the baseline
benchmark        2-3 min                      a real server, measured
===============  ===========================  ==========================

Measured on Qwen3-4B, an RTX 5070 and ``--limit 256``. The quality screen is
listed by what drives it rather than by a number because it is the one stage the
user sets the cost of: at ``--limit 8`` it is seconds, at 256 it took 12 minutes
and was the most expensive stage in the search -- more than compression, more
than the benchmark. Screening still runs before benchmarking, because it is the
stage that can reject a candidate on quality alone, but "cheap" is a property of
the limit and not of the stage.

Ordering is set by the objective, which is what makes stopping early honest: the
first candidate that qualifies is the best one under that objective, not merely
the first one tried.
"""

from __future__ import annotations

import logging
import math
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from ..cache import benchmark_key
from ..candidates.generator import Candidate, CandidateSet
from ..compression.backend import CompressionError
from ..compression.pipeline import run_compression
from ..config import CompressionSpec, DeploymentSpec, ModelSpec, RunConfig
from ..metadata.environment import EnvironmentInfo, collect_environment
from ..metadata.hardware import HardwareInfo, detect_hardware
from ..regression import SIGNIFICANCE_SIGMA
from ..results import (
    CompressionArtifact,
    DeploymentBenchmark,
    MetricValue,
    ModelInfo,
    RunRecord,
)
from ..store import RunStore
from .constraints import Constraints, Objective, Score, score_candidate, search_order
from .pareto import ParetoReport

logger = logging.getLogger(__name__)

ProgressFn = Callable[[str], None]

STAGES = ("screened", "compressed", "evaluated", "benchmarked")


@dataclass
class CandidateOutcome:
    """What happened to one candidate, and how far it got."""

    candidate: Candidate
    stage: str = "screened"
    artifact: CompressionArtifact | None = None
    benchmark: DeploymentBenchmark | None = None
    quality_metrics: dict[str, float] = field(default_factory=dict)
    quality_retention: float | None = None
    quality_stderr: float | None = None
    quality_metric_task: str | None = None
    measured_metrics: dict[str, MetricValue] = field(default_factory=dict)
    """Each task's headline metric, keyed by task name.

    Every task the user asked for, not just the first. Running two tasks and
    reporting one discards half of what the evaluation cost.
    """

    quality_metric: MetricValue | None = None
    """The metric the frontier ranks on: the first task's.

    One task has to be picked, because absolute scores from different tasks are
    not one axis -- 0.56 on hellaswag is not worse than 0.74 on arc_easy, they
    are different questions. The axis is labelled with the task it used.

    Retention needs a baseline, and a baseline is exactly what is missing when
    the unquantized model does not fit -- the case the tool most exists for. The
    measurement still happened and still costs the same minutes, so it is kept
    here and reported rather than discarded for want of something to divide by.
    """
    warnings: list[str] = field(default_factory=list)
    """Things that do not disqualify a candidate but change how much its numbers
    are worth. A measurement too noisy to settle the constraint it was checked
    against is the main one."""
    weights_bytes: int | None = None
    violations: list[str] = field(default_factory=list)
    error: str | None = None
    score: Score | None = None
    duration_s: float = 0.0
    record: RunRecord | None = None
    reused: list[str] = field(default_factory=list)
    """Stages answered from cache rather than measured. Reported, not inferred:
    a result the user did not just pay for should say so."""

    @property
    def qualified(self) -> bool:
        return not self.violations and self.error is None

    @property
    def served_model(self) -> str | None:
        if self.candidate.is_baseline:
            return None
        return self.artifact.output_dir if self.artifact else None

    def summary(self) -> str:
        if self.error:
            return f"{self.candidate.id}: failed ({self.error})"
        note = f" (reused {', '.join(self.reused)})" if self.reused else ""
        if self.violations:
            return f"{self.candidate.id}: rejected ({self.violations[0]}){note}"
        if self.warnings:
            note += f" -- {self.warnings[0]}"
        return f"{self.candidate.id}: qualified at {self.stage}{note}"


@dataclass
class OptimizationResult:
    """The full search: what was tried, what qualified, and what won."""

    model_id: str
    objective: Objective
    constraints: Constraints
    backend: str
    outcomes: list[CandidateOutcome] = field(default_factory=list)
    baseline_record: RunRecord | None = None
    stopped_early: bool = False
    duration_s: float = 0.0

    @property
    def qualified(self) -> list[CandidateOutcome]:
        return [o for o in self.outcomes if o.qualified]

    def timing(self) -> dict[str, float]:
        """Wall-clock seconds by stage, across every candidate.

        Reported because the roadmap sets a target -- roughly an hour for a
        small model -- and a search that cannot say how long it took cannot be
        held to it. The split matters more than the total: which stage dominates
        depends on the model and on --limit, not on the design.
        """
        compressing = sum(o.artifact.duration_s for o in self.outcomes if o.artifact is not None)
        evaluating = sum(o.record.total_duration_s for o in self.outcomes if o.record is not None)
        measured = sum(o.duration_s for o in self.outcomes)
        return {
            "total": self.duration_s or measured,
            "compressing": compressing,
            "evaluating": evaluating,
            # What is left is the servers: starting, warming and sweeping them.
            "benchmarking": max(measured - compressing - evaluating, 0.0),
        }

    @property
    def reused_stages(self) -> dict[str, int]:
        """How many candidates skipped each stage because the cache had it."""
        counts: dict[str, int] = {}
        for outcome in self.outcomes:
            for stage in outcome.reused:
                counts[stage] = counts.get(stage, 0) + 1
        return counts

    @property
    def recommended(self) -> CandidateOutcome | None:
        """Highest-scoring qualifying candidate under the chosen objective."""
        scored = [o for o in self.qualified if o.score is not None]
        return max(scored, key=lambda o: o.score.value if o.score else 0.0) if scored else None

    def pareto(self) -> ParetoReport:
        """The trade-off view over everything that qualified.

        Qualifying candidates only: one that violated a constraint is not a
        trade-off, it is disqualified. The baseline is kept, because "do not
        compress" is a real option and usually the best-quality one.
        """
        return ParetoReport(self.qualified)

    def explain(self) -> str:
        """Why the recommendation qualifies, in the roadmap's words."""
        best = self.recommended
        if best is None or best.score is None:
            return (
                f"No candidate satisfied {self.constraints.describe()}. "
                f"{len(self.outcomes)} configurations were evaluated."
            )

        parts = [f"{best.candidate.id} wins on {best.score.basis} ({best.score.detail})"]
        if best.quality_retention is not None:
            parts.append(f"quality retention {best.quality_retention * 100:.2f}%")
        if best.benchmark is not None and best.benchmark.peak_vram_bytes:
            parts.append(f"peak VRAM {best.benchmark.peak_vram_bytes / 1024**3:.2f} GiB")
        parts.append(f"chosen from {len(self.qualified)} qualifying of {len(self.outcomes)} tried")
        if reused := self.reused_stages:
            parts.append(
                "reused from cache: "
                + ", ".join(f"{count}x {stage}" for stage, count in sorted(reused.items()))
            )
        return "; ".join(parts) + "."


def _retention(baseline: float, candidate: float, higher_is_better: bool) -> float | None:
    """Direction-aware, matching the regression report's definition."""
    if baseline == 0 or candidate == 0:
        return None
    return candidate / baseline if higher_is_better else baseline / candidate


def _retention_stderr(
    retention: float, baseline: MetricValue, candidate: MetricValue
) -> float | None:
    """Uncertainty on a retention ratio, propagated from both measurements.

    Retention is a ratio of two measured numbers, and a ratio's relative
    uncertainty is the two relative uncertainties added in quadrature. Without
    this the ratio is reported to four decimal places whatever the sample size,
    which reads as precision the measurement does not have.
    """
    if baseline.stderr is None or candidate.stderr is None:
        return None
    if baseline.value == 0 or candidate.value == 0:
        return None
    relative = math.hypot(baseline.stderr / baseline.value, candidate.stderr / candidate.value)
    return abs(retention) * relative


@dataclass(frozen=True)
class QualityComparison:
    """How much quality a candidate held, and how well that is known.

    The second half matters as much as the first. Perplexity on a handful of
    documents carries a standard error that can rival the value, and a retention
    figure derived from two such numbers cannot settle a 95% floor no matter how
    many decimal places it is printed to.
    """

    retention: float | None = None
    per_metric: dict[str, float] = field(default_factory=dict)
    stderr: float | None = None
    worst_metric: str | None = None

    @property
    def lower_bound(self) -> float | None:
        """Retention at the pessimistic end of its uncertainty."""
        if self.retention is None or self.stderr is None:
            return None
        return self.retention - SIGNIFICANCE_SIGMA * self.stderr

    @property
    def upper_bound(self) -> float | None:
        if self.retention is None or self.stderr is None:
            return None
        return self.retention + SIGNIFICANCE_SIGMA * self.stderr

    def indistinguishable_from(self, level: float) -> bool:
        """Whether the measurement can tell retention apart from ``level``."""
        if self.retention is None or self.stderr is None:
            return False
        return abs(self.retention - level) <= SIGNIFICANCE_SIGMA * self.stderr

    def describe(self) -> str:
        if self.retention is None:
            return "not measured"
        text = f"{self.retention * 100:.2f}%"
        if self.stderr is not None:
            text += f" ± {self.stderr * 100:.2f}%"
        return text


def quality_retention(baseline: RunRecord, candidate: RunRecord) -> QualityComparison:
    """Worst per-metric retention across the shared tasks, with its uncertainty.

    The worst metric, not the average: a candidate that holds perplexity but
    collapses on the task the user actually cares about has not held quality.
    The uncertainty comes from the same metric, since that is the one the
    decision is made on.
    """
    per_metric: dict[str, float] = {}
    stderrs: dict[str, float | None] = {}

    for baseline_task in baseline.tasks:
        candidate_task = candidate.task(baseline_task.name)
        if candidate_task is None:
            continue
        for metric in baseline_task.metrics:
            other = candidate_task.metric(metric.name)
            if other is None:
                continue
            value = _retention(metric.value, other.value, metric.higher_is_better)
            if value is None:
                continue
            key = f"{baseline_task.name}/{metric.name}"
            per_metric[key] = value
            stderrs[key] = _retention_stderr(value, metric, other)

    if not per_metric:
        return QualityComparison()

    worst = min(per_metric, key=lambda k: per_metric[k])
    return QualityComparison(
        retention=per_metric[worst],
        per_metric=per_metric,
        stderr=stderrs.get(worst),
        worst_metric=worst,
    )


class Optimizer:
    """Drives candidates through the stages.

    The evaluate and benchmark steps are injected so the whole pipeline can be
    exercised without a GPU. They are also the only two places that cost real
    time, which makes them the right seam.
    """

    def __init__(
        self,
        *,
        model: ModelSpec,
        constraints: Constraints,
        objective: Objective = Objective.BALANCED,
        backend: str = "vllm",
        artifacts_root: Path = Path("artifacts"),
        evaluate_fn: Callable[[str, Candidate], RunRecord] | None = None,
        benchmark_fn: Callable[[CandidateOutcome], DeploymentBenchmark] | None = None,
        compress_fn: Callable[[Candidate], CompressionArtifact] | None = None,
        calibration=None,
        llama_cpp_dir: str | None = None,
        stop_early: bool = True,
        store: RunStore | None = None,
        reuse: bool = True,
        benchmark_settings: dict | None = None,
        progress: ProgressFn | None = None,
    ) -> None:
        self.model = model
        self.constraints = constraints
        self.objective = objective
        self.backend = backend
        self.artifacts_root = artifacts_root
        self.evaluate_fn = evaluate_fn
        self.benchmark_fn = benchmark_fn
        self.compress_fn = compress_fn or self._default_compress
        self.calibration = calibration
        self.llama_cpp_dir = llama_cpp_dir
        self.stop_early = stop_early
        self.store = store
        self.reuse = reuse
        self.benchmark_settings = benchmark_settings or {}
        self.progress = progress

        # Once per search, not once per candidate: neither can change while the
        # search is running, and both are part of every cache key it computes.
        self.hardware: HardwareInfo = detect_hardware()
        self.environment: EnvironmentInfo = collect_environment()
        self._started_at = datetime.now(timezone.utc)

    def _say(self, message: str) -> None:
        if self.progress is not None:
            self.progress(message)

    def _default_compress(self, candidate: Candidate) -> CompressionArtifact:
        if candidate.method is None:
            raise ValueError("the baseline has nothing to compress")
        spec = CompressionSpec(
            method=candidate.method,
            calibration=self.calibration,
            llama_cpp_dir=self.llama_cpp_dir,
            output_dir=None,
        )
        return run_compression(self.model, spec, output_root=self.artifacts_root, reuse=self.reuse)

    def run(self, candidate_set: CandidateSet) -> OptimizationResult:
        started = time.perf_counter()
        result = OptimizationResult(
            model_id=self.model.id,
            objective=self.objective,
            constraints=self.constraints,
            backend=self.backend,
        )

        ordered = search_order(candidate_set.accepted, self.objective)

        # The baseline is not a candidate in the search: it is the reference
        # every retention figure is computed against, so it has to be measured
        # before anything can be judged. Leaving it in search order would put it
        # last under throughput, and then no candidate would have a baseline to
        # compare to -- quality constraints would pass by default and the
        # optimizer would happily recommend a broken model.
        baselines = [c for c in ordered if c.is_baseline]
        ordered = baselines + [c for c in ordered if not c.is_baseline]

        self._say(
            f"{len(ordered)} candidates, {self.objective.value} objective, "
            f"constraints: {self.constraints.describe()}"
        )

        # A quality floor needs something to measure against. Saying so now
        # beats saying it after several compressions, and beats not saying it.
        wants_quality = self.constraints.min_quality_retention is not None
        if wants_quality and not baselines:
            self._say(
                "  warning: no baseline survived screening, so quality retention "
                "cannot be measured and the quality floor cannot be verified. "
                "Raise --max-vram, or drop --min-quality to search without it."
            )

        baseline_record: RunRecord | None = None

        for candidate in ordered:
            outcome = self._run_candidate(candidate, baseline_record)
            result.outcomes.append(outcome)

            if candidate.is_baseline and outcome.error is None and baseline_record is None:
                baseline_record = outcome.record
                result.baseline_record = baseline_record

            self._say(f"  {outcome.summary()}")

            if self.stop_early and outcome.qualified and not candidate.is_baseline:
                result.stopped_early = True
                self._say(
                    f"stopping early: {candidate.id} satisfies every constraint and "
                    f"leads on {self.objective.value}"
                )
                break

        result.duration_s = time.perf_counter() - started
        return result

    def _run_candidate(
        self, candidate: Candidate, baseline_record: RunRecord | None
    ) -> CandidateOutcome:
        started = time.perf_counter()
        outcome = CandidateOutcome(candidate=candidate)

        # The estimate until compression replaces it with a measured size.
        outcome.weights_bytes = candidate.estimate.weights_bytes

        # Stage 1: the estimate, which costs nothing.
        outcome.violations = self.constraints.check_memory(candidate.estimate.total_bytes)
        if outcome.violations:
            outcome.duration_s = time.perf_counter() - started
            return outcome

        try:
            # Stage 2: compression. Skipped for the baseline, which is the
            # model as it already exists. An identical artifact already on disk
            # is reused inside run_compression, which is why this can be the
            # most expensive stage and still cost nothing the second time.
            if not candidate.is_baseline:
                self._say(f"  compressing {candidate.id}")
                outcome.artifact = self.compress_fn(candidate)
                outcome.stage = "compressed"
                if outcome.artifact.artifact_bytes:
                    outcome.weights_bytes = outcome.artifact.artifact_bytes
                if outcome.artifact.created_at < self._started_at:
                    outcome.reused.append("compression")

            # Stage 3: cheap quality screening.
            if self.evaluate_fn is not None:
                self._say(f"  evaluating {candidate.id}")
                target = outcome.served_model or self.model.id
                record = self.evaluate_fn(target, candidate)
                outcome.record = record
                outcome.stage = "evaluated"
                if record.created_at < self._started_at:
                    outcome.reused.append("evaluation")

                # "Save the exact recipe with each result": an evaluation of a
                # compressed candidate is a result *about* an artifact, and
                # without this the record names the artifact directory as its
                # model and says nothing about what produced it.
                if outcome.artifact is not None and record.compression is None:
                    record.compression = outcome.artifact
                    if self.store is not None:
                        self.store.save(record)

                # Whatever the run measured, kept whether or not it can be
                # turned into a ratio.
                for task in record.tasks:
                    if (primary := task.primary_metric) is not None:
                        outcome.measured_metrics[task.name] = primary
                if record.tasks and (primary := record.tasks[0].primary_metric) is not None:
                    outcome.quality_metric = primary
                    outcome.quality_metric_task = record.tasks[0].name

                if candidate.is_baseline:
                    outcome.quality_retention = 1.0
                elif baseline_record is not None:
                    comparison = quality_retention(baseline_record, record)
                    outcome.quality_retention = comparison.retention
                    outcome.quality_metrics = comparison.per_metric
                    outcome.quality_stderr = comparison.stderr

                    # A floor the measurement cannot settle is worth saying out
                    # loud: the verdict below is then a coin toss wearing a
                    # decimal point.
                    if (note := self.constraints.quality_warning(comparison)) is not None:
                        outcome.warnings.append(note)

                # Distinguish "not compared yet" from "nothing to compare
                # against": only the second means the floor can never be met.
                outcome.violations = self.constraints.check_quality(
                    outcome.quality_retention,
                    measurable=candidate.is_baseline or baseline_record is not None,
                )
                if outcome.violations:
                    outcome.duration_s = time.perf_counter() - started
                    return outcome

            # Stage 4: the deployment benchmark, only for survivors.
            if self.benchmark_fn is not None and self._needs_benchmark():
                key = self._benchmark_key(outcome)
                cached = self.store.find_benchmark(key) if self.reuse and self.store else None

                if cached is not None and cached.deployment is not None:
                    self._say(f"  reusing benchmark for {candidate.id} from {cached.run_id}")
                    outcome.benchmark = cached.deployment
                    outcome.reused.append("benchmark")
                else:
                    self._say(f"  benchmarking {candidate.id}")
                    outcome.benchmark = self.benchmark_fn(outcome)
                    self._persist(outcome, key)

                outcome.stage = "benchmarked"
                outcome.violations = self.constraints.check_benchmark(outcome.benchmark)

        except (CompressionError, RuntimeError, ValueError, OSError) as exc:
            outcome.error = f"{type(exc).__name__}: {exc}"
            logger.warning("candidate %s failed: %s", candidate.id, exc)

        outcome.score = score_candidate(outcome, self.objective)
        outcome.duration_s = time.perf_counter() - started
        return outcome

    def _benchmark_key(self, outcome: CandidateOutcome) -> str:
        """What makes this candidate's benchmark reusable.

        The served weights, plus everything about how they were served and
        driven. Context length and KV dtype are candidate properties rather than
        benchmark settings, but they are launch flags, so a change to either
        means a different server and a different measurement.
        """
        candidate = outcome.candidate
        return benchmark_key(
            served_model=outcome.served_model or self.model.id,
            backend=self.backend,
            hardware=self.hardware,
            environment=self.environment,
            settings={
                **self.benchmark_settings,
                "max_model_len": candidate.max_model_len,
                "kv_dtype": candidate.kv_dtype,
            },
        )

    def _persist(self, outcome: CandidateOutcome, key: str) -> None:
        """Write a measured benchmark to its own record, so a later search reuses it.

        Its own, rather than attached to the candidate's evaluation record.
        Context length is not a compression parameter, so candidates that differ
        only in it share one artifact and one evaluation while having genuinely
        different benchmarks. Attaching each to the shared evaluation record
        would mean the last one written silently replaced the rest.
        """
        if self.store is None or outcome.benchmark is None:
            return

        benchmark = outcome.benchmark
        candidate = outcome.candidate
        served = outcome.served_model or self.model.id

        config = RunConfig(
            model=ModelSpec(id=served),
            deployment=DeploymentSpec(
                backend=self.backend,
                endpoint=benchmark.endpoint,
                served_model=served,
                prompt_tokens=benchmark.prompt_tokens_requested or 1,
                max_tokens=benchmark.max_tokens or 1,
                concurrency_levels=[p.concurrency for p in benchmark.phases] or [1],
            ),
            label=candidate.id,
            output_dir=self.store.root,
        )

        record = RunRecord(
            run_id=self.store.new_run_id(config),
            config=config,
            config_fingerprint=config.fingerprint,
            benchmark_key=key,
            candidate_id=candidate.id,
            model=ModelInfo(id=served),
            hardware=self.hardware,
            environment=self.environment,
            deployment=benchmark,
            compression=outcome.artifact,
        )

        try:
            self.store.save(record)
        except OSError as exc:
            # A search that produced good numbers should not fail because they
            # could not be filed. The cost is only that the next run repeats it.
            logger.warning("could not cache benchmark for %s: %s", candidate.id, exc)

    def _needs_benchmark(self) -> bool:
        """Benchmark when a constraint or the objective depends on it.

        Ranking by size or quality needs no deployment run, so under those
        objectives the expensive stage is skipped entirely unless a latency or
        throughput constraint forces it.
        """
        if self.constraints.needs_benchmark:
            return True
        return self.objective in (Objective.THROUGHPUT, Objective.LATENCY, Objective.BALANCED)


__all__ = [
    "STAGES",
    "CandidateOutcome",
    "OptimizationResult",
    "Optimizer",
    "QualityComparison",
    "quality_retention",
]
