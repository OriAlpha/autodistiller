"""The optimizer.

Runs candidates through progressively more expensive stages and drops them as
soon as they fail, so the costly work only ever happens on configurations that
are still plausible:

===============  ==========  ====================================
memory estimate  free        arithmetic over the model config
compress         minutes     one llmcompressor run
quality screen   seconds     perplexity against the baseline
benchmark        minutes     a real server, started and measured
===============  ==========  ====================================

Ordering is set by the objective, which is what makes stopping early honest: the
first candidate that qualifies is the best one under that objective, not merely
the first one tried.
"""

from __future__ import annotations

import logging
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
from ..results import CompressionArtifact, DeploymentBenchmark, ModelInfo, RunRecord
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


def quality_retention(baseline: RunRecord, candidate: RunRecord) -> tuple[float | None, dict]:
    """Worst per-metric retention across the shared tasks.

    The worst metric, not the average: a candidate that holds perplexity but
    collapses on the task the user actually cares about has not held quality.
    """
    per_metric: dict[str, float] = {}

    for baseline_task in baseline.tasks:
        candidate_task = candidate.task(baseline_task.name)
        if candidate_task is None:
            continue
        for metric in baseline_task.metrics:
            other = candidate_task.metric(metric.name)
            if other is None:
                continue
            value = _retention(metric.value, other.value, metric.higher_is_better)
            if value is not None:
                per_metric[f"{baseline_task.name}/{metric.name}"] = value

    return (min(per_metric.values()) if per_metric else None), per_metric


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

                if candidate.is_baseline:
                    outcome.quality_retention = 1.0
                elif baseline_record is not None:
                    retention, per_metric = quality_retention(baseline_record, record)
                    outcome.quality_retention = retention
                    outcome.quality_metrics = per_metric

                outcome.violations = self.constraints.check_quality(outcome.quality_retention)
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
    "quality_retention",
]
