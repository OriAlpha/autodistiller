"""Constrained optimization.

The optimizer's expensive stages are injected, so the whole staged pipeline --
ordering, screening, early stopping, ranking -- runs here without a GPU. What is
tested is the decision logic: which candidates get expensive work spent on them,
which are dropped and why, and whether the winner is defensible.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from autodistiller.candidates.generator import generate_candidates
from autodistiller.config import ModelSpec
from autodistiller.metadata.profiles import resolve_profile
from autodistiller.optimize.constraints import (
    Constraints,
    Objective,
    score_candidate,
    search_order,
)
from autodistiller.optimize.pipeline import (
    CandidateOutcome,
    Optimizer,
    quality_retention,
)
from autodistiller.results import (
    CompressionArtifact,
    CompressionRecipe,
    ConcurrencyResult,
    DeploymentBenchmark,
    LatencyStats,
    MetricValue,
    ModelInfo,
    RunRecord,
    TaskResult,
)
from tests.test_candidates import qwen3_06b

BLACKWELL = resolve_profile("rtx-5090")
GIB = 1024**3


def _stats(value: float) -> LatencyStats:
    return LatencyStats(mean=value, p50=value, p90=value, p99=value, min=value, max=value)


def _benchmark(
    *, throughput: float = 1000.0, ttft: float = 0.05, tpot: float = 0.005, vram: int | None = None
) -> DeploymentBenchmark:
    return DeploymentBenchmark(
        backend="vllm",
        endpoint="http://fake",
        served_model="m",
        phases=[
            ConcurrencyResult(
                concurrency=1,
                n_requests=8,
                duration_s=1.0,
                ttft=_stats(ttft),
                tpot=_stats(tpot),
                output_tokens_per_s=throughput,
                peak_vram_bytes=vram,
            )
        ],
    )


def _record(run_id: str, *, perplexity: float, accuracy: float = 0.8) -> RunRecord:
    from autodistiller.config import DatasetSpec, PerplexityTask, RunConfig

    config = RunConfig(
        model=ModelSpec(id="m"),
        tasks=[PerplexityTask(name="wikitext2", dataset=DatasetSpec(source="text", path="c.txt"))],
    )
    return RunRecord(
        run_id=run_id,
        config=config,
        config_fingerprint=config.fingerprint,
        model=ModelInfo(id="m"),
        hardware=__import__(
            "autodistiller.metadata.hardware", fromlist=["detect_hardware"]
        ).detect_hardware(),
        environment=__import__(
            "autodistiller.metadata.environment", fromlist=["collect_environment"]
        ).collect_environment(),
        tasks=[
            TaskResult(
                name="wikitext2",
                kind="perplexity",
                metrics=[
                    MetricValue(name="perplexity", value=perplexity, higher_is_better=False),
                    MetricValue(name="acc", value=accuracy, higher_is_better=True),
                ],
            )
        ],
    )


def _artifact(method: str, size: int) -> CompressionArtifact:
    return CompressionArtifact(
        recipe=CompressionRecipe(
            method=method, scheme="W4A16", algorithm="rtn", weight_bits=4, activation_bits=16
        ),
        backend="llmcompressor",
        source_model="m",
        output_dir=f"artifacts/{method}",
        artifact_bytes=size,
    )


# --- constraints --------------------------------------------------------


def test_no_constraints_rejects_nothing():
    constraints = Constraints()
    assert constraints.check_memory(999 * GIB) == []
    assert constraints.check_quality(0.01) == []
    assert constraints.check_benchmark(_benchmark()) == []


def test_memory_constraint_screens_on_the_estimate():
    """The cheapest possible rejection: no model loaded, no GPU touched."""
    constraints = Constraints(max_vram_bytes=4 * GIB)
    assert constraints.check_memory(2 * GIB) == []
    assert "exceeds" in constraints.check_memory(6 * GIB)[0]


def test_quality_floor():
    constraints = Constraints(min_quality_retention=0.95)
    assert constraints.check_quality(0.97) == []
    assert "below" in constraints.check_quality(0.90)[0]


def test_unmeasured_quality_is_not_a_violation():
    """Absence of evidence is not evidence of failure; the stage just has not run."""
    assert Constraints(min_quality_retention=0.99).check_quality(None) == []


def test_latency_and_throughput_constraints():
    constraints = Constraints(max_ttft_s=0.1, max_tpot_s=0.01, min_throughput_tokens_per_s=500)
    assert constraints.check_benchmark(_benchmark(ttft=0.05, tpot=0.005, throughput=800)) == []

    violations = constraints.check_benchmark(_benchmark(ttft=0.3, tpot=0.05, throughput=100))
    assert len(violations) == 3


def test_measured_vram_supersedes_the_estimate():
    """A candidate that fits on paper and not in practice must be rejected."""
    constraints = Constraints(max_vram_bytes=4 * GIB)
    assert constraints.check_memory(3 * GIB) == []
    violations = constraints.check_benchmark(_benchmark(vram=6 * GIB))
    assert violations and "measured peak VRAM" in violations[0]


def test_needs_benchmark_only_for_runtime_constraints():
    assert not Constraints(min_quality_retention=0.9, max_vram_bytes=GIB).needs_benchmark
    assert Constraints(max_ttft_s=0.1).needs_benchmark
    assert Constraints(min_throughput_tokens_per_s=100).needs_benchmark


def test_constraints_describe_themselves():
    text = Constraints(min_quality_retention=0.95, max_vram_bytes=8 * GIB).describe()
    assert "95.0%" in text and "8.0 GiB" in text
    assert Constraints().describe() == "none"


# --- search order -------------------------------------------------------


def test_throughput_tries_the_most_compressed_first():
    """What makes stopping early honest: the first qualifying candidate under
    this objective is also the fastest qualifying candidate."""
    candidates = generate_candidates(qwen3_06b(), profile=BLACKWELL).accepted
    ordered = search_order(candidates, Objective.THROUGHPUT)
    assert ordered[0].method is not None
    assert ordered[0].method.startswith("int4")


def test_quality_tries_the_least_lossy_first():
    candidates = generate_candidates(qwen3_06b(), profile=BLACKWELL).accepted
    ordered = search_order(candidates, Objective.QUALITY)
    assert ordered[0].is_baseline


def test_search_order_is_a_permutation():
    candidates = generate_candidates(qwen3_06b(), profile=BLACKWELL).accepted
    for objective in Objective:
        ordered = search_order(candidates, objective)
        assert sorted(c.id for c in ordered) == sorted(c.id for c in candidates)


# --- scoring ------------------------------------------------------------


def _outcome(**kwargs) -> CandidateOutcome:
    candidate = generate_candidates(qwen3_06b(), profile=BLACKWELL).accepted[0]
    return CandidateOutcome(candidate=candidate, **kwargs)


def test_throughput_scores_on_measured_throughput():
    score = score_candidate(_outcome(benchmark=_benchmark(throughput=2500)), Objective.THROUGHPUT)
    assert score.value == 2500
    assert "tok/s" in score.detail


def test_latency_scoring_prefers_lower_ttft():
    fast = score_candidate(_outcome(benchmark=_benchmark(ttft=0.02)), Objective.LATENCY)
    slow = score_candidate(_outcome(benchmark=_benchmark(ttft=0.2)), Objective.LATENCY)
    assert fast.value > slow.value


def test_size_scoring_prefers_smaller_and_needs_no_benchmark():
    small = score_candidate(_outcome(weights_bytes=GIB // 2), Objective.SIZE)
    large = score_candidate(_outcome(weights_bytes=GIB * 2), Objective.SIZE)
    assert small.value > large.value


def test_quality_scoring_needs_no_benchmark():
    score = score_candidate(_outcome(quality_retention=0.98), Objective.QUALITY)
    assert score.value == pytest.approx(0.98)


def test_unbenchmarked_candidate_scores_zero_on_runtime_objectives():
    assert score_candidate(_outcome(), Objective.THROUGHPUT).value == 0.0


# --- quality retention --------------------------------------------------


def test_retention_uses_the_worst_metric():
    """A candidate that holds perplexity but collapses on the task the user
    cares about has not held quality."""
    baseline = _record("base", perplexity=10.0, accuracy=0.80)
    candidate = _record("cand", perplexity=10.1, accuracy=0.40)

    worst, per_metric = quality_retention(baseline, candidate)
    assert worst == pytest.approx(0.5, rel=0.01)
    assert per_metric["wikitext2/perplexity"] > 0.98


def test_retention_is_direction_aware():
    """Lower perplexity is better, so an improvement must exceed 1.0."""
    baseline = _record("base", perplexity=10.0)
    better = _record("cand", perplexity=8.0)
    _, per_metric = quality_retention(baseline, better)
    assert per_metric["wikitext2/perplexity"] == pytest.approx(1.25)


# --- the staged pipeline ------------------------------------------------


def _optimizer(**kwargs) -> Optimizer:
    defaults = dict(
        model=ModelSpec(id="tiny/model"),
        constraints=Constraints(),
        objective=Objective.SIZE,
        artifacts_root=Path("artifacts"),
        compress_fn=lambda c: _artifact(c.method, GIB // 2),
        stop_early=False,
    )
    return Optimizer(**{**defaults, **kwargs})


def _small_set(n: int = 4):
    result = generate_candidates(
        qwen3_06b(), profile=BLACKWELL, methods=("int4-awq", "int8"), max_candidates=n
    )
    result.accepted = result.accepted[:n]
    return result


def test_memory_screening_happens_before_compression():
    """The whole point of the staged pipeline: a candidate that cannot fit must
    cost nothing."""
    compressed: list[str] = []

    def compress(candidate):
        compressed.append(candidate.id)
        return _artifact(candidate.method, GIB)

    optimizer = _optimizer(
        constraints=Constraints(max_vram_bytes=1),  # nothing can fit
        compress_fn=compress,
    )
    result = optimizer.run(_small_set())

    assert compressed == []
    assert not result.qualified
    assert all("exceeds" in o.violations[0] for o in result.outcomes)


def test_the_baseline_is_always_measured_first():
    """It is the reference, not a candidate. Under throughput ordering it would
    otherwise come last, leaving every candidate with nothing to compare
    against and quality constraints passing by default."""
    seen: list[str] = []

    def evaluate(target, candidate):
        seen.append(candidate.id)
        return _record("r", perplexity=10.0)

    optimizer = _optimizer(
        objective=Objective.THROUGHPUT,
        evaluate_fn=evaluate,
        benchmark_fn=lambda o: _benchmark(),
        stop_early=False,
    )
    optimizer.run(_small_set(n=4))
    assert seen[0].startswith("baseline")


def test_quality_failure_prevents_the_benchmark():
    """Expensive deployment runs happen only on candidates still in the running."""
    benchmarked: list[str] = []

    baseline = _record("base", perplexity=10.0)
    collapsed = _record("cand", perplexity=100.0)

    def evaluate(target, candidate):
        return baseline if candidate.is_baseline else collapsed

    def benchmark(outcome):
        benchmarked.append(outcome.candidate.id)
        return _benchmark()

    optimizer = _optimizer(
        constraints=Constraints(min_quality_retention=0.95),
        objective=Objective.THROUGHPUT,
        evaluate_fn=evaluate,
        benchmark_fn=benchmark,
    )
    result = optimizer.run(_small_set())

    # The baseline is benchmarked because it is the reference; nothing that
    # failed the quality floor is.
    assert all(name.startswith("baseline") for name in benchmarked)
    assert any("below" in o.violations[0] for o in result.outcomes if o.violations)


def test_a_qualifying_candidate_reaches_the_benchmark():
    baseline = _record("base", perplexity=10.0)
    good = _record("cand", perplexity=10.1)

    optimizer = _optimizer(
        constraints=Constraints(min_quality_retention=0.90),
        objective=Objective.THROUGHPUT,
        evaluate_fn=lambda target, c: baseline if c.is_baseline else good,
        benchmark_fn=lambda o: _benchmark(throughput=1500),
    )
    result = optimizer.run(_small_set())

    assert result.qualified
    best = result.recommended
    assert best is not None
    assert best.stage == "benchmarked"
    assert best.score.value == 1500


def test_early_stop_halts_after_the_first_qualifying_candidate():
    tried: list[str] = []

    baseline = _record("base", perplexity=10.0)

    def evaluate(target, candidate):
        tried.append(candidate.id)
        return baseline if candidate.is_baseline else _record("c", perplexity=10.2)

    optimizer = _optimizer(
        constraints=Constraints(min_quality_retention=0.90),
        objective=Objective.THROUGHPUT,
        evaluate_fn=evaluate,
        benchmark_fn=lambda o: _benchmark(),
        stop_early=True,
    )
    result = optimizer.run(_small_set(n=6))

    assert result.stopped_early
    assert len(result.outcomes) < 6


def test_without_early_stop_everything_is_tried():
    baseline = _record("base", perplexity=10.0)
    optimizer = _optimizer(
        evaluate_fn=lambda t, c: baseline if c.is_baseline else _record("c", perplexity=10.2),
        stop_early=False,
    )
    candidate_set = _small_set(n=5)
    result = optimizer.run(candidate_set)
    assert len(result.outcomes) == len(candidate_set.accepted)


def test_size_objective_skips_the_benchmark_entirely():
    """Ranking by size needs no deployment run, so it should not pay for one."""
    benchmarked: list[str] = []
    optimizer = _optimizer(
        objective=Objective.SIZE,
        benchmark_fn=lambda o: benchmarked.append(o.candidate.id) or _benchmark(),
        stop_early=False,
    )
    optimizer.run(_small_set())
    assert benchmarked == []


def test_a_runtime_constraint_forces_a_benchmark_even_for_size():
    benchmarked: list[str] = []

    def benchmark(outcome):
        benchmarked.append(outcome.candidate.id)
        return _benchmark(throughput=2000)

    optimizer = _optimizer(
        objective=Objective.SIZE,
        constraints=Constraints(min_throughput_tokens_per_s=100),
        benchmark_fn=benchmark,
        stop_early=False,
    )
    optimizer.run(_small_set(n=2))
    assert benchmarked


def test_a_failing_candidate_does_not_abort_the_search():
    def compress(candidate):
        if "int4" in candidate.id:
            raise RuntimeError("CUDA out of memory")
        return _artifact(candidate.method, GIB)

    optimizer = _optimizer(compress_fn=compress, stop_early=False)
    result = optimizer.run(_small_set(n=4))

    assert any(o.error for o in result.outcomes)
    assert any(o.qualified for o in result.outcomes)


def test_recommendation_explains_itself():
    baseline = _record("base", perplexity=10.0)
    optimizer = _optimizer(
        constraints=Constraints(min_quality_retention=0.9),
        objective=Objective.THROUGHPUT,
        evaluate_fn=lambda t, c: baseline if c.is_baseline else _record("c", perplexity=10.1),
        benchmark_fn=lambda o: _benchmark(throughput=1234),
        stop_early=False,
    )
    explanation = optimizer.run(_small_set()).explain()

    assert "wins on" in explanation
    assert "1234" in explanation
    assert "qualifying" in explanation


def test_no_qualifying_candidate_is_stated_plainly():
    optimizer = _optimizer(constraints=Constraints(max_vram_bytes=1))
    result = optimizer.run(_small_set())

    assert result.recommended is None
    assert "No candidate satisfied" in result.explain()


# --- the experiment cache in the search ---------------------------------


def _cached_optimizer(tmp_path, **kwargs):
    from autodistiller.store import RunStore

    baseline = _record("base", perplexity=10.0)
    defaults = dict(
        constraints=Constraints(min_quality_retention=0.90),
        objective=Objective.THROUGHPUT,
        evaluate_fn=lambda t, c: baseline if c.is_baseline else _record("c", perplexity=10.1),
        store=RunStore(tmp_path),
        stop_early=False,
        benchmark_settings={"prompt_tokens": 256, "max_tokens": 128},
    )
    return _optimizer(**{**defaults, **kwargs})


def test_a_benchmark_is_saved_so_the_next_search_can_reuse_it(tmp_path):
    optimizer = _cached_optimizer(tmp_path, benchmark_fn=lambda o: _benchmark(throughput=900))
    optimizer.run(_small_set(n=3))

    from autodistiller.store import RunStore

    saved = [r for r in RunStore(tmp_path).list_records() if r.deployment is not None]
    assert saved, "the optimizer measured a benchmark and then dropped it"
    assert all(r.benchmark_key for r in saved)
    assert all(r.candidate_id for r in saved)


def test_a_second_search_reuses_the_benchmark(tmp_path):
    """The roadmap's milestone for this phase: repeating an optimization must be
    substantially cheaper than running it the first time."""
    runs: list[str] = []

    def benchmark(outcome):
        runs.append(outcome.candidate.id)
        return _benchmark(throughput=900)

    candidates = _small_set(n=3)
    _cached_optimizer(tmp_path, benchmark_fn=benchmark).run(candidates)
    first_pass = len(runs)
    assert first_pass > 0

    result = _cached_optimizer(tmp_path, benchmark_fn=benchmark).run(candidates)

    assert len(runs) == first_pass, "re-benchmarked a candidate the cache already held"
    assert result.reused_stages.get("benchmark") == first_pass
    assert result.recommended is not None
    assert result.recommended.score.value == 900


def test_refresh_re_measures_the_benchmark(tmp_path):
    runs: list[str] = []

    def benchmark(outcome):
        runs.append(outcome.candidate.id)
        return _benchmark(throughput=900)

    candidates = _small_set(n=3)
    _cached_optimizer(tmp_path, benchmark_fn=benchmark).run(candidates)
    first_pass = len(runs)

    _cached_optimizer(tmp_path, benchmark_fn=benchmark, reuse=False).run(candidates)
    assert len(runs) == first_pass * 2


def test_a_different_request_shape_is_not_reused(tmp_path):
    """Changing what the benchmark asks the server to do changes the number, so
    the earlier measurement no longer answers the question."""
    runs: list[str] = []

    def benchmark(outcome):
        runs.append(outcome.candidate.id)
        return _benchmark(throughput=900)

    candidates = _small_set(n=3)
    _cached_optimizer(tmp_path, benchmark_fn=benchmark).run(candidates)
    first_pass = len(runs)

    _cached_optimizer(
        tmp_path,
        benchmark_fn=benchmark,
        benchmark_settings={"prompt_tokens": 256, "max_tokens": 512},
    ).run(candidates)
    assert len(runs) == first_pass * 2


def test_without_a_store_nothing_is_cached_and_nothing_breaks(tmp_path):
    """The optimizer is usable without a store; it just pays every time."""
    optimizer = _optimizer(
        constraints=Constraints(min_quality_retention=0.90),
        objective=Objective.THROUGHPUT,
        evaluate_fn=lambda t, c: _record("c", perplexity=10.0),
        benchmark_fn=lambda o: _benchmark(throughput=900),
        stop_early=False,
    )
    result = optimizer.run(_small_set(n=2))
    assert result.recommended is not None
    assert not list(tmp_path.iterdir())


def test_candidates_differing_only_in_context_keep_separate_benchmarks(tmp_path):
    """Context length is not a compression parameter, so these two share an
    artifact and an evaluation but not a benchmark. One record per evaluation
    could only hold one of them."""
    measured: dict[str, float] = {}

    def benchmark(outcome):
        result = _benchmark(throughput=100.0 * outcome.candidate.max_model_len)
        measured[outcome.candidate.id] = result.best_throughput.output_tokens_per_s
        return result

    candidates = generate_candidates(
        qwen3_06b(),
        profile=BLACKWELL,
        methods=("int8",),
        context_lengths=(2048, 4096),
        kv_dtypes=("auto",),
        max_candidates=8,
    )
    assert len({c.max_model_len for c in candidates.accepted}) > 1

    from autodistiller.store import RunStore

    _cached_optimizer(tmp_path, benchmark_fn=benchmark).run(candidates)
    assert len(measured) > 1

    saved = {
        r.candidate_id: r.deployment.best_throughput.output_tokens_per_s
        for r in RunStore(tmp_path).list_records()
        if r.deployment is not None
    }
    assert saved == measured, "a benchmark was overwritten by another candidate's"

    # And a second search reuses every one of them rather than a single survivor.
    result = _cached_optimizer(tmp_path, benchmark_fn=benchmark).run(candidates)
    assert result.reused_stages.get("benchmark") == len(measured)
