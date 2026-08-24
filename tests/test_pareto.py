"""Pareto analysis.

The milestone for this phase is that a recommendation stops being an unexplained
score, so most of what matters here is what the report *refuses* to say: it must
not rank a candidate on a number nobody measured, and it must not compare a
measured value against an estimated one.
"""

from __future__ import annotations

import pytest

from autodistiller.candidates.generator import Candidate
from autodistiller.candidates.memory import MemoryEstimate
from autodistiller.optimize.constraints import Constraints, Objective
from autodistiller.optimize.pareto import (
    QUALITY,
    THROUGHPUT,
    TTFT,
    VRAM,
    VRAM_ESTIMATED,
    ParetoReport,
    dominates,
    pareto_frontier,
)
from autodistiller.optimize.pipeline import CandidateOutcome, OptimizationResult
from autodistiller.results import ConcurrencyResult, DeploymentBenchmark, LatencyStats

GIB = 1024**3


def _stats(value: float) -> LatencyStats:
    return LatencyStats(mean=value, p50=value, p90=value, p99=value, min=value, max=value)


def _benchmark(*, throughput: float, ttft: float, vram: int) -> DeploymentBenchmark:
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
                tpot=_stats(0.005),
                output_tokens_per_s=throughput,
                peak_vram_bytes=vram,
            )
        ],
    )


def _outcome(
    name: str,
    *,
    quality: float | None = 1.0,
    throughput: float | None = None,
    ttft: float = 0.05,
    vram: int = 4 * GIB,
    weights: int = GIB,
    estimate_bytes: int = 5 * GIB,
) -> CandidateOutcome:
    # A candidate's id is derived from its method, so naming the method is how
    # a failure ends up naming the candidate the test meant.
    candidate = Candidate(
        method=None if name == "baseline" else name,
        max_model_len=2048,
        kv_dtype="auto",
        estimate=MemoryEstimate(
            weights_bytes=weights,
            kv_cache_bytes=estimate_bytes - weights,
            overhead_bytes=0,
            budget_bytes=None,
        ),
    )

    outcome = CandidateOutcome(candidate=candidate)
    outcome.quality_retention = quality
    outcome.weights_bytes = weights
    if throughput is not None:
        outcome.benchmark = _benchmark(throughput=throughput, ttft=ttft, vram=vram)
    return outcome


# --- dominance ----------------------------------------------------------


def test_better_on_every_axis_dominates():
    good = _outcome("a", quality=0.99, throughput=1500)
    bad = _outcome("b", quality=0.95, throughput=1000)
    assert dominates(good, bad, (QUALITY, THROUGHPUT))
    assert not dominates(bad, good, (QUALITY, THROUGHPUT))


def test_a_trade_off_dominates_in_neither_direction():
    """The whole reason the phase exists: neither of these is simply better."""
    quality = _outcome("a", quality=0.99, throughput=1000)
    fast = _outcome("b", quality=0.95, throughput=1500)
    assert not dominates(quality, fast, (QUALITY, THROUGHPUT))
    assert not dominates(fast, quality, (QUALITY, THROUGHPUT))


def test_an_identical_candidate_does_not_dominate():
    """Ties must not knock each other off the frontier."""
    a = _outcome("a", quality=0.98, throughput=1200)
    b = _outcome("b", quality=0.98, throughput=1200)
    assert not dominates(a, b, (QUALITY, THROUGHPUT))
    assert not dominates(b, a, (QUALITY, THROUGHPUT))


def test_equal_on_one_axis_and_better_on_another_dominates():
    a = _outcome("a", quality=0.98, throughput=1500)
    b = _outcome("b", quality=0.98, throughput=1000)
    assert dominates(a, b, (QUALITY, THROUGHPUT))


def test_lower_is_better_axes_are_read_the_right_way_round():
    quick = _outcome("a", quality=0.98, throughput=1000, ttft=0.02)
    slow = _outcome("b", quality=0.98, throughput=1000, ttft=0.20)
    assert dominates(quick, slow, (QUALITY, TTFT))
    assert not dominates(slow, quick, (QUALITY, TTFT))


def test_a_missing_measurement_never_dominates_or_is_dominated():
    """Treating an unmeasured axis as best or worst would rank on a number
    nobody produced."""
    measured = _outcome("a", quality=0.99, throughput=1500)
    unmeasured = _outcome("b", quality=0.90)
    assert not dominates(measured, unmeasured, (QUALITY, THROUGHPUT))
    assert not dominates(unmeasured, measured, (QUALITY, THROUGHPUT))


# --- frontiers ----------------------------------------------------------


def test_the_frontier_keeps_the_trade_offs_and_drops_the_losers():
    best_quality = _outcome("a", quality=0.99, throughput=800)
    fastest = _outcome("b", quality=0.93, throughput=1600)
    beaten = _outcome("c", quality=0.95, throughput=700)

    frontier = pareto_frontier([best_quality, fastest, beaten], (QUALITY, THROUGHPUT))

    assert set(id(o) for o in frontier.optimal) == {id(best_quality), id(fastest)}
    assert [id(o) for o in frontier.dominated] == [id(beaten)]


def test_unmeasured_candidates_are_set_aside_rather_than_judged():
    measured = _outcome("a", quality=0.99, throughput=1500)
    screened_only = _outcome("b", quality=0.99)

    frontier = pareto_frontier([measured, screened_only], (QUALITY, THROUGHPUT))

    assert [id(o) for o in frontier.optimal] == [id(measured)]
    assert [id(o) for o in frontier.incomparable] == [id(screened_only)]
    assert not frontier.dominated


def test_an_empty_set_produces_an_empty_frontier():
    frontier = pareto_frontier([], (QUALITY, THROUGHPUT))
    assert not frontier.optimal and not frontier.dominated


def test_a_single_candidate_is_trivially_optimal():
    only = _outcome("a", quality=0.99, throughput=1000)
    assert pareto_frontier([only], (QUALITY, THROUGHPUT)).optimal == [only]


def test_the_frontier_describes_itself():
    frontier = pareto_frontier(
        [_outcome("a", quality=0.99, throughput=1500), _outcome("b", quality=0.9)],
        (QUALITY, THROUGHPUT),
    )
    assert "1 optimal of 1 compared" in frontier.describe()
    assert "1 not measured" in frontier.describe()


# --- the report ---------------------------------------------------------


def test_vram_uses_measurements_when_any_exist():
    report = ParetoReport([_outcome("a", quality=0.99, throughput=1000, vram=6 * GIB)])
    assert report.vram_axis is VRAM
    assert not report.vram_axis.estimated


def test_vram_falls_back_to_estimates_when_nothing_was_benchmarked():
    """An estimate is still useful. Silently comparing one against a measured
    peak would not be."""
    report = ParetoReport([_outcome("a", quality=0.99), _outcome("b", quality=0.95)])
    assert report.vram_axis is VRAM_ESTIMATED
    assert report.vram_axis.estimated


def test_an_axis_never_mixes_measured_and_estimated_values():
    """With one benchmark present the axis is measured, and the unbenchmarked
    candidate becomes incomparable rather than being scored on its estimate."""
    benchmarked = _outcome("a", quality=0.99, throughput=1000, vram=6 * GIB)
    screened = _outcome("b", quality=0.95)

    report = ParetoReport([benchmarked, screened])
    assert report.vram_axis is VRAM
    assert report.vram_axis.value(screened) is None


def test_the_three_named_trade_offs_are_all_produced():
    report = ParetoReport([_outcome("a", quality=0.99, throughput=1000)])
    labels = [t.labels for t in report.trade_offs]
    assert len(labels) == 3
    assert any("VRAM" in label for label in labels)
    assert any("TTFT" in label for label in labels)
    assert any("throughput" in label.lower() for label in labels)


# --- recommendations ----------------------------------------------------


def _report() -> ParetoReport:
    return ParetoReport(
        [
            _outcome("a", quality=0.99, throughput=800, ttft=0.10, weights=2 * GIB),
            _outcome("b", quality=0.93, throughput=1600, ttft=0.02, weights=1 * GIB),
        ]
    )


def test_each_objective_picks_its_own_winner():
    picks = {r.objective: r.outcome for r in _report().recommendations}

    assert picks[Objective.QUALITY].quality_retention == 0.99
    assert picks[Objective.THROUGHPUT].benchmark.best_throughput.output_tokens_per_s == 1600
    assert picks[Objective.LATENCY].benchmark.single_stream.ttft.p50 == 0.02
    assert picks[Objective.SIZE].weights_bytes == GIB


def test_the_four_roadmap_options_are_all_offered():
    offered = {r.objective for r in _report().recommendations}
    assert {Objective.QUALITY, Objective.THROUGHPUT, Objective.SIZE, Objective.BALANCED} <= offered


def test_a_recommendation_says_what_it_gives_up():
    """The milestone: not a score, but a trade-off the reader can check."""
    fastest = next(r for r in _report().recommendations if r.objective is Objective.THROUGHPUT)
    assert "quality retention" in fastest.trade_off
    assert "93.00%" in fastest.trade_off
    assert "99.00%" in fastest.trade_off


def test_the_best_on_an_axis_gives_up_nothing_on_that_axis():
    best = next(r for r in _report().recommendations if r.objective is Objective.QUALITY)
    assert "quality retention" not in best.trade_off


def test_recommendations_report_whether_they_are_pareto_optimal():
    assert all(r.on_frontier for r in _report().recommendations)


def test_an_unmeasurable_objective_is_omitted_rather_than_scored_zero():
    """ "The fastest configuration" is not a claim you can make about candidates
    nobody benchmarked."""
    report = ParetoReport([_outcome("a", quality=0.99), _outcome("b", quality=0.95)])
    offered = {r.objective for r in report.recommendations}

    assert Objective.QUALITY in offered
    assert Objective.SIZE in offered
    assert Objective.THROUGHPUT not in offered
    assert Objective.LATENCY not in offered


def test_recommendation_for_finds_one_by_objective():
    report = _report()
    assert report.recommendation_for(Objective.QUALITY) is not None
    assert report.recommendation_for(Objective.LATENCY) is not None


def test_no_qualifying_candidate_is_stated_plainly():
    assert "No qualifying configuration" in ParetoReport([]).explain()


def test_a_single_candidate_points_at_no_stop_early():
    """Early stopping and trade-off analysis pull against each other, and the
    report should say so rather than showing a frontier of one."""
    explanation = ParetoReport([_outcome("a", quality=0.99, throughput=1000)]).explain()
    assert "--no-stop-early" in explanation


def test_the_explanation_names_every_option_and_its_cost():
    explanation = _report().explain()

    for label in ("best quality", "fastest (throughput)", "smallest", "balanced"):
        assert label in explanation
    assert "optimal of" in explanation  # how much of the set survived domination
    assert "against a best of" in explanation  # what each option gives up


# --- wiring into a search -----------------------------------------------


def test_the_result_builds_a_report_from_qualifying_candidates_only():
    good = _outcome("a", quality=0.99, throughput=1000)
    rejected = _outcome("b", quality=0.50, throughput=2000)
    rejected.violations = ["quality retention 50.00% is below the 90.0% floor"]

    result = OptimizationResult(
        model_id="m",
        objective=Objective.BALANCED,
        constraints=Constraints(),
        backend="vllm",
        outcomes=[good, rejected],
    )
    report = result.pareto()

    assert [id(o) for o in report.outcomes] == [id(good)]


@pytest.mark.parametrize("objective", list(Objective))
def test_asking_for_any_objective_is_safe_when_nothing_was_measured(objective):
    """A search that only screened must answer, not raise -- and must only offer
    the one objective its evidence supports."""
    report = ParetoReport([_outcome("a", quality=None)])
    assert isinstance(report.explain(), str)

    recommendation = report.recommendation_for(objective)
    if objective is Objective.SIZE:
        assert recommendation is not None  # artifact size needs no benchmark
    else:
        assert recommendation is None


def test_the_frontier_is_drawn_over_the_axes_that_have_data():
    """A search that skipped benchmarking still has a quality/VRAM trade-off.
    Keeping unmeasured axes in the set would make every candidate incomparable
    and the frontier empty -- true, and useless."""
    report = ParetoReport(
        [
            _outcome("a", quality=0.99, estimate_bytes=6 * GIB),
            _outcome("b", quality=0.95, estimate_bytes=3 * GIB),
        ]
    )

    assert [axis.key for axis in report.axes] == ["quality", "vram"]
    frontier = report.frontier
    assert len(frontier.optimal) == 2  # neither dominates: a is better, b is smaller
    assert not frontier.incomparable


def test_a_measured_axis_is_kept_even_when_only_some_candidates_have_it():
    """Dropping the axis would discard a real measurement; keeping it makes the
    unmeasured candidate incomparable, which is the honest answer."""
    report = ParetoReport(
        [_outcome("a", quality=0.99, throughput=1200), _outcome("b", quality=0.95)]
    )

    assert "throughput" in [axis.key for axis in report.axes]
    assert len(report.frontier.incomparable) == 1


def test_a_screened_only_search_still_trades_off_on_memory():
    """Every candidate carries a memory estimate, so even a search that measured
    nothing else can still be ranked by what it would cost to serve."""
    report = ParetoReport(
        [
            _outcome("a", quality=None, estimate_bytes=6 * GIB),
            _outcome("b", quality=None, estimate_bytes=3 * GIB),
        ]
    )

    assert [axis.key for axis in report.axes] == ["vram"]
    assert len(report.frontier.optimal) == 1  # the smaller one simply wins


def test_a_frontier_over_no_axes_calls_nothing_optimal():
    """Otherwise every candidate comes back optimal, which is the opposite of
    what having no axis to judge on means."""
    frontier = pareto_frontier([_outcome("a"), _outcome("b")], ())
    assert not frontier.optimal
    assert len(frontier.incomparable) == 2


def test_a_tie_on_the_objective_breaks_towards_the_frontier():
    """Two candidates can score identically on one objective while one is beaten
    outright on every other axis. Recommending the beaten one would be wrong for
    a reason the score itself cannot see."""
    dominated = _outcome("slow", quality=0.98, throughput=900, ttft=0.20, vram=7 * GIB)
    optimal = _outcome("fast", quality=0.98, throughput=900, ttft=0.02, vram=3 * GIB)

    report = ParetoReport([dominated, optimal])
    assert [id(o) for o in report.frontier.optimal] == [id(optimal)]

    best = report.recommendation_for(Objective.QUALITY)
    assert id(best.outcome) == id(optimal)
    assert best.on_frontier
