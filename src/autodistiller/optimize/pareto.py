"""Pareto analysis.

A single score can rank configurations but cannot explain them. "int4-awq wins
with 0.94" says nothing about what was given up to get there. This module
answers the question the roadmap actually poses -- what does more quality cost
in memory, in latency, in throughput -- by finding the configurations where you
cannot improve one axis without losing another, and naming the ones a user is
most likely to want.

Nothing here measures anything. Every number was already recorded on the
candidate outcomes by the search; this is the view over them, which is why it
costs nothing to compute and can be re-derived from stored records later.

Two rules keep the view honest:

* **Only comparable candidates are ranked.** A candidate that was never
  benchmarked has no throughput, and guessing one -- as either the best or the
  worst value -- would put it on the frontier or off it for a reason that is not
  a measurement. Those candidates are listed separately instead.
* **An axis never mixes measured and estimated values.** Peak VRAM from a real
  serving run and VRAM predicted by arithmetic are different quantities. When no
  benchmark exists the whole axis falls back to estimates and says so; it never
  compares one against the other.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from .constraints import Objective, Score, score_candidate

if TYPE_CHECKING:  # pipeline imports this module, so the outcome type is a cycle
    from .pipeline import CandidateOutcome

BYTES_PER_GIB = 1024**3

RECOMMENDED_ORDER = (
    Objective.QUALITY,
    Objective.THROUGHPUT,
    Objective.LATENCY,
    Objective.SIZE,
    Objective.BALANCED,
)
"""Presentation order: the roadmap's own list -- best-quality, fastest,
smallest, balanced -- rather than however the enum happens to be declared.
Latency sits next to throughput because both answer "fastest".
"""


@dataclass(frozen=True)
class Axis:
    """One dimension of the trade-off, and how to read it off an outcome."""

    key: str
    label: str
    higher_is_better: bool
    extract: Callable[[CandidateOutcome], float | None]
    render: Callable[[float], str]
    estimated: bool = False
    """Whether this axis carries predictions rather than measurements."""

    def value(self, outcome: CandidateOutcome) -> float | None:
        return self.extract(outcome)

    def format(self, outcome: CandidateOutcome) -> str:
        value = self.value(outcome)
        return self.render(value) if value is not None else "-"

    def better(self, a: float, b: float) -> bool:
        return a > b if self.higher_is_better else a < b


def _gib(value: float) -> str:
    return f"{value / BYTES_PER_GIB:.2f} GiB"


def _peak_vram(outcome: CandidateOutcome) -> float | None:
    benchmark = outcome.benchmark
    if benchmark is None or not benchmark.peak_vram_bytes:
        return None
    return float(benchmark.peak_vram_bytes)


def _estimated_vram(outcome: CandidateOutcome) -> float | None:
    estimate = getattr(outcome.candidate, "estimate", None)
    return float(estimate.total_bytes) if estimate is not None else None


def _ttft(outcome: CandidateOutcome) -> float | None:
    benchmark = outcome.benchmark
    single = benchmark.single_stream if benchmark else None
    return single.ttft.p50 if single is not None and single.ttft is not None else None


def _throughput(outcome: CandidateOutcome) -> float | None:
    benchmark = outcome.benchmark
    best = benchmark.best_throughput if benchmark else None
    return best.output_tokens_per_s if best is not None else None


QUALITY = Axis(
    key="quality",
    label="Quality retention",
    higher_is_better=True,
    extract=lambda o: o.quality_retention,
    render=lambda v: f"{v * 100:.2f}%",
)

VRAM = Axis(
    key="vram",
    label="Peak VRAM",
    higher_is_better=False,
    extract=_peak_vram,
    render=_gib,
)

VRAM_ESTIMATED = Axis(
    key="vram",
    label="VRAM (estimated)",
    higher_is_better=False,
    extract=_estimated_vram,
    render=_gib,
    estimated=True,
)

TTFT = Axis(
    key="ttft",
    label="TTFT p50",
    higher_is_better=False,
    extract=_ttft,
    render=lambda v: f"{v * 1000:.0f}ms",
)

THROUGHPUT = Axis(
    key="throughput",
    label="Peak throughput",
    higher_is_better=True,
    extract=_throughput,
    render=lambda v: f"{v:.0f} tok/s",
)

SIZE = Axis(
    key="size",
    label="Artifact size",
    higher_is_better=False,
    extract=lambda o: float(o.weights_bytes) if o.weights_bytes else None,
    render=_gib,
)


def dominates(a: CandidateOutcome, b: CandidateOutcome, axes: Sequence[Axis]) -> bool:
    """Whether ``a`` is at least as good as ``b`` everywhere and better somewhere.

    Both must have a value on every axis; callers filter first. A candidate that
    merely ties on all axes does not dominate, which is what keeps duplicate
    configurations from knocking each other off the frontier.
    """
    strictly_better = False
    for axis in axes:
        left, right = axis.value(a), axis.value(b)
        if left is None or right is None:
            return False
        if axis.better(right, left):
            return False
        if axis.better(left, right):
            strictly_better = True
    return strictly_better


@dataclass
class Frontier:
    """The Pareto-optimal set over some axes, and everything it excluded."""

    axes: tuple[Axis, ...]
    optimal: list[CandidateOutcome] = field(default_factory=list)
    dominated: list[CandidateOutcome] = field(default_factory=list)
    incomparable: list[CandidateOutcome] = field(default_factory=list)
    """Candidates missing a measurement on at least one axis. Not judged, since
    the only basis for judging them would be a value nobody measured."""

    @property
    def labels(self) -> str:
        return " vs ".join(axis.label for axis in self.axes)

    def describe(self) -> str:
        text = f"{len(self.optimal)} optimal of {len(self.optimal) + len(self.dominated)} compared"
        if self.incomparable:
            text += f", {len(self.incomparable)} not measured on every axis"
        return text


def pareto_frontier(outcomes: Iterable[CandidateOutcome], axes: Sequence[Axis]) -> Frontier:
    """Partition candidates into Pareto-optimal, dominated, and incomparable.

    O(n^2), which is the right algorithm here: the search space is capped at a
    dozen or so candidates by design, and a smarter sweep would cost more to read
    than it saves to run.
    """
    axes = tuple(axes)
    frontier = Frontier(axes=axes)

    if not axes:
        # Nothing to compare on. Every candidate would otherwise come back
        # "optimal", which is the opposite of what an empty axis set means.
        frontier.incomparable.extend(outcomes)
        return frontier

    comparable: list[CandidateOutcome] = []
    for outcome in outcomes:
        if all(axis.value(outcome) is not None for axis in axes):
            comparable.append(outcome)
        else:
            frontier.incomparable.append(outcome)

    for outcome in comparable:
        if any(dominates(other, outcome, axes) for other in comparable if other is not outcome):
            frontier.dominated.append(outcome)
        else:
            frontier.optimal.append(outcome)

    return frontier


@dataclass
class Recommendation:
    """One named option, and what choosing it costs."""

    objective: Objective
    outcome: CandidateOutcome
    score: Score
    on_frontier: bool
    trade_off: str = ""

    @property
    def label(self) -> str:
        return {
            Objective.QUALITY: "best quality",
            Objective.THROUGHPUT: "fastest (throughput)",
            Objective.LATENCY: "fastest (latency)",
            Objective.SIZE: "smallest",
            Objective.BALANCED: "balanced",
        }[self.objective]

    def describe(self) -> str:
        text = f"{self.label}: {self.outcome.candidate.id} - {self.score.basis} {self.score.detail}"
        return f"{text}; {self.trade_off}" if self.trade_off else text


def _trade_off_note(
    outcome: CandidateOutcome, others: Sequence[CandidateOutcome], axes: Sequence[Axis]
) -> str:
    """What this option gives up against the best available on each axis.

    The point of the phase: a recommendation should say what it costs, not only
    what it wins.
    """
    notes: list[str] = []
    for axis in axes:
        mine = axis.value(outcome)
        if mine is None:
            continue
        values = [v for o in others if (v := axis.value(o)) is not None]
        if not values:
            continue
        best = max(values) if axis.higher_is_better else min(values)
        if best == mine or not axis.better(best, mine):
            continue
        notes.append(
            f"{axis.label.lower()} {axis.render(mine)} against a best of {axis.render(best)}"
        )
    return "; ".join(notes)


class ParetoReport:
    """Trade-offs, frontiers and named options for one search.

    Built from candidate outcomes alone, so the same report can be produced from
    a fresh search or from records the experiment cache already holds.
    """

    def __init__(self, outcomes: Sequence[CandidateOutcome]) -> None:
        self.outcomes = list(outcomes)

        # Never mix a measured peak against a predicted one. If nothing was
        # benchmarked the whole axis becomes estimates and is labelled as such.
        measured = any(_peak_vram(o) is not None for o in self.outcomes)
        self.vram_axis = VRAM if measured else VRAM_ESTIMATED

        # Only axes something was actually measured on. Keeping latency and
        # throughput in the set when no candidate was benchmarked would make
        # every candidate incomparable and the frontier empty -- technically
        # true, and useless. The frontier should be drawn over the evidence that
        # exists, and say which axes those were.
        candidates = (QUALITY, self.vram_axis, TTFT, THROUGHPUT)
        self.axes: tuple[Axis, ...] = tuple(
            axis
            for axis in candidates
            if any(axis.value(outcome) is not None for outcome in self.outcomes)
        )

    @property
    def trade_offs(self) -> list[Frontier]:
        """The three the roadmap names: quality against each cost axis."""
        return [
            pareto_frontier(self.outcomes, (QUALITY, self.vram_axis)),
            pareto_frontier(self.outcomes, (QUALITY, TTFT)),
            pareto_frontier(self.outcomes, (QUALITY, THROUGHPUT)),
        ]

    @property
    def frontier(self) -> Frontier:
        """Optimal across every axis at once, not just a pair."""
        return pareto_frontier(self.outcomes, self.axes)

    @property
    def recommendations(self) -> list[Recommendation]:
        """Best-quality, fastest, smallest and balanced, each with its cost.

        An objective whose score could not be measured is left out rather than
        answered with a zero: "the fastest configuration" is not a claim that can
        be made from candidates nobody benchmarked.
        """
        optimal = self.frontier.optimal
        found: list[Recommendation] = []

        for objective in RECOMMENDED_ORDER:
            scored = [(o, score_candidate(o, objective)) for o in self.outcomes]
            usable = [(o, s) for o, s in scored if s.value > 0 and s.basis != "not benchmarked"]
            if not usable:
                continue

            # Ties break towards the frontier. Two candidates can score
            # identically on one objective while one of them is beaten outright
            # on every other axis -- recommending that one would be indefensible
            # for a reason the score cannot see.
            outcome, score = max(
                usable,
                key=lambda pair: (pair[1].value, any(o is pair[0] for o in optimal)),
            )
            others = [o for o in self.outcomes if o is not outcome]
            found.append(
                Recommendation(
                    objective=objective,
                    outcome=outcome,
                    score=score,
                    on_frontier=any(o is outcome for o in optimal),
                    trade_off=_trade_off_note(outcome, others, self.axes),
                )
            )

        return found

    def recommendation_for(self, objective: Objective) -> Recommendation | None:
        return next((r for r in self.recommendations if r.objective is objective), None)

    def explain(self) -> str:
        """The whole point of the phase, in one paragraph."""
        frontier = self.frontier
        if not self.outcomes:
            return "No qualifying configuration to compare."
        if len(self.outcomes) == 1:
            return (
                f"Only {self.outcomes[0].candidate.id} qualified, so there is no trade-off to "
                f"show. Re-run with --no-stop-early to measure the alternatives."
            )

        lines = [f"{frontier.describe()} on {frontier.labels}."]
        lines += [f"  {r.describe()}." for r in self.recommendations]
        return "\n".join(lines)


__all__ = [
    "QUALITY",
    "RECOMMENDED_ORDER",
    "SIZE",
    "THROUGHPUT",
    "TTFT",
    "VRAM",
    "VRAM_ESTIMATED",
    "Axis",
    "Frontier",
    "ParetoReport",
    "Recommendation",
    "dominates",
    "pareto_frontier",
]
