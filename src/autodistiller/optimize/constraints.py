"""Constraints and objectives.

A constraint says whether a candidate is *acceptable*; an objective says which
of the acceptable ones is *best*. Keeping them apart is what makes the final
recommendation explainable: "it qualified because quality held at 98.4% and it
fit in 6.1 GiB, and it won because it was the fastest of the four that
qualified."

The objective also decides the order candidates are tried in, which is what
makes stopping early meaningful. Searching least-lossy-first and stopping at the
first qualifying candidate would answer "which is the safest configuration that
works" -- the right answer for ``quality`` and the wrong one for ``throughput``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from ..candidates.generator import Candidate
from ..compression.methods import METHODS


class Objective(str, Enum):
    """What the user is optimizing for."""

    THROUGHPUT = "throughput"
    LATENCY = "latency"
    SIZE = "size"
    QUALITY = "quality"
    BALANCED = "balanced"

    @property
    def prefers_compression(self) -> bool:
        """Whether smaller weights tend to serve this objective.

        Determines search order: for these, the most compressed candidate is the
        most likely winner, so it is worth proving first.
        """
        return self in (Objective.THROUGHPUT, Objective.LATENCY, Objective.SIZE)


@dataclass(frozen=True)
class Constraints:
    """Limits a candidate must satisfy to be recommendable at all."""

    min_quality_retention: float | None = None
    max_vram_bytes: int | None = None
    max_ttft_s: float | None = None
    max_tpot_s: float | None = None
    min_throughput_tokens_per_s: float | None = None

    max_request_latency_s: float | None = None
    """p99 end-to-end latency of one request.

    The generation constraints above are token-shaped, and a model that answers
    once has neither a first token nor a rate between tokens. What an embedding
    or reranking workload is held to is how long the whole call takes, at the
    tail rather than the median -- a p50 that clears the budget while p99 does
    not is a service that misses its budget on every hundredth request.
    """

    min_requests_per_s: float | None = None
    """Throughput in requests rather than tokens, for the same reason."""

    @property
    def needs_benchmark(self) -> bool:
        """Whether any constraint can only be settled by a deployment run."""
        return any(
            v is not None
            for v in (
                self.max_ttft_s,
                self.max_tpot_s,
                self.min_throughput_tokens_per_s,
                self.max_request_latency_s,
                self.min_requests_per_s,
            )
        )

    def describe(self) -> str:
        parts = []
        if self.min_quality_retention is not None:
            parts.append(f"quality >= {self.min_quality_retention * 100:.1f}%")
        if self.max_vram_bytes is not None:
            parts.append(f"VRAM <= {self.max_vram_bytes / 1024**3:.1f} GiB")
        if self.max_ttft_s is not None:
            parts.append(f"TTFT <= {self.max_ttft_s * 1000:.0f}ms")
        if self.max_tpot_s is not None:
            parts.append(f"TPOT <= {self.max_tpot_s * 1000:.1f}ms")
        if self.min_throughput_tokens_per_s is not None:
            parts.append(f"throughput >= {self.min_throughput_tokens_per_s:.0f} tok/s")
        if self.max_request_latency_s is not None:
            parts.append(f"latency p99 <= {self.max_request_latency_s * 1000:.0f}ms")
        if self.min_requests_per_s is not None:
            parts.append(f"throughput >= {self.min_requests_per_s:.1f} req/s")
        return ", ".join(parts) or "none"

    # -- checks, applied as evidence arrives -----------------------------

    def check_memory(self, estimated_bytes: int) -> list[str]:
        """Screen on the estimate, before anything expensive runs."""
        if self.max_vram_bytes is None or estimated_bytes <= self.max_vram_bytes:
            return []
        return [
            f"estimated {estimated_bytes / 1024**3:.2f} GiB exceeds the "
            f"{self.max_vram_bytes / 1024**3:.2f} GiB budget"
        ]

    def check_quality(self, retention: float | None, *, measurable: bool = True) -> list[str]:
        """Whether a candidate clears the quality floor.

        ``measurable`` is False when nothing could be compared against -- no
        baseline was evaluated, so there is no retention figure and never will
        be for this candidate. That is a violation rather than a pass: the user
        asked for a floor, and a floor that cannot be checked has not been met.
        Reporting it as satisfied is the worst of the three options, because
        nothing in the output looks wrong.

        ``retention is None`` while still measurable is different -- it means
        the comparison has not happened *yet*, as during memory screening -- and
        stays silent so a candidate is not rejected before it is measured.
        """
        if self.min_quality_retention is None:
            return []

        if not measurable:
            return [
                f"quality could not be measured, so the "
                f"{self.min_quality_retention * 100:.1f}% floor cannot be verified "
                f"(no baseline was evaluated to compare against)"
            ]

        if retention is None or retention >= self.min_quality_retention:
            return []
        return [
            f"quality retention {retention * 100:.2f}% is below the "
            f"{self.min_quality_retention * 100:.1f}% floor"
        ]

    def quality_warning(self, comparison) -> str | None:
        """Whether the quality measurement is precise enough to mean what it says.

        :meth:`check_quality` compares point estimates, which is the right
        decision rule but reads as more certain than the data. Perplexity on a
        handful of documents carries a standard error that can rival the value,
        and two such numbers make a ratio that cannot settle a 95% floor. The
        verdict still stands -- refusing to decide would be worse -- but a reader
        should know when it rests on a difference the measurement cannot see.

        The giveaway in practice is retention above 100%: compression does not
        improve a model, so anything over 1.0 is noise being read as a result.
        """
        retention, stderr = comparison.retention, comparison.stderr
        if retention is None or stderr is None or stderr <= 0:
            return None

        if self.min_quality_retention is not None and comparison.indistinguishable_from(
            self.min_quality_retention
        ):
            return (
                f"quality retention {comparison.describe()} cannot be told apart from the "
                f"{self.min_quality_retention * 100:.1f}% floor at this sample size; "
                f"raise --limit before trusting the verdict"
            )

        if comparison.indistinguishable_from(1.0):
            return (
                f"quality retention {comparison.describe()} is within noise of the baseline; "
                f"the difference is not measurable at this sample size"
            )

        return None

    def check_benchmark(self, benchmark) -> list[str]:
        """Check the constraints only a real deployment run can settle."""
        if benchmark is None:
            return []

        violations: list[str] = []
        single = benchmark.single_stream
        best = benchmark.best_throughput

        has_ttft = single is not None and single.ttft is not None
        if self.max_ttft_s is not None and has_ttft and single.ttft.p50 > self.max_ttft_s:
            violations.append(
                f"TTFT p50 {single.ttft.p50 * 1000:.0f}ms exceeds {self.max_ttft_s * 1000:.0f}ms"
            )

        has_tpot = single is not None and single.tpot is not None
        if self.max_tpot_s is not None and has_tpot and single.tpot.p50 > self.max_tpot_s:
            violations.append(
                f"TPOT p50 {single.tpot.p50 * 1000:.1f}ms exceeds {self.max_tpot_s * 1000:.1f}ms"
            )

        if (
            self.min_throughput_tokens_per_s is not None
            and best is not None
            and best.output_tokens_per_s < self.min_throughput_tokens_per_s
        ):
            violations.append(
                f"peak throughput {best.output_tokens_per_s:.0f} tok/s is below "
                f"{self.min_throughput_tokens_per_s:.0f} tok/s"
            )

        # Measured for every backend, and the only latency figure a
        # non-streaming endpoint produces at all.
        has_latency = single is not None and single.request_latency is not None
        if (
            self.max_request_latency_s is not None
            and has_latency
            and single.request_latency.p99 > self.max_request_latency_s
        ):
            violations.append(
                f"request latency p99 {single.request_latency.p99 * 1000:.0f}ms exceeds "
                f"{self.max_request_latency_s * 1000:.0f}ms"
            )

        if (
            self.min_requests_per_s is not None
            and best is not None
            and best.requests_per_s < self.min_requests_per_s
        ):
            violations.append(
                f"peak throughput {best.requests_per_s:.2f} req/s is below "
                f"{self.min_requests_per_s:.2f} req/s"
            )

        # Measured VRAM supersedes the estimate once it exists.
        if (
            self.max_vram_bytes is not None
            and benchmark.peak_vram_bytes
            and benchmark.peak_vram_bytes > self.max_vram_bytes
        ):
            violations.append(
                f"measured peak VRAM {benchmark.peak_vram_bytes / 1024**3:.2f} GiB "
                f"exceeds the {self.max_vram_bytes / 1024**3:.2f} GiB budget"
            )

        return violations


@dataclass
class Score:
    """Why a candidate ranked where it did."""

    value: float
    basis: str
    detail: str = ""

    def __lt__(self, other: Score) -> bool:
        return self.value < other.value


def _weight_bits(candidate: Candidate) -> int:
    if candidate.method is None:
        return 16
    return METHODS[candidate.method].weight_bits


def search_order(candidates: list[Candidate], objective: Objective) -> list[Candidate]:
    """Order candidates so the likely winner is proven first.

    This is what makes early stopping honest. Under ``throughput`` the most
    compressed candidate is tried first, so the first one that holds quality is
    also the fastest one that holds quality. Under ``quality`` the order
    reverses, and the first qualifying candidate is the least lossy.
    """
    if objective.prefers_compression:
        return sorted(
            candidates,
            key=lambda c: (_weight_bits(c), -c.max_model_len, c.id),
        )
    return sorted(candidates, key=lambda c: (-_weight_bits(c), c.max_model_len, c.id))


def score_candidate(outcome, objective: Objective) -> Score:
    """Rank a measured candidate. Higher is better, always."""
    benchmark = outcome.benchmark
    retention = outcome.quality_retention

    if objective is Objective.QUALITY:
        return Score(
            value=retention if retention is not None else 0.0,
            basis="quality retention",
            detail=f"{(retention or 0) * 100:.2f}%",
        )

    if objective is Objective.SIZE:
        # Smaller is better, so invert. Weight bytes are known without a
        # benchmark, which is why size can be ranked from screening alone.
        weights = outcome.weights_bytes or 1
        return Score(
            value=1.0 / weights,
            basis="artifact size",
            detail=f"{weights / 1024**3:.2f} GiB",
        )

    if benchmark is None:
        return Score(value=0.0, basis="not benchmarked", detail="no deployment measurement")

    if objective is Objective.THROUGHPUT:
        best = benchmark.best_throughput
        value = best.throughput if best else 0.0
        unit = best.throughput_unit if best else "tok/s"
        return Score(value=value, basis="peak throughput", detail=f"{value:.0f} {unit}")

    if objective is Objective.LATENCY:
        single = benchmark.single_stream
        latency = single.latency_p50 if single is not None else None
        if latency is None:
            return Score(value=0.0, basis="latency", detail="not measured")
        basis = "single-stream TTFT" if single.ttft is not None else "single-stream latency"
        return Score(
            value=1.0 / max(latency, 1e-6),
            basis=basis,
            detail=f"{latency * 1000:.0f}ms",
        )

    # Balanced: quality retention against normalized throughput. Deliberately
    # simple -- Phase 7's Pareto view is the honest way to show a trade-off, and
    # a single blended number is only a tie-breaker.
    best = benchmark.best_throughput
    throughput = best.throughput if best else 0.0
    unit = best.throughput_unit if best else "tok/s"
    quality = retention if retention is not None else 0.0
    return Score(
        value=quality * throughput,
        basis="quality x throughput",
        detail=f"{quality * 100:.1f}% x {throughput:.0f} {unit}",
    )


@dataclass
class ConstraintReport:
    """Everything checked against one candidate."""

    satisfied: bool
    violations: list[str] = field(default_factory=list)

    @classmethod
    def from_violations(cls, violations: list[str]) -> ConstraintReport:
        return cls(satisfied=not violations, violations=violations)


__all__ = [
    "ConstraintReport",
    "Constraints",
    "Objective",
    "Score",
    "score_candidate",
    "search_order",
]
