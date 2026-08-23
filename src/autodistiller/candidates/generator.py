"""Candidate generation.

Enumerates a small, explainable search space: compression method x context
length x KV cache dtype, filtered by what the hardware supports, what the
serving backend can run, and what fits in memory.

Two properties matter more than cleverness here.

**Small.** The roadmap asks for roughly 15-25 candidates, not a combinatorial
explosion. Every candidate costs a compression run and a benchmark later, so the
generator's job is to be selective before anything expensive happens.

**Explainable.** Rejected candidates are kept with their reasons. "Your GPU has
no FP8" is a useful answer; a silently shorter list is not.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..compression.methods import METHODS, CompressionMethod, check_method, resolve_method
from ..metadata.profiles import GPUProfile
from .memory import MemoryEstimate, estimate_memory
from .shape import ModelShape

DEFAULT_CONTEXT_LENGTHS = (2048, 4096, 8192)
DEFAULT_MAX_CANDIDATES = 25

KV_DTYPES = ("auto", "fp8")
"""An FP8 KV cache halves the dominant cost of long-context serving, and is a
separate decision from how the weights are stored."""


@dataclass(frozen=True)
class Candidate:
    """One configuration worth measuring."""

    method: str | None
    max_model_len: int
    kv_dtype: str
    estimate: MemoryEstimate

    @property
    def is_baseline(self) -> bool:
        return self.method is None

    @property
    def id(self) -> str:
        method = self.method or "baseline"
        kv = "" if self.kv_dtype == "auto" else f"-kv{self.kv_dtype}"
        return f"{method}-ctx{self.max_model_len}{kv}"

    def describe(self) -> str:
        return f"{self.id}: {self.estimate.describe()}"


@dataclass(frozen=True)
class Rejection:
    """A candidate that was not generated, and why."""

    candidate: Candidate
    reasons: tuple[str, ...]


@dataclass
class CandidateSet:
    """The generated search space, including what was excluded."""

    model_id: str
    shape: ModelShape
    backend: str
    profile: GPUProfile | None
    budget_bytes: int | None
    concurrency: int
    accepted: list[Candidate] = field(default_factory=list)
    rejected: list[Rejection] = field(default_factory=list)

    @property
    def n_considered(self) -> int:
        return len(self.accepted) + len(self.rejected)

    @property
    def baseline(self) -> Candidate | None:
        return next((c for c in self.accepted if c.is_baseline), None)

    def rejection_summary(self) -> dict[str, int]:
        """How many candidates each reason removed, most common first."""
        counts: dict[str, int] = {}
        for rejection in self.rejected:
            for reason in rejection.reasons:
                key = reason.split(":")[0].strip()
                counts[key] = counts.get(key, 0) + 1
        return dict(sorted(counts.items(), key=lambda kv: -kv[1]))


def _sort_key(candidate: Candidate) -> tuple:
    """Least lossy first.

    Phase 5 screens cheaply before benchmarking, and the candidate most likely
    to hold quality is the one worth proving first. The baseline leads because
    everything else is measured against it.
    """
    if candidate.method is None:
        return (0, 0, 0, candidate.max_model_len)
    method = METHODS[candidate.method]
    return (
        1,
        -method.weight_bits,
        -method.activation_bits,
        candidate.max_model_len,
    )


def _context_lengths(shape: ModelShape, requested: tuple[int, ...] | None) -> list[int]:
    lengths = requested or DEFAULT_CONTEXT_LENGTHS
    # A context longer than the model's positional range is not a candidate,
    # it is a misconfiguration.
    usable = [n for n in lengths if n <= shape.max_position_embeddings]
    return usable or [min(shape.max_position_embeddings, DEFAULT_CONTEXT_LENGTHS[0])]


def generate_candidates(
    shape: ModelShape,
    *,
    backend: str = "vllm",
    profile: GPUProfile | None = None,
    budget_bytes: int | None = None,
    methods: tuple[str, ...] | None = None,
    context_lengths: tuple[int, ...] | None = None,
    kv_dtypes: tuple[str, ...] = KV_DTYPES,
    concurrency: int = 1,
    include_baseline: bool = True,
    max_candidates: int = DEFAULT_MAX_CANDIDATES,
) -> CandidateSet:
    """Enumerate and filter the search space for one model."""
    if budget_bytes is None and profile is not None:
        budget_bytes = profile.vram_bytes

    chosen: list[CompressionMethod | None] = [None] if include_baseline else []
    names = methods or tuple(METHODS)
    chosen += [resolve_method(name) for name in names]

    result = CandidateSet(
        model_id=shape.model_id,
        shape=shape,
        backend=backend,
        profile=profile,
        budget_bytes=budget_bytes,
        concurrency=concurrency,
    )

    for method in chosen:
        for max_model_len in _context_lengths(shape, context_lengths):
            for kv_dtype in kv_dtypes:
                reasons: list[str] = []

                if method is not None:
                    availability = check_method(method, profile=profile, backend=backend)
                    reasons.extend(availability.reasons)

                # An FP8 KV cache is its own hardware requirement, independent
                # of how the weights are stored.
                if kv_dtype == "fp8" and profile is not None and "fp8" not in profile.capabilities:
                    reasons.append(f"hardware: {profile.name} has no fp8 for the KV cache")

                estimate = estimate_memory(
                    shape,
                    method,
                    max_model_len=max_model_len,
                    concurrency=concurrency,
                    kv_dtype=kv_dtype,
                    budget_bytes=budget_bytes,
                )
                if not estimate.fits:
                    reasons.append(
                        f"memory: needs {estimate.total_gib:.2f} GiB of "
                        f"{(budget_bytes or 0) / (1024**3):.2f} GiB"
                    )

                candidate = Candidate(
                    method=method.name if method else None,
                    max_model_len=max_model_len,
                    kv_dtype=kv_dtype,
                    estimate=estimate,
                )
                if reasons:
                    result.rejected.append(Rejection(candidate, tuple(reasons)))
                else:
                    result.accepted.append(candidate)

    result.accepted.sort(key=_sort_key)
    if len(result.accepted) > max_candidates:
        kept, dropped = _trim(result.accepted, max_candidates)
        result.accepted = kept
        result.rejected.extend(
            Rejection(c, (f"budget: over the {max_candidates}-candidate limit",)) for c in dropped
        )

    return result


def _trim(candidates: list[Candidate], limit: int) -> tuple[list[Candidate], list[Candidate]]:
    """Cut the list to ``limit`` while keeping every method represented.

    Truncating the sorted list would be simpler and wrong: the sort puts the
    least lossy methods first, so the cap would drop the entire 4-bit family --
    exactly the candidates a memory-constrained search is for. Taking one per
    method in rotation keeps the space broad, then deepens it.
    """
    by_method: dict[str | None, list[Candidate]] = {}
    for candidate in candidates:
        by_method.setdefault(candidate.method, []).append(candidate)

    kept: list[Candidate] = []
    while len(kept) < limit and any(by_method.values()):
        for queue in by_method.values():
            if not queue:
                continue
            kept.append(queue.pop(0))
            if len(kept) >= limit:
                break

    dropped = [c for queue in by_method.values() for c in queue]
    kept.sort(key=_sort_key)
    return kept, dropped


__all__ = [
    "DEFAULT_CONTEXT_LENGTHS",
    "DEFAULT_MAX_CANDIDATES",
    "KV_DTYPES",
    "Candidate",
    "CandidateSet",
    "Rejection",
    "generate_candidates",
]
