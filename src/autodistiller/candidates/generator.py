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
from .speculative import SpeculativeSpec

DEFAULT_CONTEXT_LENGTHS = (2048, 4096, 8192)

ENCODER_SEQUENCE_LENGTHS = (128, 256, 512)
"""What an encoder is searched over instead.

The axis is the same field and a different question. A decoder's context is
how much history it can hold; an encoder's is how long a document it embeds,
and the answer is short -- a passage, not a conversation. Searching 2048 upward
would also miss the range where the quadratic attention term is decided.
"""
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
    speculative: SpeculativeSpec | None = None
    """A draft model to decode with, or None to decode normally.

    Orthogonal to ``method``: speculation does not change the target's weights,
    so it composes with every compression method rather than replacing one."""

    @property
    def is_baseline(self) -> bool:
        return self.method is None

    @property
    def id(self) -> str:
        method = self.method or "baseline"
        kv = "" if self.kv_dtype == "auto" else f"-kv{self.kv_dtype}"
        spec = f"-{self.speculative.label}" if self.speculative else ""
        return f"{method}-ctx{self.max_model_len}{kv}{spec}"

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
        return (0, 0, 0, candidate.max_model_len, "")
    method = METHODS[candidate.method]
    return (
        1,
        -method.weight_bits,
        -method.activation_bits,
        candidate.max_model_len,
        candidate.speculative.label if candidate.speculative else "",
    )


def _context_lengths(shape: ModelShape, requested: tuple[int, ...] | None) -> list[int]:
    default = ENCODER_SEQUENCE_LENGTHS if shape.is_encoder else DEFAULT_CONTEXT_LENGTHS
    lengths = requested or default
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
    speculative: SpeculativeSpec | None = None,
    supports_speculative: bool = True,
    concurrency: int = 1,
    include_baseline: bool = True,
    max_candidates: int = DEFAULT_MAX_CANDIDATES,
) -> CandidateSet:
    """Enumerate and filter the search space for one model."""
    if budget_bytes is None and profile is not None:
        budget_bytes = profile.vram_bytes

    # An FP8 KV cache is not a dimension when there is no cache to hold. Left as
    # one, every encoder candidate would be enumerated twice, identically.
    if shape.is_encoder:
        kv_dtypes = ("auto",)

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

    # None first: speculation is a second point on the same compression recipe,
    # and the sort below keeps the pair adjacent so the comparison is readable.
    speculations: list[SpeculativeSpec | None] = [None]
    if speculative is not None:
        speculations.append(speculative)

    for method in chosen:
        for max_model_len in _context_lengths(shape, context_lengths):
            for kv_dtype in kv_dtypes:
                for spec in speculations:
                    reasons: list[str] = []

                    if method is not None:
                        availability = check_method(
                            method, profile=profile, backend=backend, model_kind=shape.kind
                        )
                        reasons.extend(availability.reasons)

                    # An FP8 KV cache is its own hardware requirement, independent
                    # of how the weights are stored.
                    if (
                        kv_dtype == "fp8"
                        and profile is not None
                        and "fp8" not in profile.capabilities
                    ):
                        reasons.append(f"hardware: {profile.name} has no fp8 for the KV cache")

                    if spec is not None and not supports_speculative:
                        reasons.append(f"backend: {backend} cannot run {spec.method} drafts")

                    estimate = estimate_memory(
                        shape,
                        method,
                        max_model_len=max_model_len,
                        concurrency=concurrency,
                        kv_dtype=kv_dtype,
                        budget_bytes=budget_bytes,
                        draft_bytes=spec.weights_bytes if spec else 0,
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
                        speculative=spec,
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
    """Cut the list to ``limit`` while keeping every family represented.

    Truncating the sorted list would be simpler and wrong: the sort puts the
    least lossy methods first, so the cap would drop the entire 4-bit family --
    exactly the candidates a memory-constrained search is for. Taking one per
    family in rotation keeps the space broad, then deepens it.

    A family is a method *and* whether it speculates. Rotating on the method
    alone has the same failure in the other direction: a speculative candidate
    shares its method with its plain twin and sorts after it, so one queue per
    method fills every slot with plain candidates and drops the entire
    speculative half before anything is measured -- silently, and precisely when
    the user asked for the comparison by naming a draft.
    """
    by_family: dict[tuple[str | None, str | None], list[Candidate]] = {}
    for candidate in candidates:
        family = (
            candidate.method,
            candidate.speculative.label if candidate.speculative else None,
        )
        by_family.setdefault(family, []).append(candidate)

    kept: list[Candidate] = []
    while len(kept) < limit and any(by_family.values()):
        for queue in by_family.values():
            if not queue:
                continue
            kept.append(queue.pop(0))
            if len(kept) >= limit:
                break

    dropped = [c for queue in by_family.values() for c in queue]
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
