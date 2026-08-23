"""NVIDIA hardware profiles and numeric-format capability rules.

Phase 4 filters candidate configurations by what the target hardware can
actually run, so the question "does this GPU support FP8?" needs a single
answer everywhere. That answer comes from a rule keyed on compute capability
rather than a list of card names, because the rule stays correct for cards
nobody has added to the list yet.

The named profiles exist for the other direction: planning a deployment for
hardware you do not currently have in the machine.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .hardware import BYTES_PER_GIB, GPUInfo


@dataclass(frozen=True)
class GPUProfile:
    """A target GPU, whether or not it is the one in this machine."""

    name: str
    vram_gib: float
    compute_capability: str
    aliases: tuple[str, ...] = field(default=())

    @property
    def vram_bytes(self) -> int:
        return int(self.vram_gib * BYTES_PER_GIB)

    @property
    def capabilities(self) -> frozenset[str]:
        return capabilities_for(self.compute_capability)

    @property
    def architecture(self) -> str:
        return architecture_for(self.compute_capability)


def _sm(compute_capability: str) -> int:
    """'12.0' -> 120. Ordering these as integers is what the rules compare."""
    try:
        major, _, minor = compute_capability.partition(".")
        return int(major) * 10 + int(minor or 0)
    except ValueError:
        return 0


def architecture_for(compute_capability: str) -> str:
    sm = _sm(compute_capability)
    for threshold, name in (
        (120, "Blackwell (consumer)"),
        (100, "Blackwell (datacenter)"),
        (90, "Hopper"),
        (89, "Ada Lovelace"),
        (80, "Ampere"),
        (75, "Turing"),
        (70, "Volta"),
    ):
        if sm >= threshold:
            return name
    return "pre-Volta"


def capabilities_for(compute_capability: str) -> frozenset[str]:
    """Numeric formats a compute capability can run with tensor-core support.

    Deliberately about *hardware*, not about any particular kernel library. A
    backend may still lack an implementation; that is a separate check, and
    Phase 3 owns it.
    """
    sm = _sm(compute_capability)
    caps: set[str] = {"fp32", "fp16"}

    if sm >= 75:  # Turing: int8/int4 tensor cores, enough for AWQ/GPTQ kernels
        caps |= {"int8", "int4"}
    if sm >= 80:  # Ampere: bf16
        caps.add("bf16")
    if sm >= 89:  # Ada: native fp8 (e4m3/e5m2)
        caps.add("fp8")
    if sm >= 100:  # Blackwell: fp4
        caps.add("fp4")

    return frozenset(caps)


# A small set of the cards people actually target. Not exhaustive by design:
# capabilities_for() covers anything missing, and an unmatched GPU still
# profiles correctly from its detected compute capability.
PROFILES: dict[str, GPUProfile] = {
    profile.name: profile
    for profile in (
        GPUProfile("rtx-3090", 24, "8.6", ("NVIDIA GeForce RTX 3090",)),
        GPUProfile("rtx-4090", 24, "8.9", ("NVIDIA GeForce RTX 4090",)),
        GPUProfile("rtx-5070", 8, "12.0", ("NVIDIA GeForce RTX 5070 Laptop GPU",)),
        GPUProfile("rtx-5090", 32, "12.0", ("NVIDIA GeForce RTX 5090",)),
        GPUProfile("l4", 24, "8.9", ("NVIDIA L4",)),
        GPUProfile("l40s", 48, "8.9", ("NVIDIA L40S",)),
        GPUProfile("a10g", 24, "8.6", ("NVIDIA A10G",)),
        GPUProfile("a100-40gb", 40, "8.0", ("NVIDIA A100-SXM4-40GB",)),
        GPUProfile("a100-80gb", 80, "8.0", ("NVIDIA A100-SXM4-80GB",)),
        GPUProfile("h100", 80, "9.0", ("NVIDIA H100 80GB HBM3", "NVIDIA H100 PCIe")),
    )
}


def match_profile(gpu: GPUInfo) -> GPUProfile | None:
    """Find the named profile for a detected GPU, if there is one."""
    name = gpu.name.strip()
    for profile in PROFILES.values():
        if name in profile.aliases or name.lower() == profile.name:
            return profile
    return None


def profile_from_gpu(gpu: GPUInfo) -> GPUProfile:
    """Always return a profile: the known one, or one built from detection.

    An unrecognized card is not an error. Its compute capability and VRAM are
    measured facts, which is everything the capability rules need.
    """
    if (known := match_profile(gpu)) is not None:
        return known
    return GPUProfile(
        name=gpu.name,
        vram_gib=gpu.total_memory_bytes / BYTES_PER_GIB,
        compute_capability=gpu.compute_capability or "0.0",
    )


def resolve_profile(name: str) -> GPUProfile:
    try:
        return PROFILES[name.lower()]
    except KeyError:
        raise KeyError(
            f"unknown GPU profile {name!r}; available: {', '.join(sorted(PROFILES))}"
        ) from None


__all__ = [
    "PROFILES",
    "GPUProfile",
    "architecture_for",
    "capabilities_for",
    "match_profile",
    "profile_from_gpu",
    "resolve_profile",
]
