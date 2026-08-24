"""The compression vocabulary AutoDistiller exposes.

This is the "stable AutoDistiller interface" the roadmap asks for. A user picks
``int4-awq``; nothing above this module needs to know that the backend calls it
an ``AWQModifier`` targeting a ``W4A16`` scheme. When a second backend arrives,
it maps onto these same names rather than leaking its own.

Whether a method is *allowed* is two separate questions, kept separate on
purpose:

* **Hardware** -- does the GPU have tensor cores for this numeric format? That
  is :func:`autodistiller.metadata.profiles.capabilities_for`, a property of
  silicon.
* **Backend** -- does the serving runtime have a kernel for it? A method can be
  physically runnable and still unsupported by vLLM.

Conflating the two is how you end up recommending a configuration that
benchmarks beautifully and then cannot be served.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..metadata.profiles import GPUProfile, capabilities_for


@dataclass(frozen=True)
class CompressionMethod:
    """One compression configuration AutoDistiller can ask a backend to produce."""

    name: str
    description: str
    weight_bits: int
    activation_bits: int
    scheme: str = ""
    """The backend's scheme string, e.g. W4A16. Backend detail, kept here so
    callers never have to build one."""

    algorithm: str = "rtn"
    """rtn | gptq | awq. Round-to-nearest needs no calibration data; the others
    fit weights against sample activations and do."""

    # Scheme names are not free-form: they must be preset names the compression
    # backend recognizes. See test_schemes_are_real_preset_names.

    required_capability: str = "int8"
    """Numeric format the hardware must support. See ``capabilities_for``."""

    needs_calibration: bool = False
    backends: tuple[str, ...] = ("vllm",)
    notes: str = ""

    bits_per_weight: float | None = None
    """Measured average bits per weight across the whole file, when the nominal
    width does not describe the format.

    llama.cpp's K-quants are mixed: a "4-bit" Q4_K_M holds attention and feed
    forward tensors at several widths and lands near 4.85 bits once the block
    scales are counted. Deriving a size from ``weight_bits`` alone would
    under-estimate every GGUF candidate by around 20%, which is the difference
    between fitting in VRAM and not.
    """

    quantizes_embeddings: bool = False
    """Whether the format compresses the embedding and output tensors too.

    compressed-tensors leaves them at 16-bit because they are quantization
    sensitive, which is why a 4-bit model is never a quarter of its 16-bit size.
    GGUF quantizes them as well, so the same nominal width produces a
    noticeably smaller file -- most visibly on small models with large
    vocabularies.
    """

    compression_backend: str = "llmcompressor"
    """The tool that produces this format.

    Picking a method is picking a toolchain, so the user should not have to name
    both. It also decides the shape of what comes out, which is why ``produces``
    is derived from it rather than repeated on every method.
    """

    @property
    def produces(self) -> str:
        """Shape of the artifact: a Hugging Face ``directory``, or a GGUF ``file``."""
        return "file" if self.compression_backend == "llama.cpp" else "directory"

    @property
    def is_gguf(self) -> bool:
        return self.compression_backend == "llama.cpp"

    @property
    def compresses_activations(self) -> bool:
        return self.activation_bits < 16

    def runs_on(self, profile: GPUProfile) -> bool:
        return self.required_capability in profile.capabilities

    def servable_by(self, backend: str) -> bool:
        return backend in self.backends

    def describe_size(self) -> str:
        return f"W{self.weight_bits}A{self.activation_bits}"


# Everything here is produced by an existing implementation. AutoDistiller does
# not own a kernel, and the roadmap is explicit that it should not start.
#
# Two families, because two runtimes want different things. The compressed-tensors
# methods come from llmcompressor and are served by vLLM; the GGUF methods come
# from llama.cpp's own quantizer and are served by llama-server. A method knows
# which backends can serve it, so the search space filters itself.
METHODS: dict[str, CompressionMethod] = {
    method.name: method
    for method in (
        CompressionMethod(
            name="int8",
            description="INT8 weights and activations, round-to-nearest",
            weight_bits=8,
            activation_bits=8,
            scheme="W8A8",
            algorithm="rtn",
            required_capability="int8",
            needs_calibration=True,
            notes="Activation scales need calibration data even for RTN weights.",
        ),
        CompressionMethod(
            name="int8-weight-only",
            description="INT8 weights, 16-bit activations",
            weight_bits=8,
            activation_bits=16,
            scheme="W8A16",
            algorithm="rtn",
            required_capability="int8",
            needs_calibration=False,
            notes="No calibration required. The safest first candidate.",
        ),
        CompressionMethod(
            name="int4-gptq",
            description="INT4 weights via GPTQ, 16-bit activations",
            weight_bits=4,
            activation_bits=16,
            scheme="W4A16",
            algorithm="gptq",
            required_capability="int4",
            needs_calibration=True,
        ),
        CompressionMethod(
            name="int4-awq",
            description="INT4 weights via AWQ, 16-bit activations",
            weight_bits=4,
            activation_bits=16,
            scheme="W4A16",
            algorithm="awq",
            required_capability="int4",
            needs_calibration=True,
        ),
        CompressionMethod(
            name="fp8",
            description="FP8 weights and activations (E4M3)",
            weight_bits=8,
            activation_bits=8,
            scheme="FP8_DYNAMIC",
            algorithm="rtn",
            required_capability="fp8",
            needs_calibration=False,
            notes="Dynamic activation scales, so no calibration pass. Ada (sm_89) or newer.",
        ),
        CompressionMethod(
            name="fp8-static",
            description="FP8 weights and activations with static per-tensor scales",
            weight_bits=8,
            activation_bits=8,
            scheme="FP8",
            algorithm="rtn",
            required_capability="fp8",
            needs_calibration=True,
            notes="Static scales are calibrated once, so this needs data but "
            "avoids per-token scale computation at inference time.",
        ),
        CompressionMethod(
            name="gguf-q8-0",
            description="GGUF Q8_0: 8-bit, the near-lossless llama.cpp baseline",
            weight_bits=8,
            activation_bits=16,
            scheme="Q8_0",
            algorithm="rtn",
            required_capability="fp16",
            needs_calibration=False,
            backends=("llama.cpp",),
            compression_backend="llama.cpp",
            bits_per_weight=8.5,
            quantizes_embeddings=True,
            notes="Not a K-quant: a flat 8-bit block format. The safest GGUF.",
        ),
        CompressionMethod(
            name="gguf-q6-k",
            description="GGUF Q6_K: 6-bit K-quant",
            weight_bits=6,
            activation_bits=16,
            scheme="Q6_K",
            algorithm="rtn",
            required_capability="fp16",
            needs_calibration=False,
            backends=("llama.cpp",),
            compression_backend="llama.cpp",
            bits_per_weight=6.56,
            quantizes_embeddings=True,
        ),
        CompressionMethod(
            name="gguf-q5-k-m",
            description="GGUF Q5_K_M: 5-bit K-quant, medium mix",
            weight_bits=5,
            activation_bits=16,
            scheme="Q5_K_M",
            algorithm="rtn",
            required_capability="fp16",
            needs_calibration=False,
            backends=("llama.cpp",),
            compression_backend="llama.cpp",
            bits_per_weight=5.69,
            quantizes_embeddings=True,
        ),
        CompressionMethod(
            name="gguf-q4-k-m",
            description="GGUF Q4_K_M: 4-bit K-quant, medium mix",
            weight_bits=4,
            activation_bits=16,
            scheme="Q4_K_M",
            algorithm="rtn",
            required_capability="fp16",
            needs_calibration=False,
            backends=("llama.cpp",),
            compression_backend="llama.cpp",
            bits_per_weight=4.85,
            quantizes_embeddings=True,
            notes="The usual llama.cpp default: the best quality-per-byte of the "
            "K-quants for most models.",
        ),
        CompressionMethod(
            name="gguf-q3-k-m",
            description="GGUF Q3_K_M: 3-bit K-quant, for tight memory",
            weight_bits=3,
            activation_bits=16,
            scheme="Q3_K_M",
            algorithm="rtn",
            required_capability="fp16",
            needs_calibration=False,
            backends=("llama.cpp",),
            compression_backend="llama.cpp",
            bits_per_weight=3.74,
            quantizes_embeddings=True,
            notes="Quality falls off here, especially below about 7B parameters.",
        ),
    )
}

BASELINE = "none"
"""The uncompressed reference. Not in METHODS: it is what candidates are
measured against, not a candidate."""


@dataclass
class MethodAvailability:
    """Why a method is or is not usable here."""

    method: CompressionMethod
    available: bool
    reasons: list[str] = field(default_factory=list)


def resolve_method(name: str) -> CompressionMethod:
    try:
        return METHODS[name.lower()]
    except KeyError:
        raise KeyError(
            f"unknown compression method {name!r}; available: {', '.join(sorted(METHODS))}"
        ) from None


def check_method(
    method: CompressionMethod,
    *,
    profile: GPUProfile | None = None,
    backend: str | None = None,
) -> MethodAvailability:
    """Decide whether a method can be produced and then served here."""
    reasons: list[str] = []

    if profile is not None and not method.runs_on(profile):
        reasons.append(
            f"{profile.name} ({profile.architecture}, sm_"
            f"{profile.compute_capability.replace('.', '')}) has no "
            f"{method.required_capability} support"
        )
    if backend is not None and not method.servable_by(backend):
        reasons.append(f"backend {backend!r} cannot serve {method.scheme}")

    return MethodAvailability(method=method, available=not reasons, reasons=reasons)


def available_methods(
    *, profile: GPUProfile | None = None, backend: str | None = None
) -> list[MethodAvailability]:
    """Every method, annotated with whether it is usable in this context.

    Returns the unavailable ones too. Phase 4 needs to explain why a candidate
    was filtered out, and "it silently vanished" is not an explanation.
    """
    return [check_method(method, profile=profile, backend=backend) for method in METHODS.values()]


def supported_capabilities(compute_capability: str) -> frozenset[str]:
    """Convenience re-export so callers need only import this module."""
    return capabilities_for(compute_capability)


__all__ = [
    "BASELINE",
    "METHODS",
    "CompressionMethod",
    "MethodAvailability",
    "available_methods",
    "check_method",
    "resolve_method",
    "supported_capabilities",
]
