"""Memory estimation.

The roadmap asks for memory to be screened *before* expensive runs, because the
cheapest way to reject a candidate is to notice it cannot fit. Every estimate
here is arithmetic over the model's config: no weights are loaded and no GPU is
touched.

Three terms:

* **Weights.** Format-dependent, and the two families differ in a way that
  matters. compressed-tensors quantizes the transformer blocks and leaves
  embeddings and the output head at 16-bit, which is why a 4-bit model is never
  a quarter of its 16-bit size. GGUF quantizes those tensors too, but keeps them
  above the headline type. Either way the gap is widest on small models with
  large vocabularies, so neither can be estimated from a nominal bit width
  alone.
* **KV cache.** Two tensors per layer per KV head per token. Under continuous
  batching this scales with concurrency, and on long contexts it overtakes the
  weights entirely.
* **Overhead.** Activations, CUDA graphs, allocator fragmentation. A fraction
  rather than a model, because it is not worth more precision than that.

Estimates are approximations and are labelled as such. Their job is to reject
what obviously cannot fit, not to predict the last megabyte.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..compression.methods import CompressionMethod
from .shape import ModelShape

BYTES_PER_GIB = 1024**3

QUANT_GROUP_SIZE = 128
"""Weights per shared scale for grouped integer quantization."""

GGUF_EMBEDDING_FLOOR_BPW = 6.56
"""Bits per weight llama.cpp keeps embedding and output tensors at, at minimum.

Q6_K. The K-quant mixes hold those tensors above the headline type because they
are the most quantization-sensitive in the file, so a Q3_K_M model is not 3.74
bits everywhere.
"""

RUNTIME_OVERHEAD_FRACTION = 0.10
"""Fraction of device memory for activations, CUDA graphs and fragmentation.

Measured against vLLM on an RTX 5070: 0.27 GiB activation + 0.32 GiB CUDA graph
against 7.96 GiB total, so roughly 7%. Rounded up, since under-estimating
overhead produces candidates that OOM at serve time.
"""

DTYPE_BYTES = {"float32": 4, "float16": 2, "bfloat16": 2, "fp8": 1, "auto": 2}


@dataclass(frozen=True)
class MemoryEstimate:
    """A memory budget broken into the terms that drive it."""

    weights_bytes: int
    kv_cache_bytes: int
    overhead_bytes: int
    budget_bytes: int | None = None

    @property
    def total_bytes(self) -> int:
        return self.weights_bytes + self.kv_cache_bytes + self.overhead_bytes

    @property
    def total_gib(self) -> float:
        return self.total_bytes / BYTES_PER_GIB

    @property
    def fits(self) -> bool:
        return self.budget_bytes is None or self.total_bytes <= self.budget_bytes

    @property
    def headroom_bytes(self) -> int | None:
        if self.budget_bytes is None:
            return None
        return self.budget_bytes - self.total_bytes

    @property
    def utilization(self) -> float | None:
        if not self.budget_bytes:
            return None
        return self.total_bytes / self.budget_bytes

    def describe(self) -> str:
        parts = [
            f"weights {self.weights_bytes / BYTES_PER_GIB:.2f}",
            f"KV {self.kv_cache_bytes / BYTES_PER_GIB:.2f}",
            f"overhead {self.overhead_bytes / BYTES_PER_GIB:.2f}",
        ]
        text = f"{self.total_gib:.2f} GiB ({', '.join(parts)})"
        if self.budget_bytes:
            text += f" of {self.budget_bytes / BYTES_PER_GIB:.2f} GiB"
        return text


def weight_bytes(shape: ModelShape, method: CompressionMethod | None) -> int:
    """Bytes the weights occupy under a compression method.

    ``None`` means the uncompressed baseline at 16-bit.

    Two formats, two shapes. compressed-tensors quantizes the transformer blocks
    and leaves embeddings and the output head at 16-bit, so the nominal width
    plus the group scales describes it. GGUF quantizes everything and mixes
    widths within a single K-quant, so its published bits-per-weight describes
    it and the nominal width does not.
    """
    if method is None:
        return shape.n_parameters * 2

    if method.bits_per_weight is not None:
        # A published whole-file average, which already accounts for block
        # scales and the mixed widths a K-quant uses internally.
        total = int(shape.transformer_params * method.bits_per_weight / 8)

        if not method.quantizes_embeddings:
            return total + shape.embedding_params * 2

        # Those averages are measured on 7B-class models, where embeddings are a
        # rounding error. On a small model with a large vocabulary they are not:
        # Qwen3-0.6B carries 26% of its parameters in a 151936-entry embedding,
        # and llama.cpp deliberately keeps those tensors above the headline type
        # because they are the most quantization-sensitive in the file. Applying
        # the headline average to them under-estimates the artifact by roughly
        # 15%, and a memory screen that under-estimates produces candidates that
        # OOM at serve time.
        #
        # ponytail: one floor rather than llama.cpp's per-tensor type table,
        # which is version-dependent and would need tracking upstream. Replace
        # with the real table if measured GGUF sizes drift from these estimates.
        embedding_bpw = max(method.bits_per_weight, GGUF_EMBEDDING_FLOOR_BPW)
        return total + int(shape.embedding_params * embedding_bpw / 8)

    bits = method.weight_bits
    quantized = shape.transformer_params * bits // 8

    # Grouped integer schemes store a scale (and for asymmetric, a zero point)
    # per group. Small next to the weights, but not nothing at 4 bits.
    if bits < 8:
        groups = shape.transformer_params // QUANT_GROUP_SIZE
        quantized += groups * 2

    # Embeddings and the output head are left alone: they are quantization
    # sensitive, and `ignore=["lm_head"]` is the default recipe.
    return quantized + shape.embedding_params * 2


def kv_cache_bytes(
    shape: ModelShape,
    *,
    max_model_len: int,
    concurrency: int = 1,
    kv_dtype: str = "auto",
) -> int:
    """KV cache for ``concurrency`` sequences each up to ``max_model_len``.

    This is the worst case: every sequence filling its context. Real workloads
    use less, but a candidate that cannot hold its advertised context is one
    that will fail under load rather than gracefully.
    """
    per_token = shape.kv_bytes_per_token
    if kv_dtype in ("fp8", "fp8_e4m3", "fp8_e5m2"):
        per_token //= 2  # the cache is 16-bit by default
    return per_token * max_model_len * concurrency


def estimate_memory(
    shape: ModelShape,
    method: CompressionMethod | None,
    *,
    max_model_len: int,
    concurrency: int = 1,
    kv_dtype: str = "auto",
    budget_bytes: int | None = None,
    overhead_fraction: float = RUNTIME_OVERHEAD_FRACTION,
) -> MemoryEstimate:
    """Estimate device memory for one configuration."""
    weights = weight_bytes(shape, method)
    kv = kv_cache_bytes(
        shape, max_model_len=max_model_len, concurrency=concurrency, kv_dtype=kv_dtype
    )

    # Overhead scales with the device when there is a budget, since CUDA graphs
    # and the allocator grow into what is available.
    basis = budget_bytes if budget_bytes else weights + kv
    overhead = int(basis * overhead_fraction)

    return MemoryEstimate(
        weights_bytes=weights,
        kv_cache_bytes=kv,
        overhead_bytes=overhead,
        budget_bytes=budget_bytes,
    )


def max_context_for_budget(
    shape: ModelShape,
    method: CompressionMethod | None,
    *,
    budget_bytes: int,
    concurrency: int = 1,
    kv_dtype: str = "auto",
    overhead_fraction: float = RUNTIME_OVERHEAD_FRACTION,
) -> int:
    """Longest context that fits, given weights and overhead.

    Useful in the other direction: not "does this fit" but "how much context can
    I afford". Returns 0 when the weights alone exceed the budget.
    """
    weights = weight_bytes(shape, method)
    overhead = int(budget_bytes * overhead_fraction)
    remaining = budget_bytes - weights - overhead
    if remaining <= 0:
        return 0

    per_token = shape.kv_bytes_per_token
    if kv_dtype in ("fp8", "fp8_e4m3", "fp8_e5m2"):
        per_token //= 2
    return int(remaining // (per_token * max(concurrency, 1)))


_SIZE_UNITS = {
    "b": 1,
    "kb": 1000,
    "mb": 1000**2,
    "gb": 1000**3,
    "tb": 1000**4,
    "kib": 1024,
    "mib": 1024**2,
    "gib": 1024**3,
    "tib": 1024**4,
}


def parse_size(text: str) -> int:
    """Parse '8GB', '8GiB', '8192MiB' into bytes.

    GB and GiB are both accepted and are not the same number. A GPU advertised
    as 8GB has 8 GiB of VRAM, so the difference is worth honouring rather than
    guessing.
    """
    cleaned = text.strip().replace(" ", "").lower()
    for unit in sorted(_SIZE_UNITS, key=len, reverse=True):
        if cleaned.endswith(unit):
            number = cleaned[: -len(unit)]
            try:
                return int(float(number) * _SIZE_UNITS[unit])
            except ValueError:
                break
    try:
        return int(float(cleaned))
    except ValueError:
        raise ValueError(
            f"could not parse size {text!r}; try forms like '8GB', '8GiB' or '8192MiB'"
        ) from None


__all__ = [
    "BYTES_PER_GIB",
    "RUNTIME_OVERHEAD_FRACTION",
    "MemoryEstimate",
    "estimate_memory",
    "kv_cache_bytes",
    "max_context_for_budget",
    "parse_size",
    "weight_bytes",
]
