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
  weights entirely. An encoder has none, and pays for activations instead --
  which is a different shape, quadratic in sequence length rather than linear.
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

GGUF_EMBEDDING_FLOOR_BPW = 6.5625
"""Bits per weight llama.cpp keeps the output head at, at minimum.

Q6_K, confirmed by reading a produced artifact: `output.weight` came back Q6_K
in a Q4_K_M file. The K-quant mixes hold that tensor above the headline type
because it is the most quantization-sensitive in the file.
"""

RUNTIME_OVERHEAD_FRACTION = 0.10
"""Fraction of device memory for activations, CUDA graphs and fragmentation.

Measured against vLLM on an RTX 5070: 0.27 GiB activation + 0.32 GiB CUDA graph
against 7.96 GiB total, so roughly 7%. Rounded up, since under-estimating
overhead produces candidates that OOM at serve time.
"""

DTYPE_BYTES = {"float32": 4, "float16": 2, "bfloat16": 2, "fp8": 1, "auto": 2}

ENCODER_OVERHEAD_FLOOR_BYTES = int(0.15 * BYTES_PER_GIB)
"""Fixed device cost of serving an encoder, whatever the card.

Measured against vLLM 0.27's pooling server on bge-small-en-v1.5: weights took
0.04 GiB, CUDA graph capture took 0.10 GiB, and there was no KV cache line at
all. Nothing here scales with the device, so the decoder's "fraction of the
budget" does not describe it -- on an 8 GiB card that rule claimed 0.75 GiB for
a model whose weights are 0.04, which is twelve times the model and enough to
make every encoder configuration report the same total.

0.15 rather than the measured 0.10, because under-estimating overhead produces
candidates that OOM at serve time and over-estimating only costs a candidate
that would have fit.

Confirmed not to scale with the model: bert-base-uncased is three times
bge-small and its graph capture cost the same 0.10 GiB. A fraction of either the
model or the device would have been wrong in both directions.
"""

ACTIVATION_RESIDENT_TOKEN_BUDGET = 32768
"""Tokens a serving runtime holds resident at once, whatever is in flight.

A scheduler does not encode every queued text simultaneously -- it admits work
up to a token budget and runs the rest after. Without this cap the estimate is a
function of the client's concurrency, which the server does not honour:
256 texts of 512 tokens came out at 2.44 GiB against a server that peaked at
0.77 GiB including weights and CUDA graphs, so it rejected configurations that
run comfortably.

Checked at 3x the size: bert-base-uncased (109M) estimates 1.20 GiB at the
budget against a server whose whole-device peak was 1.00 GiB, of which roughly
1.00 is the runtime context the launcher accounts for separately. Conservative
at both sizes, which is the direction a screen should err in.

# ponytail: one budget for every runtime, calibrated against vLLM pooling rather
# than read out of a scheduler config, which is version-dependent. Revisit if a
# runtime lands far from it.
"""

ACTIVATION_BLOCK_COPIES = 2
"""Live copies of a block's widened activations during one forward pass.

Roughly what goes in and what comes out. Inference keeps no autograd graph, so
only one block's tensors are live at a time -- this does not multiply by layer
count, which is the difference between an estimate near the truth and one that
is a factor of twelve out.
"""


@dataclass(frozen=True)
class MemoryEstimate:
    """A memory budget broken into the terms that drive it."""

    weights_bytes: int
    kv_cache_bytes: int
    overhead_bytes: int
    budget_bytes: int | None = None
    draft_bytes: int = 0
    """Part of ``weights_bytes``, held separately so the report can name it."""

    dynamic_label: str = "KV"
    """What ``kv_cache_bytes`` actually holds for this model.

    The field is one slot -- memory that grows with the workload rather than
    with the weights -- but it is a KV cache for a decoder and activations for
    an encoder. Reporting an encoder's activations under "KV cache" would name a
    thing the model does not have.
    """

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
            f"weights {(self.weights_bytes - self.draft_bytes) / BYTES_PER_GIB:.2f}",
            f"{self.dynamic_label} {self.kv_cache_bytes / BYTES_PER_GIB:.2f}",
            f"overhead {self.overhead_bytes / BYTES_PER_GIB:.2f}",
        ]
        if self.draft_bytes:
            parts.insert(1, f"draft {self.draft_bytes / BYTES_PER_GIB:.2f}")
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

        # GGUF writes the embedding twice, even when the model ties them:
        # `token_embd.weight` at the headline type, and `output.weight` at a
        # higher one because the output head is the most quantization-sensitive
        # tensor in the file. Counting it once under-reported a measured
        # Qwen3-0.6B Q4_K_M artifact by 18.5% -- and a screen that
        # under-estimates produces candidates that OOM at serve time, which is
        # the one thing it exists to prevent.
        #
        # A tied model reports one embedding, an untied one reports both
        # tensors already, so halving recovers the single size in either case.
        #
        # ponytail: two rates rather than llama.cpp's per-tensor type table,
        # which is version-dependent and would need tracking upstream.
        # Calibrated against one measured artifact; revisit if others drift.
        single = (
            shape.embedding_params if shape.tie_word_embeddings else shape.embedding_params // 2
        )
        token_embd = single * method.bits_per_weight / 8
        output_head = single * max(method.bits_per_weight, GGUF_EMBEDDING_FLOOR_BPW) / 8
        return total + int(token_embd + output_head)

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


def activation_bytes(shape: ModelShape, *, seq_len: int, batch: int) -> int:
    """Peak activation memory for one encoder forward pass, at 16-bit.

    Two terms carry it. The linear one is the residual stream and the widened
    feed-forward, and it grows with tokens. The attention score matrix is
    ``batch x heads x seq x seq``, and it grows with the *square* of sequence
    length -- which is why a batch that fits at 128 tokens can fail at 512, and
    why an encoder's memory question is a different question rather than the
    same one with the cache removed.

    Bounded by what a scheduler will hold at once rather than by what the client
    asks for: see ``ACTIVATION_RESIDENT_TOKEN_BUDGET``.

    # ponytail: two terms, not a per-op accounting. It is a screen, and the
    # quadratic one dominates well before anything else would matter.
    """
    tokens = min(batch * seq_len, ACTIVATION_RESIDENT_TOKEN_BUDGET)
    # Whole sequences, because the score matrix is per sequence and at least one
    # is always resident however long it is.
    resident = max(tokens // seq_len, 1)

    linear = tokens * (shape.hidden_size + shape.intermediate_size) * 2 * ACTIVATION_BLOCK_COPIES
    scores = resident * shape.n_attention_heads * seq_len * seq_len * 2
    return linear + scores


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
    draft_bytes: int = 0,
) -> MemoryEstimate:
    """Estimate device memory for one configuration.

    ``draft_bytes`` is a speculative decoding draft model, which sits on the
    device beside the target for the whole run. It is counted with the weights
    because that is what it is -- a second set of them.
    """
    weights = weight_bytes(shape, method) + draft_bytes
    # The same two numbers mean different things for an encoder: max_model_len
    # is the sequence it encodes rather than a context it grows into, and
    # concurrency is the batch encoded at once rather than sequences held open.
    # Both still decide whether it fits, so the caller passes the same pair and
    # the arithmetic is picked here.
    kv = (
        activation_bytes(shape, seq_len=max_model_len, batch=concurrency)
        if shape.is_encoder
        else kv_cache_bytes(
            shape, max_model_len=max_model_len, concurrency=concurrency, kv_dtype=kv_dtype
        )
    )

    if shape.is_encoder:
        # An encoder's runtime cost does not grow into the card: there is no
        # cache to size and no context to reserve for, so the floor is what a
        # pooling server actually holds.
        overhead = max(int((weights + kv) * overhead_fraction), ENCODER_OVERHEAD_FLOOR_BYTES)
    else:
        # Overhead scales with the device when there is a budget, since CUDA
        # graphs and the allocator grow into what is available.
        basis = budget_bytes if budget_bytes else weights + kv
        overhead = int(basis * overhead_fraction)

    return MemoryEstimate(
        dynamic_label="activations" if shape.is_encoder else "KV",
        weights_bytes=weights,
        kv_cache_bytes=kv,
        overhead_bytes=overhead,
        budget_bytes=budget_bytes,
        draft_bytes=draft_bytes,
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
