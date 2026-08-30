"""Model dimensions, read without loading weights.

Screening a search space must not cost a model download per candidate. Every
number here comes from the Hugging Face ``config.json``, which is a few
kilobytes, so a whole candidate set can be evaluated for memory fit before any
GPU work happens at all.

Parameter counts are computed from the architecture rather than reported by the
checkpoint, which is what makes the same arithmetic work for a model that has
not been downloaded.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel


@dataclass(frozen=True)
class ModelShape:
    """The dimensions that determine memory use."""

    model_id: str
    n_layers: int
    hidden_size: int
    intermediate_size: int
    n_attention_heads: int
    n_kv_heads: int
    head_dim: int
    vocab_size: int
    max_position_embeddings: int
    tie_word_embeddings: bool = False
    architecture: str | None = None

    n_experts: int = 0
    """Routed experts per MoE layer. Zero for a dense model."""

    expert_intermediate_size: int = 0
    """One routed expert's width. Falls back to ``intermediate_size``."""

    shared_expert_intermediate_size: int = 0
    """Summed width of a layer's always-on experts, if it has any."""

    n_dense_layers: int = 0
    """Leading layers that are dense despite the model being MoE (DeepSeek)."""

    @property
    def embedding_params(self) -> int:
        """Input embeddings, plus the output head when it is not tied."""
        table = self.vocab_size * self.hidden_size
        return table if self.tie_word_embeddings else table * 2

    @property
    def attention_params_per_layer(self) -> int:
        q = self.hidden_size * self.n_attention_heads * self.head_dim
        kv = 2 * self.hidden_size * self.n_kv_heads * self.head_dim
        out = self.n_attention_heads * self.head_dim * self.hidden_size
        return q + kv + out

    @property
    def is_moe(self) -> bool:
        return self.n_experts > 0

    @property
    def dense_mlp_params_per_layer(self) -> int:
        # Gated MLP: gate and up projections in, down projection out.
        return 3 * self.hidden_size * self.intermediate_size

    @property
    def moe_mlp_params_per_layer(self) -> int:
        """One MoE layer: every expert, plus any shared expert and the router.

        Every expert, not the ones a token routes to. All of them sit in VRAM
        whether or not a given token uses them, so the count that decides fit is
        the total rather than the active subset the model's name advertises --
        Qwen3-30B-A3B is 3B active and 30B resident, and only the second number
        has to fit on the card.
        """
        width = self.expert_intermediate_size or self.intermediate_size
        experts = self.n_experts * 3 * self.hidden_size * width
        shared = 3 * self.hidden_size * self.shared_expert_intermediate_size
        router = self.hidden_size * self.n_experts
        return experts + shared + router

    @property
    def mlp_params_per_layer(self) -> int:
        """The feed-forward cost of a typical layer."""
        return self.moe_mlp_params_per_layer if self.is_moe else self.dense_mlp_params_per_layer

    @property
    def layer_params(self) -> int:
        return self.attention_params_per_layer + self.mlp_params_per_layer

    @property
    def transformer_params(self) -> int:
        """Parameters in the blocks. These are what quantization acts on.

        # ponytail: layers are either dense-prefix or MoE. Models that alternate
        # on a period (Qwen2-MoE's decoder_sparse_step, DeepSeek's moe_layer_freq)
        # are counted as fully sparse; add the period here if one is off enough
        # to matter.
        """
        dense = min(self.n_dense_layers, self.n_layers)
        return (
            self.attention_params_per_layer * self.n_layers
            + self.dense_mlp_params_per_layer * dense
            + self.mlp_params_per_layer * (self.n_layers - dense)
        )

    @property
    def n_parameters(self) -> int:
        return self.transformer_params + self.embedding_params

    @property
    def kv_bytes_per_token(self) -> int:
        """KV cache cost of one token at 16-bit, across all layers.

        Two tensors (keys and values), per layer, per KV head. Grouped-query
        attention makes ``n_kv_heads`` much smaller than the attention head
        count, which is why it dominates long-context serving cost.
        """
        return 2 * self.n_layers * self.n_kv_heads * self.head_dim * 2

    def describe(self) -> str:
        moe = f", {self.n_experts} experts" if self.is_moe else ""
        return (
            f"{self.n_parameters / 1e9:.2f}B params, {self.n_layers} layers, "
            f"{self.n_kv_heads} KV heads x {self.head_dim}{moe}"
        )


class ModelShapeRecord(BaseModel):
    """Serializable form, for run records."""

    model_id: str
    n_layers: int
    hidden_size: int
    n_kv_heads: int
    head_dim: int
    vocab_size: int
    n_parameters: int
    kv_bytes_per_token: int


def _find_int(config: Any, *names: str) -> int | None:
    """The first positive integer among these attribute names, or None."""
    for name in names:
        value = getattr(config, name, None)
        if isinstance(value, int) and value > 0:
            return value
    return None


def _first_int(config: Any, *names: str, default: int | None = None) -> int:
    if (found := _find_int(config, *names)) is not None:
        return found
    if default is None:
        raise ValueError(f"config has none of {names}")
    return default


def _shared_expert_width(config: Any) -> int:
    """Total intermediate width of a layer's always-on experts.

    Spelled two ways: Qwen2-MoE gives the width directly, DeepSeek gives a count
    of shared experts that each have the routed expert's width.
    """
    if (direct := _find_int(config, "shared_expert_intermediate_size")) is not None:
        return direct
    count = _find_int(config, "n_shared_experts")
    width = _find_int(config, "moe_intermediate_size")
    return count * width if count and width else 0


def text_config_of(config: Any) -> Any:
    """The language model's own config, for models that nest it.

    A vision-language model puts its two towers side by side: ``text_config``
    holds the decoder, ``vision_config`` the image encoder, and the top level
    holds neither's dimensions. Gemma 3 is shaped this way, as are Qwen-VL and
    Llama 3.2 Vision.

    The decoder is the part that gets quantized and the part whose KV cache
    dominates memory, so it is the right thing to measure. The vision tower is
    small by comparison and is not what a serving decision turns on -- but it is
    also not counted here, so a VLM's estimate is of its language half.
    """
    nested = getattr(config, "text_config", None)
    if nested is None:
        return config
    # Only trust it if it actually carries what we need; some configs define the
    # attribute and leave it empty.
    return nested if _find_int(nested, "hidden_size", "n_embd", "d_model") else config


_DECODER_SUFFIXES = ("ForCausalLM", "ForConditionalGeneration")
"""Architecture names whose shape this module's arithmetic describes.

A vision-language model is ``ForConditionalGeneration`` and is in scope: its
decoder is what gets quantized. An encoder-decoder is the same suffix and is
not, which is why ``is_encoder_decoder`` is checked separately.
"""


def _reject_if_not_decoder(architectures: Any) -> None:
    """Refuse to describe a model this arithmetic does not fit.

    Every formula below assumes a decoder-only transformer: a gated MLP of three
    matrices, and a KV cache of two tensors per layer per token. An encoder --
    BERT, ViT, a reranker -- has a two-matrix MLP and no KV cache at all, but
    its config carries the same field names, so the arithmetic runs and returns
    a confident wrong answer instead of failing. A wrong memory estimate is
    worse than none: it is what a candidate is screened against.

    A config with no ``architectures`` is left alone. That is a local or
    hand-built config where we cannot tell, and guessing wrong in the
    restrictive direction would block a model that works.
    """
    names = [str(name) for name in (architectures or ())]
    if not names or any(name.endswith(_DECODER_SUFFIXES) for name in names):
        return
    raise ValueError(
        f"{names[0]} is not a decoder-only language model. AutoDistiller's memory "
        f"arithmetic (gated MLP, KV cache) does not describe it, and the estimate "
        f"would be wrong rather than missing. Only causal LMs and vision-language "
        f"models are supported."
    )


def shape_from_config(model_id: str, config: Any) -> ModelShape:
    """Extract dimensions from a loaded Hugging Face config object."""
    architectures_source = config
    config = text_config_of(config)
    architectures = getattr(architectures_source, "architectures", None) or getattr(
        config, "architectures", None
    )
    if getattr(architectures_source, "is_encoder_decoder", False):
        raise ValueError(
            f"{model_id} is an encoder-decoder model. Its encoder has no KV cache and "
            f"is not counted here, so the estimate would be wrong rather than missing."
        )
    _reject_if_not_decoder(architectures)

    hidden = _first_int(config, "hidden_size", "n_embd", "d_model")
    n_heads = _first_int(config, "num_attention_heads", "n_head", "num_heads")

    return ModelShape(
        model_id=model_id,
        n_layers=_first_int(config, "num_hidden_layers", "n_layer", "num_layers"),
        hidden_size=hidden,
        intermediate_size=_first_int(config, "intermediate_size", "ffn_dim", default=4 * hidden),
        n_attention_heads=n_heads,
        # Grouped-query models report fewer KV heads; older ones report none and
        # use one per attention head.
        n_kv_heads=_first_int(config, "num_key_value_heads", "num_kv_heads", default=n_heads),
        head_dim=_first_int(config, "head_dim", default=hidden // n_heads),
        vocab_size=_first_int(config, "vocab_size"),
        # Zero means dense. Each family spells the expert count its own way.
        n_experts=_first_int(
            config, "num_experts", "num_local_experts", "n_routed_experts", default=0
        ),
        expert_intermediate_size=_first_int(config, "moe_intermediate_size", default=0),
        shared_expert_intermediate_size=_shared_expert_width(config),
        n_dense_layers=_first_int(config, "first_k_dense_replace", default=0),
        max_position_embeddings=_first_int(
            config, "max_position_embeddings", "n_positions", default=4096
        ),
        tie_word_embeddings=bool(getattr(config, "tie_word_embeddings", False)),
        architecture=architectures[0] if architectures else None,
    )


def load_shape(model_id: str, *, revision: str | None = None, trust_remote_code: bool = False):
    """Fetch just the config for a model and derive its shape."""
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(
        model_id, revision=revision, trust_remote_code=trust_remote_code
    )
    return shape_from_config(model_id, config)


def to_record(shape: ModelShape) -> ModelShapeRecord:
    return ModelShapeRecord(
        model_id=shape.model_id,
        n_layers=shape.n_layers,
        hidden_size=shape.hidden_size,
        n_kv_heads=shape.n_kv_heads,
        head_dim=shape.head_dim,
        vocab_size=shape.vocab_size,
        n_parameters=shape.n_parameters,
        kv_bytes_per_token=shape.kv_bytes_per_token,
    )


__all__ = [
    "ModelShape",
    "ModelShapeRecord",
    "load_shape",
    "shape_from_config",
    "to_record",
]
