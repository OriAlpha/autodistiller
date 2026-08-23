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
    def mlp_params_per_layer(self) -> int:
        # Gated MLP: gate and up projections in, down projection out.
        return 3 * self.hidden_size * self.intermediate_size

    @property
    def layer_params(self) -> int:
        return self.attention_params_per_layer + self.mlp_params_per_layer

    @property
    def transformer_params(self) -> int:
        """Parameters in the blocks. These are what quantization acts on."""
        return self.layer_params * self.n_layers

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
        return (
            f"{self.n_parameters / 1e9:.2f}B params, {self.n_layers} layers, "
            f"{self.n_kv_heads} KV heads x {self.head_dim}"
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


def _first_int(config: Any, *names: str, default: int | None = None) -> int:
    for name in names:
        value = getattr(config, name, None)
        if isinstance(value, int) and value > 0:
            return value
    if default is None:
        raise ValueError(f"config has none of {names}")
    return default


def shape_from_config(model_id: str, config: Any) -> ModelShape:
    """Extract dimensions from a loaded Hugging Face config object."""
    hidden = _first_int(config, "hidden_size", "n_embd", "d_model")
    n_heads = _first_int(config, "num_attention_heads", "n_head", "num_heads")
    architectures = getattr(config, "architectures", None)

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
