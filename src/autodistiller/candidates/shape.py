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

from ..architecture import DECODER, ENCODER, VISION, kind_of_config, model_kind


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

    kind: str = DECODER
    """``decoder``, ``encoder`` or ``vision``. Three formulas below turn on it.

    A decoder block has a gated MLP of three matrices and a KV cache; an encoder
    block has a two-matrix MLP and no cache at all. Both configs spell their
    dimensions with the same field names, which is why this has to be carried
    rather than inferred from the numbers.

    A vision tower is an encoder that reads pixels: same block, but its
    sequence length is decided by the image and the patch grid rather than by a
    tokenizer, and what it embeds is a patch projection rather than a
    vocabulary.
    """

    patch_size: int = 0
    """Side of one square image patch. Vision only; zero elsewhere."""

    image_size: int = 0
    """Side of the input image the checkpoint was trained at. Vision only.

    Not a free parameter the way a decoder's context is: the position table has
    one row per patch, so a different resolution is a different checkpoint.
    """

    n_channels: int = 3
    """Image channels the patch projection reads. Three, except when it isn't."""

    n_labels: int = 0
    """Classes the head predicts. Vision only; part of the weights, so counted."""

    inference_image_size: int = 0
    """Side of the image the model's own processor produces. Vision only.

    Not always ``image_size``, and the difference is not cosmetic. DINOv2 was
    trained at 518 pixels and its position table has a row per patch at that
    resolution, while its processor centre-crops to 224 -- so 1370 rows are
    stored and 257 tokens actually run. One number belongs to the weights and
    the other to the forward pass, and using either for both is wrong in a
    direction that matters: five times the activations, and a recorded
    evaluation context that describes something that did not happen.

    Zero means the processor was not consulted, and ``image_size`` stands in.
    """

    gated_mlp: bool = False
    """Whether a block's feed-forward has three matrices rather than two.

    Decoders have had one since Llama; the original BERT block does not, which
    is what the encoder arithmetic assumes. Several modern embedding models have
    since adopted one, and counting two matrices where there are three
    under-estimates them by a fifth. See ``ENCODER_BLOCK_OVERRIDES``.
    """

    has_position_table: bool = True
    """Whether a learned absolute position embedding is stored in the weights.

    False for rotary, ALiBi, or relative position bias, which compute position
    rather than looking it up. Only an encoder or a vision tower counts one at
    all; a modern decoder has never stored one.
    """

    n_experts: int = 0
    """Routed experts per MoE layer. Zero for a dense model."""

    expert_intermediate_size: int = 0
    """One routed expert's width. Falls back to ``intermediate_size``."""

    shared_expert_intermediate_size: int = 0
    """Summed width of a layer's always-on experts, if it has any."""

    n_dense_layers: int = 0
    """Leading layers that are dense despite the model being MoE (DeepSeek)."""

    @property
    def is_encoder(self) -> bool:
        return self.kind == ENCODER

    @property
    def is_vision(self) -> bool:
        return self.kind == VISION

    @property
    def has_kv_cache(self) -> bool:
        """Whether anything is kept between forward passes.

        Only a decoder does. Both encoder kinds see their whole input at once
        and keep nothing, so they pay for activations instead -- a different
        shape, quadratic in sequence length rather than linear in it.
        """
        return self.kind == DECODER

    @property
    def n_image_tokens(self) -> int:
        """Patches plus the class token: the sequence a vision tower encodes.

        Fixed by the checkpoint's own processor rather than chosen. This is why
        sequence length is a search axis for a text encoder and not for this
        one -- there is exactly one value, and asking for another describes a
        model that does not exist.
        """
        return self._tokens_at(self.inference_image_size or self.image_size)

    @property
    def position_tokens(self) -> int:
        """Rows in the stored position table, which is a fact about the weights.

        The resolution the checkpoint was *trained* at, where ``n_image_tokens``
        is the resolution it is *run* at. Equal for ViT, DeiT and BEiT, and not
        for DINOv2.
        """
        return self._tokens_at(self.image_size)

    def _tokens_at(self, side: int) -> int:
        if not self.is_vision or not self.patch_size or not side:
            return 0
        return (side // self.patch_size) ** 2 + 1

    @property
    def embedding_params(self) -> int:
        """Input embeddings, plus the output head when there is one.

        An encoder has no output head to tie or untie -- what it has instead is
        a learned position table, where a modern decoder uses rotary embeddings
        and stores nothing. A vision tower has neither a vocabulary nor a tied
        head: it projects patches in and reads labels out.
        """
        if self.is_vision:
            # No vocabulary at all: what turns input into hidden states is one
            # convolution over each patch. Everything outside the blocks, in
            # one place -- patch projection, class token, position table, and
            # the classifier that makes it a classifier.
            patch = self.patch_size * self.patch_size * self.n_channels * self.hidden_size
            # The stored table, not the sequence that runs: a checkpoint trained
            # at 518 pixels carries 1370 rows however small the images it is
            # later handed.
            table = self.position_tokens * self.hidden_size if self.has_position_table else 0
            return patch + self.hidden_size + table + self.hidden_size * self.n_labels
        table = self.vocab_size * self.hidden_size
        if self.is_encoder:
            positions = self.max_position_embeddings if self.has_position_table else 0
            return table + positions * self.hidden_size
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
        # Gated MLP: gate and up projections in, down projection out. An encoder
        # predates the gate and has two matrices, so counting it as three
        # overstates a BERT block's feed-forward by half. A ViT block is the
        # same two. An encoder that has since adopted a gated MLP says so.
        matrices = 3 if self.kind == DECODER or self.gated_mlp else 2
        return matrices * self.hidden_size * self.intermediate_size

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
        if not self.has_kv_cache:
            # Nothing is cached between tokens: an encoder sees the whole
            # sequence at once and keeps nothing afterwards. This is the term
            # that dominates LLM serving and simply does not exist here.
            return 0
        return 2 * self.n_layers * self.n_kv_heads * self.head_dim * 2

    def describe(self) -> str:
        if self.is_vision:
            side = self.inference_image_size or self.image_size
            # Named when they differ, because "518px" beside "257 tokens" does
            # not divide and reads as an error rather than as two resolutions.
            trained = f" (trained at {self.image_size}px)" if side != self.image_size else ""
            return (
                f"{self.n_parameters / 1e6:.0f}M params, {self.n_layers} layers, "
                f"{side}px / {self.patch_size}px patches "
                f"= {self.n_image_tokens} tokens{trained}, {self.n_labels} classes"
            )
        if self.is_encoder:
            return (
                f"{self.n_parameters / 1e6:.0f}M params, {self.n_layers} layers, "
                f"hidden {self.hidden_size}, no KV cache"
            )
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


GATED_MLP_ACTIVATIONS = frozenset({"swiglu", "geglu", "glu", "reglu"})
"""Activation names that *are* a gated MLP, spelled out in the config.

Not a guess: an activation called ``swiglu`` names the third matrix. Nomic's
config says this plainly, which is the only reason its size can be read rather
than looked up.
"""

ABSOLUTE_POSITION_TYPES = frozenset({"absolute", ""})
"""``position_embedding_type`` values that mean a stored table.

Anything else -- ``rotary``, ``alibi``, ``relative_key`` -- computes position
instead of looking it up, so there is no table to count.
"""

ENCODER_BLOCK_OVERRIDES: dict[str, tuple[bool, bool]] = {
    # architecture stem -> (gated MLP, stores a position table)
    "ModernBert": (True, False),
    "JinaBert": (True, False),
    "NomicBert": (True, False),
}
"""Encoders whose block is not BERT's and whose config does not say so.

Each entry is a measured correction, not a guess. Counting these as classic
BERT blocks under-estimated them by 9% to 20% -- and under is the direction
that admits a candidate which then runs out of memory at serve time:

===============================  =========  =========
Model                            before     after
===============================  =========  =========
``ModernBERT-base``              90.75%     99.54%
``jina-embeddings-v2-base-en``   83.48%     99.51%
``nomic-embed-text-v1.5``        80.42%     99.98%
===============================  =========  =========

Three of them because three were checked. ModernBERT's config claims
``position_embedding_type: absolute`` and it uses rotary; Jina's says ``alibi``
but nothing about its GLU feed-forward; only Nomic's spells out ``swiglu``.

# ponytail: a stem table, like ENCODER_FAMILIES above it, not a registry.
# A family that is not listed keeps the classic-BERT arithmetic, which is what
# it got before this existed. Add one when its measured size disagrees.
"""


def _encoder_block(config: Any, architectures: Any) -> tuple[bool, bool]:
    """``(gated_mlp, has_position_table)`` for an encoder block."""
    names = [str(name) for name in (architectures or ())]
    for stem, override in ENCODER_BLOCK_OVERRIDES.items():
        if any(stem in name for name in names):
            return override

    activation = str(
        getattr(config, "activation_function", None) or getattr(config, "hidden_act", "")
    ).lower()
    position_type = str(getattr(config, "position_embedding_type", "") or "").lower()
    rotary = getattr(config, "rotary_emb_fraction", None)

    return (
        activation in GATED_MLP_ACTIVATIONS,
        position_type in ABSOLUTE_POSITION_TYPES and not rotary,
    )


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


def _kind_or_reject(architectures: Any, model_type: Any = None) -> str:
    """Which arithmetic describes this model, or refuse to guess.

    Three shapes are described here: a decoder's gated MLP and KV cache, an
    encoder's two-matrix MLP and no cache, and a vision tower's patch grid.
    The first two configs carry the same field names, so getting that wrong
    does not fail -- it returns a confident wrong answer, and a wrong memory
    estimate is worse than none because it is what a candidate is screened
    against.

    A config with no ``architectures`` is read as a decoder. That is a local or
    hand-built config where we cannot tell, and guessing wrong in the
    restrictive direction would block a model that works.

    A name matching neither shape is refused rather than assumed into one. A
    speculative draft proves the point: it is a decoder block whose name ends in
    neither suffix, and reading it as an encoder would apply a two-matrix MLP
    and a zero KV cache to a model with neither.
    """
    if (kind := model_kind(architectures, model_type)) is not None:
        return kind
    names = [str(name) for name in (architectures or ())]
    raise ValueError(
        f"{names[0]} is not a decoder-only language model, and not a recognised "
        f"encoder or image classifier either. AutoDistiller's memory arithmetic "
        f"does not describe it, and the estimate would be wrong rather than missing."
    )


def _vision_shape(
    model_id: str,
    config: Any,
    architecture: str | None,
    inference_image_size: int | None = None,
) -> ModelShape:
    """Dimensions of a plain vision transformer.

    Plain meaning uniform: one hidden size, one block repeated, one patch grid.
    That describes ViT and DeiT and stops there. Swin re-partitions and doubles
    its width every stage, ConvNeXt has no attention at all, and both spell
    their dimensions as lists -- so neither is measured with this arithmetic,
    and neither is quietly read as if it were.
    """
    missing = [
        name
        for name in ("image_size", "patch_size", "num_hidden_layers", "num_attention_heads")
        if _find_int(config, name) is None
    ]
    if missing:
        raise ValueError(
            f"{model_id} ({architecture}) classifies images but is not a plain vision "
            f"transformer: its config has no {', '.join(missing)}. A staged or "
            f"convolutional backbone (Swin, ConvNeXt) has a different width in every "
            f"stage, so this arithmetic would be wrong rather than missing."
        )

    hidden = _first_int(config, "hidden_size")
    n_heads = _first_int(config, "num_attention_heads")
    patch = _first_int(config, "patch_size")
    image = _first_int(config, "image_size")
    # The position table's height, which is a property of the weights. What
    # actually runs comes from the processor and can be smaller.
    n_tokens = (image // patch) ** 2 + 1

    return ModelShape(
        model_id=model_id,
        n_layers=_first_int(config, "num_hidden_layers"),
        hidden_size=hidden,
        intermediate_size=_first_int(config, "intermediate_size", default=4 * hidden),
        n_attention_heads=n_heads,
        n_kv_heads=n_heads,
        head_dim=_first_int(config, "head_dim", default=hidden // n_heads),
        # No tokenizer, so no vocabulary. The position table is one row per
        # patch, which is what max_position_embeddings holds for every other
        # kind -- so the field keeps its meaning and the number comes from the
        # patch grid instead of from a config field.
        vocab_size=0,
        max_position_embeddings=n_tokens,
        patch_size=patch,
        image_size=image,
        inference_image_size=inference_image_size or 0,
        n_channels=_first_int(config, "num_channels", default=3),
        n_labels=len(getattr(config, "id2label", None) or ()) or 0,
        architecture=architecture,
        kind=VISION,
    )


def shape_from_config(
    model_id: str, config: Any, *, inference_image_size: int | None = None
) -> ModelShape:
    """Extract dimensions from a loaded Hugging Face config object.

    ``inference_image_size`` is what the model's image processor produces, for a
    vision model whose training resolution differs from it. Passed in rather
    than read here, because this function is handed a config and must not go to
    the network to answer.
    """
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
    kind = _kind_or_reject(architectures, getattr(architectures_source, "model_type", None))
    architecture = architectures[0] if architectures else None

    if kind == VISION:
        return _vision_shape(model_id, config, architecture, inference_image_size)

    hidden = _first_int(config, "hidden_size", "n_embd", "d_model")
    n_heads = _first_int(config, "num_attention_heads", "n_head", "num_heads")
    # Only an encoder counts a position table or asks about its MLP shape; a
    # decoder's is gated by definition and it stores no table.
    gated_mlp, has_position_table = (
        _encoder_block(config, architectures) if kind == ENCODER else (False, True)
    )

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
        architecture=architecture,
        kind=kind,
        gated_mlp=gated_mlp,
        has_position_table=has_position_table,
    )


def processor_image_size(
    model_id: str, *, revision: str | None = None, trust_remote_code: bool = False
) -> int | None:
    """The image side a model's own processor produces, or None if unreadable.

    A few kilobytes of JSON, like the config beside it, so this keeps the
    module's promise that a shape costs no model download. Best effort: a
    checkpoint with no processor is not an error, it just leaves the config's
    training resolution standing.
    """
    try:
        from transformers import AutoImageProcessor

        processor = AutoImageProcessor.from_pretrained(
            model_id, revision=revision, trust_remote_code=trust_remote_code
        )
    except Exception:  # no processor, no network, or a format we cannot read
        return None

    # Crop wins where there is one: it is the last thing applied, so it is the
    # size the model actually sees.
    for attribute in ("crop_size", "size"):
        spec = getattr(processor, attribute, None)
        if spec is None:
            continue
        for key in ("height", "width", "shortest_edge"):
            value = spec[key] if isinstance(spec, dict) else getattr(spec, key, None)
            if isinstance(value, int) and value > 0:
                return value
    return None


def load_shape(model_id: str, *, revision: str | None = None, trust_remote_code: bool = False):
    """Fetch just the config for a model and derive its shape."""
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(
        model_id, revision=revision, trust_remote_code=trust_remote_code
    )
    inference_image_size = (
        processor_image_size(model_id, revision=revision, trust_remote_code=trust_remote_code)
        if kind_of_config(config) == VISION
        else None
    )
    return shape_from_config(model_id, config, inference_image_size=inference_image_size)


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
    "processor_image_size",
    "shape_from_config",
    "to_record",
]
