"""Speculative decoding as a searchable dimension.

Speculative decoding is not compression. It does not touch the target model's
weights, and verification guarantees the target's own output distribution -- so
quality retention is 1.0 by construction and the evaluation machinery has
nothing to measure. What it costs is memory (a second model resident) and what
it buys is decode throughput, which are exactly the axes the deployment
benchmark already reports. That makes it a candidate dimension rather than a
compression method: same search, same measurements, one more thing varied.

The draft checkpoint is named by the user and never derived. Drafts are trained
against one specific target and published under several organizations with
several suffixes -- ``z-lab/Qwen3.6-27B-DFlash``, ``incoai/Qwen3.8-27B-DFlash2``,
``RedHatAI/gemma-4-31B-it-speculator.dflash`` -- so a guessed repo id is a
plausible-looking string that 404s in the middle of a search.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from .shape import ModelShape, _find_int, _first_int

DFLASH = "dflash"

DTYPE_BYTES = {"float32": 4, "float16": 2, "bfloat16": 2, "float64": 8}
"""Bytes per weight for a draft's declared dtype. Drafts ship as published;
AutoDistiller does not quantize them."""


@dataclass(frozen=True)
class SpeculativeSpec:
    """A draft model, how many tokens it proposes, and what it costs to hold."""

    method: str
    model: str
    n_tokens: int
    weights_bytes: int = 0

    @property
    def label(self) -> str:
        return f"{self.method}{self.n_tokens}"

    def as_config(self) -> str:
        """The JSON vLLM's ``--speculative-config`` takes."""
        return json.dumps(
            {
                "method": self.method,
                "model": self.model,
                "num_speculative_tokens": self.n_tokens,
            }
        )

    def describe(self) -> str:
        return (
            f"{self.method} draft {self.model}, {self.n_tokens} tokens/step, "
            f"{self.weights_bytes / 1024**3:.2f} GiB resident"
        )


def draft_shape_from_config(model_id: str, config: Any) -> ModelShape:
    """A draft's dimensions, read the same way as a target's.

    Parsed here rather than through :func:`~.shape.shape_from_config` because a
    draft is not a servable causal LM: ``DFlashDraftModel`` is non-causal, and
    the guard there correctly refuses to describe it. It is still an ordinary
    stack of gated-MLP transformer blocks -- ``model_type`` is the target's own
    family -- so :class:`ModelShape`'s arithmetic describes it exactly. Only the
    validation differs, which is why the class is reused and the parser is not.
    """
    hidden = _first_int(config, "hidden_size")
    n_heads = _first_int(config, "num_attention_heads")
    architectures = getattr(config, "architectures", None)

    return ModelShape(
        model_id=model_id,
        n_layers=_first_int(config, "num_hidden_layers"),
        hidden_size=hidden,
        intermediate_size=_first_int(config, "intermediate_size", default=4 * hidden),
        n_attention_heads=n_heads,
        n_kv_heads=_first_int(config, "num_key_value_heads", default=n_heads),
        head_dim=_first_int(config, "head_dim", default=hidden // n_heads),
        vocab_size=_first_int(config, "vocab_size"),
        max_position_embeddings=_first_int(config, "max_position_embeddings", default=4096),
        tie_word_embeddings=bool(getattr(config, "tie_word_embeddings", False)),
        architecture=architectures[0] if architectures else None,
    )


def _block_size(config: Any) -> int | None:
    """Tokens the draft emits in one pass, however the checkpoint spells it.

    Top level in DFlash, inside ``dflash_config`` in DFlash2.
    """
    if (top := _find_int(config, "block_size")) is not None:
        return top
    nested = getattr(config, "dflash_config", None) or {}
    value = nested.get("block_size") if isinstance(nested, dict) else None
    return value if isinstance(value, int) and value > 0 else None


def _draft_weight_bytes(shape: ModelShape, config: Any) -> int:
    """Bytes the draft's weights occupy, at the dtype it was published in.

    # ponytail: counts every parameter the config declares, embeddings included.
    # A draft conditions on the target's hidden states and may well share its
    # embedding table rather than materializing a second one -- on a 248k vocab
    # that is 2.5 GiB of the total. Over-counting keeps a candidate off the list
    # that would otherwise OOM at serve time, which is the direction this module
    # is meant to fail in. Narrow it if a measured peak comes in well under.
    """
    dtype = str(getattr(config, "dtype", None) or getattr(config, "torch_dtype", None) or "")
    return shape.n_parameters * DTYPE_BYTES.get(dtype, 2)


MASK_TOKEN_FIELDS = (
    "mask_token_id",
    "dspark_noise_token_id",
    "pard_token",
    "ptd_token_id",
)
"""What a draft must name for a runtime to draft a block in parallel.

Published checkpoints do not all carry one. The ``QAT-*`` variants of an
otherwise fine draft omit it, and the omission is invisible until a server has
already been started -- after the candidate has been compressed and evaluated,
minutes in, with an error that names the field but not the checkpoint.
"""


def _reject_if_undraftable(draft_model: str, config: Any) -> None:
    """Refuse a draft a runtime will not accept, before anything is spent on it."""
    nested = getattr(config, "dflash_config", None) or {}
    nested = nested if isinstance(nested, dict) else {}
    if any(
        getattr(config, field, None) is not None or nested.get(field) is not None
        for field in MASK_TOKEN_FIELDS
    ):
        return
    raise ValueError(
        f"{draft_model} names none of {', '.join(MASK_TOKEN_FIELDS)}, so a serving "
        f"runtime cannot use it for parallel drafting. This is a property of the "
        f"checkpoint, not of your hardware -- pick another draft for this model."
    )


def resolve_speculative(
    draft_model: str,
    *,
    method: str = DFLASH,
    n_tokens: int | None = None,
    revision: str | None = None,
) -> SpeculativeSpec:
    """Read a published draft checkpoint into a spec, without downloading weights.

    ``n_tokens`` defaults to one less than the draft's block size: the block is
    what it emits in a pass, and the last slot is the target's own verified
    token rather than a proposal. Both published families follow it -- block 16
    is served with 15, block 8 with 7.
    """
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(draft_model, revision=revision, trust_remote_code=False)
    _reject_if_undraftable(draft_model, config)
    shape = draft_shape_from_config(draft_model, config)

    if n_tokens is None:
        block = _block_size(config)
        if block is None:
            raise ValueError(
                f"{draft_model} declares no block size, so the tokens per step cannot be "
                f"derived. Pass --speculative-tokens explicitly."
            )
        n_tokens = max(block - 1, 1)

    return SpeculativeSpec(
        method=method,
        model=draft_model,
        n_tokens=n_tokens,
        weights_bytes=_draft_weight_bytes(shape, config),
    )


__all__ = [
    "DFLASH",
    "MASK_TOKEN_FIELDS",
    "SpeculativeSpec",
    "draft_shape_from_config",
    "resolve_speculative",
]
