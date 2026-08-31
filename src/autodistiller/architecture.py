"""What kind of model this is, from its architecture name alone.

Three modules need the same answer for different reasons -- the loader picks an
Auto class with it, the shape estimator screens on it, and the compression
methods declare which kinds they can be produced from -- so it lives here rather
than as three copies that drift.

Deliberately free of imports from the rest of the package, and of torch. A
config is a few kilobytes and answering this must not cost a model load.
"""

from __future__ import annotations

from typing import Any

DECODER = "decoder"
ENCODER = "encoder"

DECODER_SUFFIXES = ("ForCausalLM", "ForConditionalGeneration")
"""Architecture names that predict a next token.

A vision-language model ends in ``ForConditionalGeneration`` and its decoder is
what gets quantized, which is why the suffix counts. An encoder-decoder shares
that suffix and is not a decoder-only model, so callers that care about the KV
cache check ``is_encoder_decoder`` separately.
"""


def model_kind(architectures: Any) -> str:
    """``decoder`` or ``encoder``, from a config's ``architectures`` list.

    Decoder is the default, and this departs from it only when the config says
    outright that the model is something else. A config with no ``architectures``
    is a local or hand-written one, and reading that silence as "encoder" would
    change how every such checkpoint has been treated since before encoders were
    supported at all.
    """
    names = [str(name) for name in (architectures or ())]
    if names and not any(name.endswith(DECODER_SUFFIXES) for name in names):
        return ENCODER
    return DECODER


def kind_of_config(config: Any) -> str:
    """``model_kind`` for an already-loaded Hugging Face config object."""
    return model_kind(getattr(config, "architectures", None))


__all__ = [
    "DECODER",
    "DECODER_SUFFIXES",
    "ENCODER",
    "kind_of_config",
    "model_kind",
]
