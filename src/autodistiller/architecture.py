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
VISION = "vision"

DECODER_SUFFIXES = ("ForCausalLM", "ForConditionalGeneration")
"""Architecture names that predict a next token.

A vision-language model ends in ``ForConditionalGeneration`` and its decoder is
what gets quantized, which is why the suffix counts. An encoder-decoder shares
that suffix and is not a decoder-only model, so callers that care about the KV
cache check ``is_encoder_decoder`` separately.
"""


ENCODER_SUFFIXES = ("ForMaskedLM", "ForSequenceClassification", "ForTokenClassification")
"""Heads only an encoder wears."""

VISION_SUFFIXES = ("ForImageClassification",)
"""The head an image classifier wears.

Narrower than "the model looks at pixels" on purpose. A vision-language model
ends in ``ForConditionalGeneration`` and is a decoder with an image tower
bolted on; this is the other thing -- a tower on its own, ending in a label.
Only the second has no KV cache and a token count fixed by its resolution,
which is what the arithmetic downstream turns on.
"""


ENCODER_FAMILIES = ("Bert", "Roberta", "Deberta", "Electra", "MPNet", "XLM")
"""Architecture stems of the BERT family, which is what an embedding model is.

Matching on the name is crude, and deliberately so: the alternative is guessing
from config fields that decoders and encoders spell identically, which is the
mistake this module exists to stop.

# ponytail: a stem list, not a registry. Add one when a family shows up that
# nothing here matches -- an unrecognised name is refused, not misread, so the
# failure is a clear message rather than a wrong number.
"""


ENCODER_MODEL_TYPES = tuple(stem.lower() for stem in ENCODER_FAMILIES)
"""The same stems, matched against ``model_type`` instead of a class name.

For configs that carry no ``architectures`` at all. DeBERTa-v3 is the one that
forced this: `microsoft/deberta-v3-base` omits the key entirely, and the rule
below reads that as a decoder -- which gave it a gated MLP and a KV cache it
does not have, a 0.21B estimate for a 184M encoder. Its ``model_type`` says
``deberta-v2`` and was sitting there unread.

Substring matching, so ``deberta-v2`` matches ``deberta`` and ``xlm-roberta``
matches both of its halves.
"""


def model_kind(architectures: Any, model_type: Any = None) -> str | None:
    """``decoder``, ``encoder``, ``vision``, or ``None`` when the name does not say.

    Absence of a decoder suffix is not evidence of an encoder. A speculative
    draft checkpoint is a decoder block that ends in neither -- reading that as
    an encoder would apply two-matrix-MLP arithmetic and a zero KV cache to a
    model that has neither property, which is a confident wrong answer of
    exactly the kind a memory screen must not produce. So every kind is
    recognised positively and everything else returns ``None`` for the caller
    to decide about.

    A config with no ``architectures`` falls back to ``model_type``, and then to
    ``decoder`` if that says nothing either -- a local or hand-written config is
    read as a decoder because that is how every such checkpoint has been treated
    since before encoders were described here.
    """
    names = [str(name) for name in (architectures or ())]
    if not names:
        spelling = str(model_type or "").lower()
        if spelling and any(stem in spelling for stem in ENCODER_MODEL_TYPES):
            return ENCODER
        return DECODER
    if any(name.endswith(DECODER_SUFFIXES) for name in names):
        return DECODER
    if any(name.endswith(VISION_SUFFIXES) for name in names):
        return VISION
    if any(
        name.endswith(ENCODER_SUFFIXES) or any(stem in name for stem in ENCODER_FAMILIES)
        for name in names
    ):
        return ENCODER
    return None


def kind_of_config(config: Any) -> str | None:
    """``model_kind`` for an already-loaded Hugging Face config object."""
    return model_kind(getattr(config, "architectures", None), getattr(config, "model_type", None))


__all__ = [
    "DECODER",
    "DECODER_SUFFIXES",
    "ENCODER",
    "ENCODER_FAMILIES",
    "ENCODER_MODEL_TYPES",
    "ENCODER_SUFFIXES",
    "VISION",
    "VISION_SUFFIXES",
    "kind_of_config",
    "model_kind",
]
