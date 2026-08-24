"""Hugging Face model loading.

Loading is delegated to Transformers. What matters here is recording exactly
what got loaded, because that identity is what every later comparison depends
on.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import transformers
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from ..metadata.hashing import hash_obj
from ..results import ModelInfo

logger = logging.getLogger(__name__)

_TRANSFORMERS_MAJOR = int(transformers.__version__.split(".")[0])

# Transformers renamed `torch_dtype` to `dtype` in v5. Support both so the
# project is not pinned to a single major version.
_DTYPE_KWARG = "dtype" if _TRANSFORMERS_MAJOR >= 5 else "torch_dtype"

MAX_DEFAULT_CONTEXT = 2048
"""Cap on the auto-selected evaluation window.

Modern configs advertise 128k+ contexts. Evaluating perplexity at that width
would allocate enormous logit buffers for no screening benefit, and 2048 is the
window the quantization literature conventionally reports against, so the
automatic choice is capped and always written into the run record.
"""

_DTYPE_MAP: dict[str, torch.dtype] = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


@dataclass
class LoadedModel:
    """A loaded model plus the provenance needed to trust its numbers."""

    model: Any
    tokenizer: Any
    info: ModelInfo
    device: torch.device
    dtype: torch.dtype

    @property
    def context_length(self) -> int:
        return self.info.context_length or MAX_DEFAULT_CONTEXT

    def free(self) -> None:
        """Release the model and empty the CUDA allocator cache."""
        self.model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def resolve_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_dtype(requested: str, device: torch.device) -> torch.dtype:
    """Pick a dtype, preferring bf16 on hardware that supports it.

    bf16 avoids the overflow failure modes fp16 has on some architectures, which
    matters when the whole point is a clean reference measurement.
    """
    if requested != "auto":
        return _DTYPE_MAP[requested]
    if device.type == "cuda":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    if device.type == "mps":
        return torch.float16
    return torch.float32  # CPU fp16 is slow and often unsupported


def resolve_context_length(config: Any, override: int | None) -> int:
    if override:
        return override
    for attr in ("max_position_embeddings", "n_positions", "seq_length", "max_seq_len"):
        value = getattr(config, attr, None)
        if isinstance(value, int) and value > 0:
            return min(value, MAX_DEFAULT_CONTEXT)
    return MAX_DEFAULT_CONTEXT


def _resolve_commit(model_id: str, revision: str | None) -> tuple[str | None, bool]:
    """Return ``(commit_sha, is_local)`` for a model reference."""
    if Path(model_id).exists():
        return None, True
    try:
        from huggingface_hub import model_info

        return model_info(model_id, revision=revision).sha, False
    except Exception as exc:  # offline, gated, or not a hub repo
        logger.debug("Could not resolve commit for %s: %s", model_id, exc)
        return None, False


def _architecture_fingerprint(model: Any) -> str:
    """Hash parameter names, shapes and dtypes.

    This identifies the *structure* that was loaded. Two runs whose
    architecture fingerprints differ are not comparable, regardless of what the
    repo id says. That is the check that catches a silently changed revision.
    """
    entries = [
        (name, tuple(param.shape), str(param.dtype))
        for name, param in sorted(model.named_parameters(), key=lambda kv: kv[0])
    ]
    return hash_obj(entries)


def _weights_size_bytes(model: Any) -> int:
    params = sum(p.numel() * p.element_size() for p in model.parameters())
    buffers = sum(b.numel() * b.element_size() for b in model.buffers())
    return params + buffers


def gguf_file_in(model_id: str) -> str | None:
    """The GGUF filename inside an artifact directory, if that is what this is.

    Transformers reads GGUF by being handed the directory plus the filename, and
    it dequantizes on load: the weights come back as ordinary tensors carrying
    the quantization error. That is exactly what a quality screen wants to
    measure, and exactly not a claim about llama.cpp's inference kernels --
    which is why deployment numbers still come from llama-server itself.
    """
    directory = Path(model_id)
    if not directory.is_dir():
        return None
    found = sorted(directory.glob("*.gguf"))
    return found[0].name if found else None


def load_model(spec: Any) -> LoadedModel:
    """Load a model + tokenizer described by a :class:`~autodistiller.config.ModelSpec`."""
    device = resolve_device(spec.device)
    dtype = resolve_dtype(spec.dtype, device)
    commit, is_local = _resolve_commit(spec.id, spec.revision)

    logger.info("Loading %s (dtype=%s, device=%s)", spec.id, dtype, device)

    common: dict[str, Any] = {
        "revision": spec.revision,
        "trust_remote_code": spec.trust_remote_code,
    }

    if (gguf_file := gguf_file_in(spec.id)) is not None:
        logger.info("Loading %s as GGUF (%s), dequantized for evaluation", spec.id, gguf_file)
        common["gguf_file"] = gguf_file

    config = AutoConfig.from_pretrained(spec.id, **common)
    tokenizer = AutoTokenizer.from_pretrained(spec.id, use_fast=True, **common)
    if tokenizer.pad_token_id is None:
        # Needed for batched evaluation; padded positions are always masked out.
        tokenizer.pad_token = tokenizer.eos_token

    load_kwargs: dict[str, Any] = {**common, _DTYPE_KWARG: dtype}
    if spec.attn_implementation:
        load_kwargs["attn_implementation"] = spec.attn_implementation

    model: Any = AutoModelForCausalLM.from_pretrained(spec.id, config=config, **load_kwargs)
    model.to(device)
    model.eval()
    model.config.use_cache = True

    architectures = getattr(config, "architectures", None)
    info = ModelInfo(
        id=spec.id,
        revision=spec.revision,
        resolved_commit=commit,
        architecture=architectures[0] if architectures else type(model).__name__,
        dtype=str(dtype).replace("torch.", ""),
        device=str(device),
        n_parameters=sum(p.numel() for p in model.parameters()),
        context_length=resolve_context_length(config, spec.max_position_embeddings),
        vocab_size=getattr(config, "vocab_size", None),
        weights_size_bytes=_weights_size_bytes(model),
        architecture_fingerprint=_architecture_fingerprint(model),
        is_local=is_local,
    )

    return LoadedModel(model=model, tokenizer=tokenizer, info=info, device=device, dtype=dtype)


@contextmanager
def loaded_model(spec: Any) -> Iterator[LoadedModel]:
    """Load a model and guarantee VRAM is released afterwards."""
    handle = load_model(spec)
    try:
        yield handle
    finally:
        handle.free()
