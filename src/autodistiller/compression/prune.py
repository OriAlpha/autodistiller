"""Depth pruning: drop whole transformer blocks.

The one reduction both runtimes can actually serve. 2:4 sparsity would have been
the cheaper answer -- llmcompressor produces it and the format still exists in
compressed-tensors -- but vLLM 0.27 removed sparsity support entirely (no
schemes, no kernels, and a hard error on any ``sparsity_config``), and
llama.cpp never had it. A depth-pruned model is an ordinary dense checkpoint
with a smaller ``num_hidden_layers``, so every runtime serves it with no special
kernel, and it composes with quantization: prune, then hand the directory to
:func:`~autodistiller.compression.pipeline.run_compression` as the model.

Which layers to drop comes from *block influence*: a layer whose output barely
differs from its input is doing little, so score each block by the angular
distance it moves the residual stream and drop the smallest. One forward pass
over calibration text, no gradients, no training.

Unlike quantization this is not a mild perturbation. Without a healing pass --
which needs a trainer, and is deliberately out of scope here -- quality falls off
a cliff, and where that cliff sits is model-specific. Measuring it is the point:
``evaluate`` the result before deciding to serve it.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from ..config import DatasetSpec, ModelSpec
from ..metadata.hashing import hash_obj, hash_text_stream
from ..results import CompressionArtifact, CompressionRecipe

logger = logging.getLogger(__name__)

PRUNE_SCHEME = "depth"
"""Marks a recipe as a prune rather than a quantization.

A pruned artifact reuses ``CompressionRecipe`` -- it is produced, cached and
exported like any other -- so something has to tell the two apart when reading
one back.
"""

DEFAULT_MAX_LENGTH = 2048
DEFAULT_SAMPLES = 32
"""Documents the block scores are averaged over.

Scoring is one forward pass per document and the scores are a ranking, not a
measurement -- the gap between the least influential blocks and the rest is
wide, and a few dozen documents settle the order.
"""

ProgressFn = Callable[[str], None]


@dataclass
class PruneJob:
    """One depth-pruning request.

    Carries ``output_dir`` and ``artifact_key`` under the same names a
    :class:`~autodistiller.compression.backend.CompressionJob` uses, so the
    sidecar and reuse machinery in :mod:`.pipeline` works on it unchanged.
    """

    model: ModelSpec
    n_drop: int
    calibration_texts: list[str] = field(default_factory=list)
    max_length: int = DEFAULT_MAX_LENGTH
    output_dir: Path = Path()

    @property
    def method(self) -> str:
        return f"prune{self.n_drop}"

    def recipe(self) -> CompressionRecipe:
        return CompressionRecipe(
            method=self.method,
            scheme=PRUNE_SCHEME,
            algorithm="block-influence",
            # Pruning removes weights, it does not narrow them. The survivors
            # are still whatever the source checkpoint stored them as.
            weight_bits=16,
            activation_bits=16,
            needs_calibration=True,
            n_calibration_samples=len(self.calibration_texts),
            max_seq_length=self.max_length,
            calibration_fingerprint=(
                hash_text_stream(self.calibration_texts) if self.calibration_texts else None
            ),
        )

    @property
    def artifact_key(self) -> str:
        """Identity of the weights this job would produce.

        The scores decide which layers go, and the scores are a function of the
        calibration text -- so two jobs that differ only in calibration data are
        different artifacts even at the same drop count.
        """
        return hash_obj(
            {
                "recipe": self.recipe().model_dump(mode="json"),
                "model_id": self.model.id,
                "revision": self.model.revision,
                "dtype": self.model.dtype,
            }
        )


def build_prune_job(
    model: ModelSpec,
    n_drop: int,
    *,
    calibration: DatasetSpec | None,
    num_samples: int = DEFAULT_SAMPLES,
    max_length: int = DEFAULT_MAX_LENGTH,
    output_root: Path = Path("artifacts"),
    output_dir: Path | None = None,
) -> PruneJob:
    """Resolve a pruning request into a runnable job.

    The counterpart of :func:`~autodistiller.compression.pipeline.build_job`, so
    the CLI and the optimizer reach a pruned artifact by the same path and land
    on the same content-addressed directory.
    """
    from ..evaluation.datasets import load_text_corpus
    from .pipeline import artifact_dir

    if calibration is None:
        raise ValueError(
            "depth pruning scores blocks against real activations, so it needs "
            "calibration data; pass --calibration"
        )

    corpus = load_text_corpus(calibration)
    texts = corpus.documents[:num_samples]
    if not texts:
        raise ValueError(f"calibration dataset {corpus.source} produced no documents")

    job = PruneJob(model=model, n_drop=n_drop, calibration_texts=texts, max_length=max_length)
    job.output_dir = (
        Path(output_dir)
        if output_dir
        else artifact_dir(model.id, job.method, output_root, key=job.artifact_key)
    )
    return job


def layer_list(model: Any) -> tuple[str, torch.nn.ModuleList]:
    """The decoder's block list, whatever this architecture calls it.

    ``model.layers`` on Llama and Qwen, ``transformer.h`` on GPT-2 and Falcon,
    ``decoder.layers`` on OPT. Rather than keep a table of those, find the
    ``ModuleList`` as long as the config says the model is deep.

    ``named_modules`` is depth-first pre-order, so the decoder's own list is
    reached before anything nested inside it -- which is what keeps a MoE
    model's expert list from matching first when it happens to hold as many
    experts as the model has layers.
    """
    n_layers = model.config.num_hidden_layers
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.ModuleList) and len(module) == n_layers:
            return name, module
    raise ValueError(
        f"no block list of length {n_layers} found in {type(model).__name__}; "
        "this architecture stores its layers somewhere unexpected"
    )


def block_influence(
    model: Any,
    tokenizer: Any,
    texts: list[str],
    *,
    device: torch.device,
    max_length: int = DEFAULT_MAX_LENGTH,
) -> list[float]:
    """How much each block moves the residual stream, one score per layer.

    ``1 - cos(input, output)`` averaged over tokens and documents. A block that
    returns its input unchanged scores 0 and is the cheapest thing to remove.

    Documents are run one at a time rather than batched: padding positions would
    otherwise have to be masked out of the average, and a handful of short
    forward passes is not the expensive part of pruning.
    """
    n_layers = model.config.num_hidden_layers
    totals = [0.0] * n_layers
    counted = 0

    for text in texts:
        encoded = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
        input_ids = encoded["input_ids"].to(device)
        if input_ids.shape[1] < 2:
            continue

        with torch.no_grad():
            hidden = model(input_ids, output_hidden_states=True, use_cache=False).hidden_states

        # One state before the first block and one after each of them.
        if len(hidden) != n_layers + 1:
            raise ValueError(
                f"expected {n_layers + 1} hidden states, got {len(hidden)}; "
                "this model does not expose one state per block"
            )

        for index in range(n_layers):
            similarity = torch.nn.functional.cosine_similarity(
                hidden[index].float(), hidden[index + 1].float(), dim=-1
            )
            totals[index] += float(1.0 - similarity.mean())
        counted += 1

    if not counted:
        raise ValueError("no calibration document survived tokenization")

    return [total / counted for total in totals]


def choose_layers(influence: list[float], n_drop: int) -> tuple[list[int], list[int]]:
    """Which layers to drop, and which to keep, lowest influence first.

    The final block is never a candidate. Its output is what the final norm and
    the output head were fitted against, and removing it is a much larger change
    than its influence score suggests.
    """
    n_layers = len(influence)
    droppable = list(range(n_layers - 1))
    if not 0 < n_drop <= len(droppable):
        raise ValueError(
            f"cannot drop {n_drop} of {n_layers} layers; 1 to {len(droppable)} is the range "
            "(the last block is never dropped)"
        )

    dropped = sorted(sorted(droppable, key=lambda i: influence[i])[:n_drop])
    keep = [i for i in range(n_layers) if i not in set(dropped)]
    return dropped, keep


def slice_per_layer_config(config: Any, keep: list[int], n_layers: int) -> list[str]:
    """Cut per-layer config lists down to the surviving layers.

    Modern configs describe layers individually -- Qwen3's ``layer_types``,
    Gemma's alternating attention pattern -- and a list still describing the
    original depth makes the saved checkpoint fail to reload, long after the
    pruning itself reported success.

    A list that happens to be as long as the model is deep without being
    per-layer would be sliced too. Returns what it touched so the caller can say
    so rather than leave it silent.
    """
    sliced = []
    for key, value in list(vars(config).items()):
        if isinstance(value, list) and len(value) == n_layers:
            setattr(config, key, [value[i] for i in keep])
            sliced.append(key)
    return sliced


def _directory_size(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def run_prune(job: PruneJob, *, progress: ProgressFn | None = None) -> CompressionArtifact:
    """Score the blocks, drop the least influential, save what is left."""
    from ..models.loader import loaded_model

    started = time.perf_counter()

    def say(message: str) -> None:
        if progress is not None:
            progress(message)

    with loaded_model(job.model) as handle:
        model = handle.model
        n_layers = model.config.num_hidden_layers

        say(f"scoring {n_layers} blocks over {len(job.calibration_texts)} documents")
        influence = block_influence(
            model,
            handle.tokenizer,
            job.calibration_texts,
            device=handle.device,
            max_length=job.max_length,
        )
        dropped, keep = choose_layers(influence, job.n_drop)
        say(f"dropping layers {dropped}, influence {[round(influence[i], 4) for i in dropped]}")

        name, blocks = layer_list(model)
        kept = torch.nn.ModuleList([blocks[i] for i in keep])
        parent_name, _, attribute = name.rpartition(".")
        setattr(model.get_submodule(parent_name) if parent_name else model, attribute, kept)

        if touched := slice_per_layer_config(model.config, keep, n_layers):
            say(f"sliced per-layer config: {', '.join(touched)}")
        model.config.num_hidden_layers = len(keep)

        job.output_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(job.output_dir)
        handle.tokenizer.save_pretrained(job.output_dir)

    import transformers

    artifact = CompressionArtifact(
        recipe=job.recipe(),
        backend="autodistiller-prune",
        source_model=job.model.id,
        output_dir=str(job.output_dir),
        artifact_bytes=_directory_size(job.output_dir),
        duration_s=time.perf_counter() - started,
        versions={"transformers": transformers.__version__, "torch": torch.__version__},
    )
    logger.info("pruned %s: dropped %s, kept %s layers", job.model.id, dropped, len(keep))
    return artifact


__all__ = [
    "DEFAULT_MAX_LENGTH",
    "DEFAULT_SAMPLES",
    "PRUNE_SCHEME",
    "PruneJob",
    "block_influence",
    "build_prune_job",
    "choose_layers",
    "layer_list",
    "run_prune",
    "slice_per_layer_config",
]
