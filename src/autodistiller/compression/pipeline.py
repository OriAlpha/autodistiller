"""Turning a compression request into an artifact.

Sits between the CLI and the backend adapter: resolves the method, checks it
against the hardware and the serving backend, gathers calibration text through
the same loader Phase 1 evaluates with, and hands a fully-specified job down.

Calibration data comes from AutoDistiller rather than being named for the
backend to fetch, so it is fingerprinted with the same machinery as evaluation
data and two runs can be shown to have used the same bytes.

Artifacts are content-addressed by that job. Naming a directory after the model
and method alone is not enough to identify what is in it -- the same model and
method with different calibration data produce genuinely different weights -- so
the recipe's fingerprint is part of the path. Directories are then safe to reuse
rather than overwrite, which is what makes compression cacheable at all: it is
the most expensive step the optimizer repeats.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

from ..config import CompressionSpec, ModelSpec
from ..evaluation.datasets import load_text_corpus
from ..metadata.profiles import GPUProfile
from ..results import CompressionArtifact
from ..serving.launcher import wsl_path
from .backend import CompressionJob, ProgressFn, resolve_compression_backend
from .methods import check_method, resolve_method

logger = logging.getLogger(__name__)

_SLUG = re.compile(r"[^a-zA-Z0-9._-]+")

ARTIFACT_SIDECAR = "autodistiller-artifact.json"
"""Records what an artifact directory holds.

Named for AutoDistiller because llmcompressor writes its own ``recipe.yaml``
there and the two should not be confused.
"""


def artifact_dir(model_id: str, method: str, root: Path, key: str | None = None) -> Path:
    """Where an artifact lands: ``<root>/<model>-<method>-<key>``.

    The key makes the path identify the weights. Without it, compressing one
    model with one method but two different calibration sets writes both to the
    same directory, and the second silently replaces the first -- while every
    record already written points at the path and describes the weights that
    used to be there.
    """
    slug = _SLUG.sub("-", model_id.split("/")[-1]).strip("-") or "model"
    stem = f"{slug}-{method}"
    return root / (f"{stem}-{key[:8]}" if key else stem)


def build_job(
    model: ModelSpec,
    spec: CompressionSpec,
    *,
    output_root: Path = Path("artifacts"),
) -> CompressionJob:
    """Resolve a spec into a runnable job, failing early on bad combinations."""
    method = resolve_method(spec.method)

    calibration_texts: list[str] = []
    if method.needs_calibration:
        if spec.calibration is None:
            raise ValueError(
                f"method {method.name!r} ({method.algorithm}) needs calibration data; "
                f"set compression.calibration or pass --calibration"
            )
        corpus = load_text_corpus(spec.calibration)
        calibration_texts = corpus.documents[: spec.num_calibration_samples]
        if not calibration_texts:
            raise ValueError(f"calibration dataset {corpus.source} produced no documents")
    elif spec.calibration is not None:
        # Not an error, but the user probably expected it to matter.
        calibration_texts = []

    job = CompressionJob(
        model_id=model.id,
        method=method,
        # Filled in below. The default path is derived from the job's own
        # identity, so the job has to exist before its directory is known.
        output_dir=Path(),
        calibration_texts=calibration_texts,
        num_calibration_samples=spec.num_calibration_samples,
        max_seq_length=spec.max_seq_length,
        ignore=tuple(spec.ignore),
        trust_remote_code=model.trust_remote_code,
        dtype=model.dtype if model.dtype != "auto" else "auto",
    )
    job.output_dir = (
        Path(spec.output_dir)
        if spec.output_dir
        else artifact_dir(model.id, method.name, output_root, key=job.artifact_key)
    )
    return job


def _is_complete(directory: Path) -> bool:
    """Whether a directory holds servable weights rather than a failed attempt.

    A crashed or interrupted run leaves the directory and the config behind but
    no weights, and reusing that would fail much later and much less clearly.

    Either shape counts: a Hugging Face directory from llmcompressor, or a GGUF
    file from llama.cpp. Asking the shape-agnostic question here saves threading
    the method through every caller to answer it.
    """
    if any(directory.glob("*.gguf")):
        return True
    return (directory / "config.json").is_file() and any(directory.glob("*.safetensors"))


def read_cached_artifact(job: CompressionJob) -> CompressionArtifact | None:
    """The artifact already sitting in this job's directory, if it is the right one."""
    sidecar = job.output_dir / ARTIFACT_SIDECAR
    if not sidecar.is_file() or not _is_complete(job.output_dir):
        return None

    try:
        payload = json.loads(sidecar.read_text(encoding="utf-8"))
        stored_key = payload.pop("artifact_key", None)
        artifact = CompressionArtifact.model_validate(payload)
    except (json.JSONDecodeError, ValueError) as exc:
        logger.debug("ignoring unreadable artifact sidecar %s: %s", sidecar, exc)
        return None

    if stored_key != job.artifact_key:
        # The path is derived from the key, so this means a hand-picked
        # --output-dir now holds something else. Recompressing would overwrite
        # weights the user put there, so refuse instead.
        raise ValueError(
            f"{job.output_dir} already holds a different artifact "
            f"({artifact.recipe.label}, key {stored_key}). "
            f"Choose another --output-dir, or delete it first."
        )

    return artifact


def write_artifact_sidecar(job: CompressionJob, artifact: CompressionArtifact) -> Path:
    """Record what was produced, next to the weights themselves."""
    path = job.output_dir / ARTIFACT_SIDECAR
    payload = artifact.model_dump(mode="json") | {"artifact_key": job.artifact_key}
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def run_compression(
    model: ModelSpec,
    spec: CompressionSpec,
    *,
    output_root: Path = Path("artifacts"),
    profile: GPUProfile | None = None,
    serving_backend: str | None = None,
    reuse: bool = True,
    progress: ProgressFn | None = None,
) -> CompressionArtifact:
    """Produce one compressed artifact, or reuse the identical one already on disk.

    ``profile`` and ``serving_backend`` are checked before any work starts:
    producing an artifact the GPU cannot run, or that the target runtime cannot
    serve, wastes minutes to reach a useless result.
    """
    method = resolve_method(spec.method)

    # Naming a method is naming a runtime, so an unspecified serving backend
    # means "whichever one serves this". The check still fires when the caller
    # asks for a combination that cannot work -- a GGUF built for vLLM is
    # minutes spent reaching a useless result -- but it should not fire on a
    # default the user never chose.
    target = serving_backend or (method.backends[0] if method.backends else None)
    availability = check_method(method, profile=profile, backend=target)
    if not availability.available:
        raise ValueError(f"{method.name} is not usable here: {'; '.join(availability.reasons)}")

    job = build_job(model, spec, output_root=output_root)

    if reuse and (cached := read_cached_artifact(job)) is not None:
        if progress is not None:
            progress(f"reusing {method.name} artifact at {job.output_dir}")
        return cached

    # Picking a method is picking a toolchain, so the spec need not name both.
    backend_name = spec.backend or method.compression_backend
    backend = resolve_compression_backend(
        backend_name,
        python_executable=spec.python_executable,
        llama_cpp_dir=spec.llama_cpp_dir,
        wrapper=spec.llama_cpp_wrapper,
        path_translator=wsl_path if spec.llama_cpp_wrapper else None,
    )

    usable, detail = backend.available()
    if not usable:
        raise RuntimeError(f"compression backend {backend_name!r} unavailable: {detail}")

    artifact = backend.compress(job, progress=progress)
    write_artifact_sidecar(job, artifact)
    return artifact


__all__ = [
    "ARTIFACT_SIDECAR",
    "artifact_dir",
    "build_job",
    "read_cached_artifact",
    "run_compression",
    "write_artifact_sidecar",
]
