"""Turning a compression request into an artifact.

Sits between the CLI and the backend adapter: resolves the method, checks it
against the hardware and the serving backend, gathers calibration text through
the same loader Phase 1 evaluates with, and hands a fully-specified job down.

Calibration data comes from AutoDistiller rather than being named for the
backend to fetch, so it is fingerprinted with the same machinery as evaluation
data and two runs can be shown to have used the same bytes.
"""

from __future__ import annotations

import re
from pathlib import Path

from ..config import CompressionSpec, ModelSpec
from ..evaluation.datasets import load_text_corpus
from ..metadata.profiles import GPUProfile
from ..results import CompressionArtifact
from .backend import CompressionJob, ProgressFn, resolve_compression_backend
from .methods import check_method, resolve_method

_SLUG = re.compile(r"[^a-zA-Z0-9._-]+")


def artifact_dir(model_id: str, method: str, root: Path) -> Path:
    """Where an artifact lands by default: ``<root>/<model>-<method>``."""
    slug = _SLUG.sub("-", model_id.split("/")[-1]).strip("-") or "model"
    return root / f"{slug}-{method}"


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

    output_dir = spec.output_dir or artifact_dir(model.id, method.name, output_root)

    return CompressionJob(
        model_id=model.id,
        method=method,
        output_dir=Path(output_dir),
        calibration_texts=calibration_texts,
        num_calibration_samples=spec.num_calibration_samples,
        max_seq_length=spec.max_seq_length,
        ignore=tuple(spec.ignore),
        trust_remote_code=model.trust_remote_code,
        dtype=model.dtype if model.dtype != "auto" else "auto",
    )


def run_compression(
    model: ModelSpec,
    spec: CompressionSpec,
    *,
    output_root: Path = Path("artifacts"),
    profile: GPUProfile | None = None,
    serving_backend: str | None = None,
    progress: ProgressFn | None = None,
) -> CompressionArtifact:
    """Produce one compressed artifact.

    ``profile`` and ``serving_backend`` are checked before any work starts:
    producing an artifact the GPU cannot run, or that the target runtime cannot
    serve, wastes minutes to reach a useless result.
    """
    method = resolve_method(spec.method)
    availability = check_method(method, profile=profile, backend=serving_backend)
    if not availability.available:
        raise ValueError(f"{method.name} is not usable here: {'; '.join(availability.reasons)}")

    job = build_job(model, spec, output_root=output_root)
    backend = resolve_compression_backend(spec.backend, python_executable=spec.python_executable)

    usable, detail = backend.available()
    if not usable:
        raise RuntimeError(f"compression backend {spec.backend!r} unavailable: {detail}")

    return backend.compress(job, progress=progress)


__all__ = ["artifact_dir", "build_job", "run_compression"]
