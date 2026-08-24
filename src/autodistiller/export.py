"""Export and reproducibility.

A measured recommendation is only worth something if someone else can deploy it
and rebuild it. Those are different requirements and this module serves both:

* **Deployable.** The compressed artifact is already a Hugging Face directory --
  llmcompressor writes ``config.json``, safetensors and the tokenizer -- so
  exporting does not convert anything. What it does is *check* the claim rather
  than assert it: the weights are there, the tokenizer is there, and the
  quantization format is one the target runtime actually has a kernel for. An
  artifact that benchmarks beautifully and cannot be served is the failure this
  is here to catch.
* **Reproducible.** The manifest carries the exact recipe, the calibration
  fingerprint, the measurements, and the hardware and software stack they were
  taken on -- plus the commands that would rebuild and re-measure it.

Nothing here re-runs a measurement. Everything in a manifest was recorded by an
earlier phase; export assembles it, verifies it, and writes it next to the
weights so the directory explains itself.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from . import __version__
from .compression.pipeline import ARTIFACT_SIDECAR
from .metadata.environment import EnvironmentInfo
from .metadata.hardware import HardwareInfo
from .results import (
    CompressionArtifact,
    DeploymentBenchmark,
    ModelInfo,
    RunRecord,
    TaskResult,
)
from .serving.backends import resolve_backend

MANIFEST_FILENAME = "autodistiller-manifest.json"
README_FILENAME = "DEPLOY.md"
CONFIG_FILENAME = "autodistiller-config.yaml"

MANIFEST_SCHEMA_VERSION = 1

WEIGHT_PATTERNS = ("*.safetensors", "*.bin", "*.pt")
TOKENIZER_FILES = ("tokenizer.json", "tokenizer.model", "tokenizer_config.json", "vocab.json")

SERVABLE_QUANT_METHODS = {
    "vllm": {
        "compressed-tensors",
        "fp8",
        "awq",
        "awq_marlin",
        "gptq",
        "gptq_marlin",
    },
}
"""Quantization formats a runtime has kernels for.

Keyed by the value of ``quantization_config.quant_method`` in ``config.json``,
which is what the runtime itself dispatches on. Everything AutoDistiller
produces through llmcompressor lands on ``compressed-tensors``; the rest are
here because a user can point ``export`` at weights they got elsewhere.
"""


@dataclass(frozen=True)
class Check:
    """One deployability question, and the answer."""

    name: str
    ok: bool
    detail: str

    def describe(self) -> str:
        return f"{'ok' if self.ok else 'FAIL'}  {self.name}: {self.detail}"


def _directory_size(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def _weight_files(directory: Path) -> list[Path]:
    found: list[Path] = []
    for pattern in WEIGHT_PATTERNS:
        found.extend(sorted(directory.glob(pattern)))
    return found


def read_quant_method(directory: Path) -> str | None:
    """The format string the serving runtime will dispatch on, if any.

    ``None`` means the weights are not quantized, which is not a problem: an
    uncompressed baseline is servable by everything.
    """
    config_path = directory / "config.json"
    if not config_path.is_file():
        return None
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    quantization = config.get("quantization_config")
    return quantization.get("quant_method") if isinstance(quantization, dict) else None


def read_artifact_sidecar(directory: Path) -> CompressionArtifact | None:
    """The recipe AutoDistiller recorded when it produced these weights.

    A run record only carries its compression artifact when the optimizer put it
    there. Weights produced by ``compress`` directly have no run record at all,
    and an evaluation of a compressed candidate records the artifact directory
    as its model id rather than the recipe that built it. The sidecar sits beside
    the weights in every one of those cases, so reading it is what makes export
    work on an artifact however it was produced.
    """
    path = Path(directory) / ARTIFACT_SIDECAR
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload.pop("artifact_key", None)
        return CompressionArtifact.model_validate(payload)
    except (json.JSONDecodeError, OSError, ValueError):
        return None


def inspect_artifact(directory: Path, *, backend: str = "vllm") -> list[Check]:
    """Whether this directory can actually be loaded and served.

    Checked rather than assumed. Each of these has a failure mode that only
    shows up at serve time, minutes later, with an error that names none of it.
    """
    directory = Path(directory)
    checks: list[Check] = []

    if not directory.is_dir():
        return [Check("directory", False, f"{directory} does not exist")]

    has_config = (directory / "config.json").is_file()
    checks.append(
        Check("config", has_config, "config.json present" if has_config else "config.json missing")
    )

    weights = _weight_files(directory)
    total = sum(f.stat().st_size for f in weights)
    checks.append(
        Check(
            "weights",
            bool(weights),
            f"{len(weights)} file(s), {total / 1024**3:.2f} GiB"
            if weights
            else f"no weights matching {', '.join(WEIGHT_PATTERNS)}",
        )
    )

    tokenizer = [name for name in TOKENIZER_FILES if (directory / name).is_file()]
    checks.append(
        Check(
            "tokenizer",
            bool(tokenizer),
            ", ".join(tokenizer) if tokenizer else "no tokenizer files; the server cannot encode",
        )
    )

    method = read_quant_method(directory)
    servable = SERVABLE_QUANT_METHODS.get(backend, set())
    if method is None:
        checks.append(Check("format", True, f"unquantized; {backend} serves it as-is"))
    elif method in servable:
        checks.append(Check("format", True, f"{method}, which {backend} has kernels for"))
    else:
        checks.append(
            Check(
                "format",
                False,
                f"{method} is not in {backend}'s known formats "
                f"({', '.join(sorted(servable)) or 'none recorded'})",
            )
        )

    return checks


def gguf_note(quant_method: str | None) -> str:
    """Whether a GGUF conversion applies here, and what to run if it does.

    The roadmap asks for GGUF export *where applicable*, and the honest answer
    for a compressed-tensors artifact is that it is not: GGUF carries its own
    quantization schemes, and llama.cpp converts from unquantized Hugging Face
    weights rather than from someone else's quantized ones. Converting the base
    model and quantizing inside llama.cpp is the supported path, so that is the
    command reported. The conversion itself belongs to Phase 9, where llama.cpp
    becomes a measured backend rather than a suggestion.
    """
    if quant_method is not None:
        return (
            f"Not applicable: these weights are {quant_method}, and llama.cpp converts from "
            f"unquantized Hugging Face weights. Convert the source model instead, then "
            f"quantize with llama-quantize."
        )
    return (
        "python convert_hf_to_gguf.py <model> --outfile model.gguf  "
        "# from a llama.cpp checkout, then: llama-quantize model.gguf model-q4.gguf Q4_K_M"
    )


class ExportManifest(BaseModel):
    """Everything needed to redeploy or rebuild one result.

    Composed from records the earlier phases already wrote, so a manifest never
    disagrees with the run it came from.
    """

    model_config = ConfigDict(protected_namespaces=())

    schema_version: int = MANIFEST_SCHEMA_VERSION
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    autodistiller_version: str = __version__

    run_id: str
    candidate_id: str | None = None
    backend: str = "vllm"

    model: ModelInfo
    source_model: str
    """The model the artifact was built from.

    Distinct from ``model.id``: a compressed candidate is evaluated out of its
    artifact directory, so the record calls that directory its model. A reader
    needs to know these weights came from Qwen/Qwen3-0.6B, not from a path.
    """

    artifact: CompressionArtifact | None = None
    artifact_dir: str | None = None
    artifact_bytes: int | None = None

    tasks: list[TaskResult] = Field(default_factory=list)
    quality_retention: float | None = None
    deployment: DeploymentBenchmark | None = None

    hardware: HardwareInfo
    environment: EnvironmentInfo

    serve_command: str
    reproduce: list[str] = Field(default_factory=list)
    gguf: str = ""
    checks: list[dict] = Field(default_factory=list)

    @property
    def deployable(self) -> bool:
        return all(check.get("ok") for check in self.checks)

    @property
    def served_path(self) -> str:
        return self.artifact_dir or self.model.id

    def to_json(self) -> str:
        return json.dumps(self.model_dump(mode="json"), indent=2, ensure_ascii=False)

    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json(), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: Path | str) -> ExportManifest:
        return cls.model_validate_json(Path(path).read_text(encoding="utf-8"))


def _reproduce_commands(record: RunRecord, artifact: CompressionArtifact | None) -> list[str]:
    """The commands that rebuild this result from the source model.

    Compression first, because the evaluation is of the artifact it produces.
    Both are exact: the recipe carries the calibration fingerprint, and the
    saved config carries every evaluation setting.
    """
    commands: list[str] = []

    if artifact is not None:
        recipe = artifact.recipe
        parts = [
            "autodistiller compress",
            f"--model {artifact.source_model}",
            f"--method {recipe.method}",
        ]
        if recipe.needs_calibration:
            parts.append(f"--samples {recipe.n_calibration_samples}")
            parts.append(f"--max-seq-length {recipe.max_seq_length}")
            parts.append("--calibration <the corpus fingerprinted below>")
        commands.append(" ".join(parts))

    commands.append(f"autodistiller evaluate --config {CONFIG_FILENAME}")
    return commands


def build_manifest(
    record: RunRecord,
    *,
    backend: str = "vllm",
    quality_retention: float | None = None,
    artifact_dir: Path | str | None = None,
) -> ExportManifest:
    """Assemble a manifest from a stored run.

    ``artifact_dir`` defaults to whatever the run's compression artifact points
    at. For a baseline there is none, and the manifest describes the source
    model instead -- "do not compress" is a deployable answer too.
    """
    artifact = record.compression
    directory = Path(artifact_dir) if artifact_dir else None
    if directory is None and artifact is not None:
        directory = Path(artifact.output_dir)
    if directory is None and Path(record.model.id).is_dir():
        # A compressed candidate is evaluated straight out of its artifact
        # directory, so the directory is what the record calls its model.
        directory = Path(record.model.id)

    if artifact is None and directory is not None:
        artifact = read_artifact_sidecar(directory)

    checks = inspect_artifact(directory, backend=backend) if directory else []
    size = _directory_size(directory) if directory and directory.is_dir() else None

    served = str(directory) if directory else record.model.id

    # The model's own context length, not the benchmark's max_tokens -- those
    # are output tokens per request, and serving with --max-model-len 128 would
    # truncate every prompt.
    max_model_len = record.model.context_length

    return ExportManifest(
        run_id=record.run_id,
        candidate_id=record.candidate_id,
        backend=backend,
        model=record.model,
        source_model=artifact.source_model if artifact else record.model.id,
        artifact=artifact,
        artifact_dir=str(directory) if directory else None,
        artifact_bytes=size,
        tasks=record.tasks,
        quality_retention=quality_retention,
        deployment=record.deployment,
        hardware=record.hardware,
        environment=record.environment,
        serve_command=resolve_backend(backend).launch_command(served, max_model_len=max_model_len),
        reproduce=_reproduce_commands(record, artifact),
        gguf=gguf_note(read_quant_method(directory) if directory else None),
        checks=[{"name": c.name, "ok": c.ok, "detail": c.detail} for c in checks],
    )


def render_readme(manifest: ExportManifest) -> str:
    """A human-readable deployment note to sit beside the weights."""
    recipe = manifest.artifact.recipe if manifest.artifact else None
    lines = [
        f"# {manifest.source_model}"
        + (f" - {recipe.label}" if recipe else " - uncompressed baseline"),
        "",
        f"Produced by AutoDistiller {manifest.autodistiller_version} "
        f"on {manifest.created_at:%Y-%m-%d}. Full provenance in `{MANIFEST_FILENAME}`.",
        "",
        "## Serve it",
        "",
        "```bash",
        manifest.serve_command,
        "```",
        "",
        "## What was measured",
        "",
    ]

    if manifest.quality_retention is not None:
        lines.append(
            f"- Quality retention: **{manifest.quality_retention * 100:.2f}%** of baseline"
        )
    for task in manifest.tasks:
        if (primary := task.primary_metric) is not None:
            lines.append(f"- {task.name}: {primary.name} = {primary.format()}")

    if (benchmark := manifest.deployment) is not None:
        peak = benchmark.best_throughput
        single = benchmark.single_stream
        if peak is not None:
            lines.append(
                f"- Peak throughput: {peak.output_tokens_per_s:.0f} tok/s "
                f"at concurrency {peak.concurrency} ({benchmark.backend})"
            )
        if single is not None and single.ttft is not None:
            lines.append(f"- Single-stream TTFT p50: {single.ttft.p50 * 1000:.0f}ms")
        if benchmark.peak_vram_bytes:
            lines.append(f"- Peak VRAM: {benchmark.peak_vram_bytes / 1024**3:.2f} GiB")
    else:
        lines.append("- No deployment benchmark was run for this result.")

    lines += [
        "",
        f"Measured on {manifest.hardware.describe()}, "
        f"torch {manifest.environment.torch_version}, "
        f"transformers {manifest.environment.packages.get('transformers', 'n/a')}. "
        "Numbers move with the stack; the manifest records it exactly.",
        "",
        "## Rebuild it",
        "",
        "```bash",
        *manifest.reproduce,
        "```",
        "",
        "## GGUF / llama.cpp",
        "",
        manifest.gguf,
        "",
        "## Deployability checks",
        "",
    ]
    lines += [
        f"- {'PASS' if check['ok'] else 'FAIL'} {check['name']}: {check['detail']}"
        for check in manifest.checks
    ] or ["- No artifact directory to check."]

    return "\n".join(lines) + "\n"


def export(
    record: RunRecord,
    *,
    backend: str = "vllm",
    quality_retention: float | None = None,
    artifact_dir: Path | str | None = None,
    output_dir: Path | str | None = None,
    copy_weights: bool = False,
) -> tuple[ExportManifest, Path]:
    """Write the manifest, the deployment note and the config beside the weights.

    With no ``output_dir`` the files land in the artifact directory itself, so
    the thing you would serve is the thing that explains itself and nothing is
    copied. Give an ``output_dir`` to assemble a standalone bundle; ``copy_weights``
    then brings the weights along rather than referring to them, which is the
    difference between a bundle you can move and a bundle you cannot.
    """
    manifest = build_manifest(
        record,
        backend=backend,
        quality_retention=quality_retention,
        artifact_dir=artifact_dir,
    )

    destination = Path(output_dir) if output_dir else None
    if destination is None:
        if manifest.artifact_dir is None:
            raise ValueError(
                "this result has no artifact directory to write into (it is the "
                "uncompressed baseline); pass an output directory instead"
            )
        destination = Path(manifest.artifact_dir)

    destination.mkdir(parents=True, exist_ok=True)

    if copy_weights and manifest.artifact_dir:
        source = Path(manifest.artifact_dir)
        if source.resolve() != destination.resolve():
            shutil.copytree(source, destination, dirs_exist_ok=True)
            # The bundle is now self-contained, so the manifest must describe
            # the copy rather than pointing back at where it came from.
            manifest.artifact_dir = str(destination)
            manifest.serve_command = resolve_backend(backend).launch_command(str(destination))

    manifest.save(destination / MANIFEST_FILENAME)
    (destination / README_FILENAME).write_text(render_readme(manifest), encoding="utf-8")
    record.config.save(destination / CONFIG_FILENAME)

    return manifest, destination


__all__ = [
    "CONFIG_FILENAME",
    "MANIFEST_FILENAME",
    "MANIFEST_SCHEMA_VERSION",
    "README_FILENAME",
    "SERVABLE_QUANT_METHODS",
    "Check",
    "ExportManifest",
    "build_manifest",
    "export",
    "gguf_note",
    "inspect_artifact",
    "read_artifact_sidecar",
    "read_quant_method",
    "render_readme",
]
