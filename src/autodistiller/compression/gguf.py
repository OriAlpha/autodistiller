"""GGUF production through llama.cpp's own tooling.

The same bargain as the llmcompressor adapter: AutoDistiller owns the
vocabulary, the calibration data and the record of what was produced; llama.cpp
owns the format and the quantizer. Nothing here writes a GGUF byte.

Two steps, because llama.cpp works in two:

1. ``convert_hf_to_gguf.py`` turns a Hugging Face directory into an unquantized
   GGUF. This is a Python script from a llama.cpp checkout rather than an
   installed binary, and it carries thousands of lines of per-architecture
   tensor mapping -- exactly the kind of thing the roadmap says to compose
   rather than reimplement.
2. ``llama-quantize`` turns that into the target type.

The intermediate is deleted afterwards. It is the model at 16-bit, so keeping it
alongside the quantized result roughly triples what the artifact costs on disk
for something no one will serve.

Discovery is explicit and its failures are legible. llama.cpp is not pip
installable, so a machine either has a checkout and built binaries or it does
not, and "not" should say which piece is missing rather than failing inside a
subprocess.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import time
from pathlib import Path

from ..results import CompressionArtifact
from .backend import CompressionBackend, CompressionError, CompressionJob, ProgressFn

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT_S = 3600

CONVERT_SCRIPT = "convert_hf_to_gguf.py"
QUANTIZE_BINARY = "llama-quantize"
ARTIFACT_NAME = "model.gguf"
INTERMEDIATE_NAME = "model-f16.gguf"

CONVERTER_REQUIREMENTS = (
    "--with",
    "gguf",
    "--with",
    "sentencepiece",
    "--with",
    "transformers",
    "--with",
    "torch",
    "--with",
    "numpy",
)
"""What ``convert_hf_to_gguf.py`` imports.

Mirrors llama.cpp's own ``requirements-convert_hf_to_gguf.txt``. Torch here is
whatever PyPI serves by default, deliberately: the converter reads tensors on
the CPU and never touches a GPU, so pulling the CUDA build would cost gigabytes
for nothing.
"""

LLAMA_CPP_DIR_ENV = "LLAMA_CPP_DIR"
"""Where a llama.cpp checkout lives, when it is not on PATH.

The converter is a script inside the repository, not something that gets
installed, so a path is the only way to find it.
"""

SEARCH_DIRS = (
    "~/llama.cpp",
    "~/src/llama.cpp",
    "/opt/llama.cpp",
    "/usr/local/share/llama.cpp",
)


def _candidate_roots(explicit: str | None = None) -> list[Path]:
    roots: list[Path] = []
    if explicit:
        roots.append(Path(explicit).expanduser())
    if env := os.environ.get(LLAMA_CPP_DIR_ENV):
        roots.append(Path(env).expanduser())
    roots.extend(Path(d).expanduser() for d in SEARCH_DIRS)
    return roots


def find_convert_script(llama_cpp_dir: str | None = None) -> Path | None:
    """Locate ``convert_hf_to_gguf.py`` in a llama.cpp checkout."""
    for root in _candidate_roots(llama_cpp_dir):
        script = root / CONVERT_SCRIPT
        if script.is_file():
            return script
    return None


def find_quantize_binary(llama_cpp_dir: str | None = None) -> Path | None:
    """Locate ``llama-quantize``, on PATH or in a checkout's build tree."""
    if found := shutil.which(QUANTIZE_BINARY):
        return Path(found)

    for root in _candidate_roots(llama_cpp_dir):
        for relative in ("build/bin", "build", "bin", "."):
            for name in (QUANTIZE_BINARY, f"{QUANTIZE_BINARY}.exe"):
                candidate = root / relative / name
                if candidate.is_file():
                    return candidate
    return None


class LlamaCppBackend(CompressionBackend):
    """Produces GGUF artifacts by driving llama.cpp's converter and quantizer."""

    name = "llama.cpp"

    def __init__(
        self,
        *,
        llama_cpp_dir: str | None = None,
        python_executable: str | None = None,
        timeout_s: int = DEFAULT_TIMEOUT_S,
    ) -> None:
        self.llama_cpp_dir = llama_cpp_dir
        self.python_executable = python_executable
        self.timeout_s = timeout_s

    def available(self) -> tuple[bool, str]:
        script = find_convert_script(self.llama_cpp_dir)
        binary = find_quantize_binary(self.llama_cpp_dir)

        missing = []
        if script is None:
            missing.append(f"{CONVERT_SCRIPT} (set {LLAMA_CPP_DIR_ENV} to a llama.cpp checkout)")
        if binary is None:
            missing.append(f"{QUANTIZE_BINARY} (build llama.cpp, or put it on PATH)")

        if missing:
            return False, "missing " + "; ".join(missing)
        return True, f"{script}, {binary}"

    def _converter_command(self) -> list[str]:
        """How to invoke Python for the converter script.

        Its dependencies are llama.cpp's, not AutoDistiller's: ``gguf`` is not
        installed here at all, so running it with this interpreter fails on the
        first import. ``uv run --with`` builds and caches a throwaway
        environment, the same trick the llmcompressor adapter uses, and
        ``--converter-python`` points at a prepared interpreter instead.
        """
        if self.python_executable:
            return [self.python_executable]

        uv = shutil.which("uv")
        if uv is None:
            raise CompressionError(
                "uv is not on PATH, so the GGUF converter has nowhere to get its "
                "dependencies. Install uv, or pass --converter-python pointing at an "
                "interpreter that already has gguf and sentencepiece."
            )
        return [uv, "run", "--quiet", *CONVERTER_REQUIREMENTS, "python"]

    def _run(self, command: list[str], *, step: str, job: CompressionJob) -> None:
        logger.debug("%s: %s", step, " ".join(command))
        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=self.timeout_s,
            )
        except subprocess.TimeoutExpired as exc:
            raise CompressionError(
                f"{job.method.name}: {step} timed out after {self.timeout_s}s"
            ) from exc
        except OSError as exc:
            raise CompressionError(f"{job.method.name}: could not start {step}: {exc}") from exc

        if completed.returncode != 0:
            # llama.cpp's own diagnostics are far more useful than ours.
            tail = (completed.stderr or completed.stdout or "").strip()[-1200:]
            raise CompressionError(
                f"{job.method.name}: {step} failed (exit {completed.returncode})\n{tail}"
            )

    def compress(
        self, job: CompressionJob, *, progress: ProgressFn | None = None
    ) -> CompressionArtifact:
        script = find_convert_script(self.llama_cpp_dir)
        binary = find_quantize_binary(self.llama_cpp_dir)
        if script is None or binary is None:
            raise CompressionError(f"llama.cpp tooling unavailable: {self.available()[1]}")

        job.output_dir.mkdir(parents=True, exist_ok=True)
        intermediate = job.output_dir / INTERMEDIATE_NAME
        artifact_path = job.output_dir / ARTIFACT_NAME

        started = time.perf_counter()

        if progress is not None:
            progress(f"{self.name}: converting {job.model_id} to GGUF")

        self._run(
            [
                *self._converter_command(),
                str(script),
                job.model_id,
                "--outfile",
                str(intermediate),
                "--outtype",
                "f16",
            ],
            step="convert_hf_to_gguf",
            job=job,
        )

        if progress is not None:
            progress(f"{self.name}: quantizing to {job.method.scheme}")

        try:
            self._run(
                [str(binary), str(intermediate), str(artifact_path), job.method.scheme],
                step="llama-quantize",
                job=job,
            )
        finally:
            # The f16 intermediate is the whole model again. Keeping it would
            # cost more disk than the artifact anyone actually serves, and it is
            # reproducible from the source model in one command.
            intermediate.unlink(missing_ok=True)

        if not artifact_path.is_file():
            raise CompressionError(
                f"{job.method.name}: llama-quantize reported success but wrote no "
                f"{artifact_path.name}"
            )

        return CompressionArtifact(
            recipe=job.recipe(),
            backend=self.name,
            source_model=job.model_id,
            output_dir=str(job.output_dir),
            artifact_bytes=artifact_path.stat().st_size,
            duration_s=time.perf_counter() - started,
            versions=_versions(binary),
        )


def _versions(binary: Path) -> dict[str, str]:
    """Whatever llama.cpp will tell us about itself. Best effort."""
    try:
        completed = subprocess.run(
            [str(binary), "--help"], capture_output=True, text=True, timeout=30
        )
        for line in (completed.stdout + completed.stderr).splitlines():
            if "version" in line.lower():
                return {"llama.cpp": line.strip()[:120]}
    except (OSError, subprocess.SubprocessError):
        pass
    return {}


__all__ = [
    "ARTIFACT_NAME",
    "CONVERTER_REQUIREMENTS",
    "CONVERT_SCRIPT",
    "LLAMA_CPP_DIR_ENV",
    "QUANTIZE_BINARY",
    "LlamaCppBackend",
    "find_convert_script",
    "find_quantize_binary",
]
