"""Compression backends.

An adapter's job is translation and isolation, not compression. AutoDistiller
owns the vocabulary (:mod:`.methods`), the calibration data and the record of
what was produced; the backend owns the kernels and the algorithm.

llmcompressor covers the whole Phase 3 method list -- INT8, INT4 via GPTQ or
AWQ, and FP8 -- so one adapter is enough. A second backend would implement
:class:`CompressionBackend` and map onto the same method names.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from ..metadata.hashing import hash_text_stream
from ..results import CompressionArtifact, CompressionRecipe
from .methods import CompressionMethod

logger = logging.getLogger(__name__)

RUNNER = Path(__file__).with_name("_runner.py")
DEFAULT_TIMEOUT_S = 3600
DEFAULT_TORCH_INDEX = "https://download.pytorch.org/whl/cu128"
"""CUDA wheels for the isolated environment. See ``_command`` for why."""

ProgressFn = Callable[[str], None]


class CompressionError(RuntimeError):
    """The backend failed to produce an artifact."""


@dataclass
class CompressionJob:
    """Everything needed to produce one compressed artifact."""

    model_id: str
    method: CompressionMethod
    output_dir: Path
    calibration_texts: list[str]
    num_calibration_samples: int = 128
    max_seq_length: int = 2048
    ignore: tuple[str, ...] = ("lm_head",)
    trust_remote_code: bool = False
    dtype: str = "auto"

    def recipe(self) -> CompressionRecipe:
        """The reproducible description of what was asked for."""
        return CompressionRecipe(
            method=self.method.name,
            scheme=self.method.scheme,
            algorithm=self.method.algorithm,
            weight_bits=self.method.weight_bits,
            activation_bits=self.method.activation_bits,
            ignore=list(self.ignore),
            needs_calibration=self.method.needs_calibration,
            n_calibration_samples=(
                min(self.num_calibration_samples, len(self.calibration_texts))
                if self.calibration_texts
                else 0
            ),
            max_seq_length=self.max_seq_length,
            # Calibration data changes the produced weights, so it belongs in
            # the recipe identity as much as the algorithm does.
            calibration_fingerprint=(
                hash_text_stream(self.calibration_texts) if self.calibration_texts else None
            ),
        )

    def to_payload(self) -> dict:
        return {
            "model_id": self.model_id,
            "output_dir": str(self.output_dir),
            "scheme": self.method.scheme,
            "algorithm": self.method.algorithm,
            "ignore": list(self.ignore),
            "calibration_texts": self.calibration_texts,
            "num_calibration_samples": self.num_calibration_samples,
            "max_seq_length": self.max_seq_length,
            "trust_remote_code": self.trust_remote_code,
            "dtype": self.dtype,
        }


class CompressionBackend:
    """Produces a compressed artifact from a job."""

    name = "base"

    def available(self) -> tuple[bool, str]:
        raise NotImplementedError

    def compress(self, job: CompressionJob, *, progress: ProgressFn | None = None):
        raise NotImplementedError


class LLMCompressorBackend(CompressionBackend):
    """Drives llmcompressor in an isolated environment.

    ``uv run --with`` builds and caches that environment, so there is no venv
    lifecycle to manage here. Point ``python_executable`` at an existing
    interpreter to reuse a prepared environment instead -- useful when the
    compression environment lives somewhere else entirely, such as WSL.
    """

    name = "llmcompressor"

    def __init__(
        self,
        *,
        python_executable: str | None = None,
        requirement: str = "llmcompressor",
        python_version: str = "3.12",
        torch_index: str | None = DEFAULT_TORCH_INDEX,
        timeout_s: int = DEFAULT_TIMEOUT_S,
    ) -> None:
        self.python_executable = python_executable
        self.requirement = requirement
        self.python_version = python_version
        self.torch_index = torch_index
        self.timeout_s = timeout_s

    def _command(self) -> list[str]:
        if self.python_executable:
            return [self.python_executable, str(RUNNER)]

        uv = shutil.which("uv")
        if uv is None:
            raise CompressionError(
                "uv is not on PATH. Install uv, or pass --compress-python pointing at "
                "an interpreter that already has llmcompressor."
            )
        command = [uv, "run", "--quiet", "--python", self.python_version]
        if self.torch_index:
            # Without this the ephemeral environment resolves a CPU-only torch:
            # `uv run` has no --torch-backend flag and does not read
            # UV_TORCH_BACKEND. Quantization then runs on CPU, slowly, and AWQ
            # fails outright because it requires an accelerator.
            #
            # The index strategy matters as much as the index. uv defaults to
            # first-match, which would take every package from the torch mirror
            # and pin transformers back to 4.x.
            command += ["--index", self.torch_index, "--index-strategy", "unsafe-best-match"]
        command += ["--with", self.requirement, "python", str(RUNNER)]
        return command

    def available(self) -> tuple[bool, str]:
        if self.python_executable:
            return (
                Path(self.python_executable).exists(),
                f"interpreter {self.python_executable}",
            )
        if shutil.which("uv") is None:
            return False, "uv not found on PATH"
        return True, f"uv run --with {self.requirement}"

    def compress(
        self, job: CompressionJob, *, progress: ProgressFn | None = None
    ) -> CompressionArtifact:
        job.output_dir.mkdir(parents=True, exist_ok=True)
        command = self._command()

        env = dict(os.environ)

        if progress is not None:
            progress(f"{self.name}: {job.method.name} ({job.method.scheme}) -> {job.output_dir}")

        started = time.perf_counter()
        try:
            completed = subprocess.run(
                command,
                input=json.dumps(job.to_payload()),
                capture_output=True,
                text=True,
                timeout=self.timeout_s,
                env=env,
            )
        except subprocess.TimeoutExpired as exc:
            raise CompressionError(f"{job.method.name} timed out after {self.timeout_s}s") from exc
        except OSError as exc:
            raise CompressionError(f"could not start the compression backend: {exc}") from exc

        duration = time.perf_counter() - started
        result = self._parse(completed, job)

        if not result.get("ok"):
            # The backend's own traceback is far more useful than ours.
            detail = result.get("error", "unknown failure")
            logger.debug("compression stderr:\n%s", completed.stderr[-4000:])
            raise CompressionError(f"{job.method.name} failed: {detail}")

        return CompressionArtifact(
            recipe=job.recipe(),
            backend=self.name,
            source_model=job.model_id,
            output_dir=str(job.output_dir),
            artifact_bytes=result.get("artifact_bytes"),
            duration_s=result.get("duration_s", duration),
            versions=result.get("versions", {}),
        )

    @staticmethod
    def _parse(completed: subprocess.CompletedProcess, job: CompressionJob) -> dict:
        """Pull the JSON result out of stdout.

        The runner writes only JSON to stdout, but a dependency can still print
        there. Taking the last JSON object tolerates that without hiding it.
        """
        for line in reversed(completed.stdout.strip().splitlines()):
            line = line.strip()
            if line.startswith("{"):
                try:
                    return json.loads(line)
                except json.JSONDecodeError:
                    continue

        tail = (completed.stderr or completed.stdout or "").strip()[-800:]
        raise CompressionError(
            f"{job.method.name}: backend produced no result (exit {completed.returncode}).\n{tail}"
        )


COMPRESSION_BACKENDS: dict[str, type[CompressionBackend]] = {
    "llmcompressor": LLMCompressorBackend,
}


def resolve_compression_backend(name: str, **kwargs) -> CompressionBackend:
    try:
        factory = COMPRESSION_BACKENDS[name.lower()]
    except KeyError:
        raise KeyError(
            f"unknown compression backend {name!r}; "
            f"available: {', '.join(sorted(COMPRESSION_BACKENDS))}"
        ) from None
    return factory(**kwargs)


def default_python() -> str:
    """The interpreter running AutoDistiller, for callers that want to opt out
    of environment isolation."""
    return sys.executable


__all__ = [
    "COMPRESSION_BACKENDS",
    "CompressionBackend",
    "CompressionError",
    "CompressionJob",
    "LLMCompressorBackend",
    "ProgressFn",
    "resolve_compression_backend",
]
