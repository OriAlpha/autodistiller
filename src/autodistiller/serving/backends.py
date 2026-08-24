"""Deployment backends.

AutoDistiller measures a server it did not start. That is a deliberate
simplification: serving runtimes have heavy, pinned environments (vLLM does not
install on Windows at all, and pins its own torch), they take a while to become
ready, and owning their lifecycle buys a class of bugs -- zombie processes, port
races, startup timeouts -- for no measurement benefit.

So a backend here is mostly *knowledge*: how you would launch it, and which
protocol extensions it honours. The benchmark client itself is backend-agnostic
because vLLM and llama.cpp both speak the OpenAI API.

Process management lives in :mod:`.launcher` rather than here. Phase 2 deferred
it on the grounds that printing the command was enough; benchmarking a dozen
candidates in sequence is the condition under which it stopped being enough.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

DEFAULT_PORT = 8000


@dataclass(frozen=True)
class Backend:
    name: str
    description: str
    default_port: int = DEFAULT_PORT
    # vLLM honours `ignore_eos`, which pins every request to exactly max_tokens
    # and keeps decode measurements comparable. Servers without it still work;
    # the reported token counts stay honest either way.
    supports_ignore_eos: bool = True
    supports_concurrency: bool = True
    artifact_shape: str = "directory"
    """What this runtime is pointed at: a Hugging Face ``directory``, or a
    single GGUF ``file``."""

    kv_dtypes: tuple[str, ...] = ("auto", "fp8")
    """KV cache types worth searching over. Backend-specific: llama.cpp has its
    own quantized cache vocabulary and no fp8, so offering fp8 there would
    generate candidates it cannot serve."""

    kv_flag_template: str = "--kv-cache-dtype {kv_dtype}"
    """How this runtime is told to use a non-default KV cache type."""

    notes: str = ""

    def model_path(self, artifact_dir: str) -> str:
        """The path to hand this runtime for an artifact AutoDistiller produced.

        vLLM takes the directory. llama-server takes the ``.gguf`` inside it, so
        the same artifact is named differently depending on who is serving it --
        which is the backend-specific semantics the roadmap asks to preserve
        rather than paper over.
        """
        return artifact_dir

    def launch_command(
        self,
        model: str,
        *,
        port: int | None = None,
        max_model_len: int | None = None,
        gpu_memory_utilization: float | None = None,
    ) -> str:
        raise NotImplementedError


@dataclass(frozen=True)
class VLLMBackend(Backend):
    def launch_command(
        self,
        model: str,
        *,
        port: int | None = None,
        max_model_len: int | None = None,
        gpu_memory_utilization: float | None = None,
    ) -> str:
        parts = ["vllm serve", model, f"--port {port or self.default_port}"]
        if max_model_len:
            parts.append(f"--max-model-len {max_model_len}")
        if gpu_memory_utilization:
            parts.append(f"--gpu-memory-utilization {gpu_memory_utilization}")
        return " ".join(parts)


@dataclass(frozen=True)
class LlamaCppBackend(Backend):
    """llama.cpp's llama-server, driven through the same OpenAI-compatible client."""

    def model_path(self, artifact_dir: str) -> str:
        directory = Path(artifact_dir)
        if directory.is_file():
            return artifact_dir  # already a .gguf
        found = sorted(directory.glob("*.gguf")) if directory.is_dir() else []
        return str(found[0]) if found else artifact_dir

    def launch_command(
        self,
        model: str,
        *,
        port: int | None = None,
        max_model_len: int | None = None,
        gpu_memory_utilization: float | None = None,
    ) -> str:
        parts = ["llama-server", f"-m {model}", f"--port {port or self.default_port}"]
        if max_model_len:
            parts.append(f"-c {max_model_len}")
        return " ".join(parts)


BACKENDS: dict[str, Backend] = {
    "vllm": VLLMBackend(
        name="vllm",
        description="vLLM OpenAI-compatible server",
        notes="Linux/WSL only. On an 8 GiB card, lower --gpu-memory-utilization "
        "if the KV cache will not fit.",
    ),
    "llama.cpp": LlamaCppBackend(
        name="llama.cpp",
        description="llama.cpp llama-server (GGUF)",
        default_port=8080,
        supports_ignore_eos=False,
        artifact_shape="file",
        kv_dtypes=("auto",),
        kv_flag_template="--cache-type-k {kv_dtype} --cache-type-v {kv_dtype}",
        notes="Serves GGUF. Runs on CPU as well as CUDA, and needs no ignore_eos: "
        "token counts stay honest without it, but decode timings vary more.",
    ),
}


def resolve_backend(name: str) -> Backend:
    try:
        return BACKENDS[name.lower()]
    except KeyError:
        raise KeyError(
            f"unknown backend {name!r}; available: {', '.join(sorted(BACKENDS))}"
        ) from None


__all__ = ["BACKENDS", "Backend", "resolve_backend"]
