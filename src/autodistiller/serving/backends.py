"""Deployment backends.

AutoDistiller measures a server it did not start. That is a deliberate
simplification: serving runtimes have heavy, pinned environments (vLLM does not
install on Windows at all, and pins its own torch), they take a while to become
ready, and owning their lifecycle buys a class of bugs -- zombie processes, port
races, startup timeouts -- for no measurement benefit.

So a backend here is mostly *knowledge*: how you would launch it, and which
protocol extensions it honours. The benchmark client itself is backend-agnostic
because vLLM and llama.cpp both speak the OpenAI API.

ponytail: no process management; add a launcher only if printing the command
proves annoying in practice.
"""

from __future__ import annotations

from dataclasses import dataclass

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
    notes: str = ""

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
    """Phase 9. The client already works against it; only the launch differs."""

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
        notes="Phase 9. Benchmarked through the same OpenAI-compatible client.",
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
