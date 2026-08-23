"""Streaming client for an OpenAI-compatible serving endpoint.

AutoDistiller never imports vLLM. It talks to a running server over HTTP, which
buys three things: the measurement happens against the real deployment path a
user would ship, the serving runtime keeps its own (heavy, pinned) environment,
and llama.cpp's server in Phase 9 needs no new client because it speaks the same
protocol.

Timings are taken around the stream, so time-to-first-token is the real thing
the caller would observe, not a number derived from a batch API.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field

import httpx

DONE_SENTINEL = "[DONE]"


@dataclass
class RequestMetrics:
    """What one streamed request cost."""

    ok: bool
    total_s: float
    ttft_s: float | None = None
    n_prompt_tokens: int = 0
    n_output_tokens: int = 0
    error: str | None = None

    @property
    def decode_s(self) -> float | None:
        """Seconds spent generating after the first token arrived."""
        if self.ttft_s is None:
            return None
        return max(self.total_s - self.ttft_s, 0.0)

    @property
    def tpot_s(self) -> float | None:
        """Time per output token, excluding prefill.

        Measured across the tokens *after* the first, since the first token's
        cost is prefill and is reported separately as TTFT.
        """
        decode = self.decode_s
        if decode is None or self.n_output_tokens < 2:
            return None
        return decode / (self.n_output_tokens - 1)

    @property
    def output_tokens_per_s(self) -> float | None:
        decode = self.decode_s
        if not decode or self.n_output_tokens < 2:
            return None
        return (self.n_output_tokens - 1) / decode


@dataclass
class EndpointInfo:
    """What a server reports about itself."""

    url: str
    models: list[str] = field(default_factory=list)
    reachable: bool = False
    error: str | None = None

    @property
    def default_model(self) -> str | None:
        return self.models[0] if self.models else None


async def probe_endpoint(
    url: str,
    *,
    timeout: float = 5.0,
    transport: httpx.AsyncBaseTransport | None = None,
) -> EndpointInfo:
    """Ask a server which models it serves. Used to fail fast and to default
    ``--served-model`` when the caller does not name one.

    ``transport`` is the injection point that lets the benchmark be tested
    against a fake server, with no GPU and no runtime installed.
    """
    info = EndpointInfo(url=url)
    try:
        async with httpx.AsyncClient(timeout=timeout, transport=transport) as client:
            response = await client.get(f"{url.rstrip('/')}/v1/models")
            response.raise_for_status()
            payload = response.json()
        info.models = [entry["id"] for entry in payload.get("data", [])]
        info.reachable = True
    except Exception as exc:
        info.error = f"{type(exc).__name__}: {exc}"
    return info


def _parse_sse_line(line: str) -> dict | None:
    """Decode one Server-Sent Events line, or None when it carries no payload."""
    if not line.startswith("data:"):
        return None
    data = line[len("data:") :].strip()
    if not data or data == DONE_SENTINEL:
        return None
    try:
        return json.loads(data)
    except json.JSONDecodeError:
        return None


def _chunk_text(chunk: dict) -> str:
    """Pull generated text out of either completions or chat-completions shape."""
    choices = chunk.get("choices") or []
    if not choices:
        return ""
    choice = choices[0]
    if "text" in choice:  # /v1/completions
        return choice["text"] or ""
    return (choice.get("delta") or {}).get("content") or ""  # /v1/chat/completions


async def stream_request(
    client: httpx.AsyncClient,
    *,
    url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    use_chat: bool = False,
    ignore_eos: bool = True,
    timeout: float = 300.0,
) -> RequestMetrics:
    """Issue one streaming request and time it.

    ``ignore_eos`` makes every request emit exactly ``max_tokens``, which keeps
    decode measurements comparable instead of varying with how chatty the model
    feels. It is a vLLM extension; servers that do not know the field ignore it,
    and the token counts reported below stay honest either way.
    """
    base = url.rstrip("/")
    if use_chat:
        endpoint = f"{base}/v1/chat/completions"
        body: dict = {"model": model, "messages": [{"role": "user", "content": prompt}]}
    else:
        endpoint = f"{base}/v1/completions"
        body = {"model": model, "prompt": prompt}

    body.update(
        {
            "max_tokens": max_tokens,
            "temperature": 0.0,  # greedy: the benchmark must not vary run to run
            "stream": True,
            # Ask for a final usage chunk so token counts come from the server
            # rather than from counting stream chunks.
            "stream_options": {"include_usage": True},
        }
    )
    if ignore_eos:
        body["ignore_eos"] = True

    started = time.perf_counter()
    ttft: float | None = None
    n_chunks = 0
    n_prompt_tokens = 0
    n_output_tokens = 0

    try:
        async with client.stream("POST", endpoint, json=body, timeout=timeout) as response:
            if response.status_code != 200:
                detail = (await response.aread()).decode(errors="replace")[:200]
                return RequestMetrics(
                    ok=False,
                    total_s=time.perf_counter() - started,
                    error=f"HTTP {response.status_code}: {detail}",
                )

            async for line in response.aiter_lines():
                chunk = _parse_sse_line(line)
                if chunk is None:
                    continue

                if _chunk_text(chunk):
                    if ttft is None:
                        ttft = time.perf_counter() - started
                    n_chunks += 1

                if usage := chunk.get("usage"):
                    n_prompt_tokens = usage.get("prompt_tokens", 0)
                    n_output_tokens = usage.get("completion_tokens", 0)

    except Exception as exc:
        return RequestMetrics(
            ok=False,
            total_s=time.perf_counter() - started,
            ttft_s=ttft,
            error=f"{type(exc).__name__}: {exc}",
        )

    total = time.perf_counter() - started

    # Servers that omit the usage chunk still stream one token per chunk, so the
    # chunk count is the honest fallback.
    if not n_output_tokens:
        n_output_tokens = n_chunks

    return RequestMetrics(
        ok=ttft is not None,
        total_s=total,
        ttft_s=ttft,
        n_prompt_tokens=n_prompt_tokens,
        n_output_tokens=n_output_tokens,
        error=None if ttft is not None else "no tokens streamed",
    )


__all__ = ["EndpointInfo", "RequestMetrics", "probe_endpoint", "stream_request"]
