"""Deployment benchmark: drive a serving endpoint and record what it costs.

The roadmap is explicit that deployment claims must be measured in the
deployment runtime, so everything here runs against a live server. Results are
tagged ``is_deployment_claim=True``, in contrast to the Transformers-level smoke
test from Phase 1.

Load is applied at several concurrency levels rather than one, because the
interesting question is not "how fast is one request" but "where does latency
start to fall over as load rises" -- which is what a serving decision actually
turns on.
"""

from __future__ import annotations

import asyncio
import statistics
import time
from collections.abc import Callable

import httpx

from ..metadata.hardware import current_vram_bytes
from ..results import ConcurrencyResult, DeploymentBenchmark, LatencyStats
from .client import RequestMetrics, probe_endpoint, stream_request

VRAM_POLL_INTERVAL_S = 0.25

# Deterministic filler so every run sends the same prompt. Real prompts vary,
# but a benchmark that varies its input cannot be compared against itself.
_FILLER = (
    "Deployment optimization requires measuring a model in the runtime that will "
    "actually serve it, under load that resembles production traffic. "
)

ProgressFn = Callable[[str], None]


def _percentile(values: list[float], fraction: float) -> float:
    """Nearest-rank percentile.

    Preferred over interpolation here: benchmark samples are few and a reported
    p99 should be a latency that actually happened.
    """
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(int(fraction * len(ordered)), len(ordered) - 1)
    return ordered[index]


def _stats(values: list[float]) -> LatencyStats | None:
    if not values:
        return None
    return LatencyStats(
        mean=statistics.fmean(values),
        p50=_percentile(values, 0.50),
        p90=_percentile(values, 0.90),
        p99=_percentile(values, 0.99),
        min=min(values),
        max=max(values),
    )


def build_prompt(target_tokens: int) -> str:
    """Filler text of roughly ``target_tokens`` tokens.

    Approximate by design: the server reports the true prompt token count in the
    usage chunk, and that measured number is what gets recorded.
    """
    # ~1.3 tokens per word is a reasonable cross-tokenizer approximation.
    words_needed = max(int(target_tokens / 1.3), 1)
    words = (_FILLER * (words_needed // len(_FILLER.split()) + 2)).split()
    return " ".join(words[:words_needed])


class _VramSampler:
    """Polls device VRAM in the background and keeps the peak.

    Device-wide, not process-scoped: it includes anything else resident on the
    GPU. That is the right number for "will this deployment fit", and it is the
    only number available when the server runs across a WSL boundary.
    """

    def __init__(self, device_index: int = 0) -> None:
        self.device_index = device_index
        self.peak_used: int | None = None
        self.total: int | None = None
        self._task: asyncio.Task | None = None
        self._stop = asyncio.Event()

    def _sample(self) -> None:
        reading = current_vram_bytes(self.device_index)
        if reading is None:
            return
        free, total = reading
        used = total - free
        self.total = total
        self.peak_used = used if self.peak_used is None else max(self.peak_used, used)

    async def _loop(self) -> None:
        while not self._stop.is_set():
            self._sample()
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=VRAM_POLL_INTERVAL_S)
            except asyncio.TimeoutError:
                continue

    async def __aenter__(self) -> _VramSampler:
        self._sample()
        self._stop.clear()
        self._task = asyncio.create_task(self._loop())
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        self._stop.set()
        if self._task is not None:
            await self._task
        self._sample()


async def _run_phase(
    client: httpx.AsyncClient,
    *,
    url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    concurrency: int,
    n_requests: int,
    use_chat: bool,
    ignore_eos: bool,
) -> tuple[list[RequestMetrics], float]:
    """Fire ``n_requests`` with at most ``concurrency`` in flight."""
    limiter = asyncio.Semaphore(concurrency)

    async def one() -> RequestMetrics:
        async with limiter:
            return await stream_request(
                client,
                url=url,
                model=model,
                prompt=prompt,
                max_tokens=max_tokens,
                use_chat=use_chat,
                ignore_eos=ignore_eos,
            )

    started = time.perf_counter()
    results = await asyncio.gather(*(one() for _ in range(n_requests)))
    return list(results), time.perf_counter() - started


def _aggregate(
    results: list[RequestMetrics], duration_s: float, concurrency: int
) -> ConcurrencyResult:
    ok = [r for r in results if r.ok]
    failed = [r for r in results if not r.ok]

    ttfts = [r.ttft_s for r in ok if r.ttft_s is not None]
    tpots = [r.tpot_s for r in ok if r.tpot_s is not None]
    latencies = [r.total_s for r in ok]
    total_output_tokens = sum(r.n_output_tokens for r in ok)

    return ConcurrencyResult(
        concurrency=concurrency,
        n_requests=len(results),
        n_failed=len(failed),
        duration_s=duration_s,
        ttft=_stats(ttfts),
        tpot=_stats(tpots),
        request_latency=_stats(latencies),
        total_output_tokens=total_output_tokens,
        output_tokens_per_s=total_output_tokens / duration_s if duration_s > 0 else 0.0,
        requests_per_s=len(ok) / duration_s if duration_s > 0 else 0.0,
        mean_prompt_tokens=(statistics.fmean([r.n_prompt_tokens for r in ok]) if ok else 0.0),
        errors=sorted({r.error for r in failed if r.error})[:5],
    )


async def run_deployment_benchmark(
    *,
    url: str,
    backend: str,
    model: str | None = None,
    prompt_tokens: int = 256,
    max_tokens: int = 128,
    concurrency_levels: tuple[int, ...] = (1, 4, 16),
    requests_per_level: int | None = None,
    warmup_requests: int = 2,
    use_chat: bool = False,
    ignore_eos: bool = True,
    device_index: int = 0,
    progress: ProgressFn | None = None,
    runtime_version: str | None = None,
    transport: httpx.AsyncBaseTransport | None = None,
) -> DeploymentBenchmark:
    """Benchmark a running endpoint across a concurrency sweep."""

    def say(message: str) -> None:
        if progress is not None:
            progress(message)

    info = await probe_endpoint(url, transport=transport)
    if not info.reachable:
        raise ConnectionError(f"no serving endpoint at {url}: {info.error}")

    served_model = model or info.default_model
    if served_model is None:
        raise ValueError(f"{url} reports no models; pass --served-model explicitly")

    prompt = build_prompt(prompt_tokens)
    phases: list[ConcurrencyResult] = []

    async with httpx.AsyncClient(timeout=httpx.Timeout(300.0), transport=transport) as client:
        # The first requests pay for CUDA graph capture and cache warmup, which
        # no steady-state workload repeats.
        if warmup_requests:
            say(f"warmup ({warmup_requests} requests)")
            await _run_phase(
                client,
                url=url,
                model=served_model,
                prompt=prompt,
                max_tokens=max_tokens,
                concurrency=1,
                n_requests=warmup_requests,
                use_chat=use_chat,
                ignore_eos=ignore_eos,
            )

        async with _VramSampler(device_index) as vram:
            for concurrency in concurrency_levels:
                n_requests = requests_per_level or max(concurrency * 4, 8)
                say(f"concurrency {concurrency}: {n_requests} requests")
                results, duration = await _run_phase(
                    client,
                    url=url,
                    model=served_model,
                    prompt=prompt,
                    max_tokens=max_tokens,
                    concurrency=concurrency,
                    n_requests=n_requests,
                    use_chat=use_chat,
                    ignore_eos=ignore_eos,
                )
                phase = _aggregate(results, duration, concurrency)
                phase.peak_vram_bytes = vram.peak_used
                phases.append(phase)
                summary = f"  {phase.output_tokens_per_s:.1f} tok/s"
                if phase.ttft is not None:
                    summary += f" | TTFT p50 {phase.ttft.p50 * 1000:.0f}ms"
                if phase.n_failed:
                    summary += f" | {phase.n_failed} failed"
                say(summary)

    return DeploymentBenchmark(
        backend=backend,
        runtime_version=runtime_version,
        endpoint=url,
        served_model=served_model,
        prompt_tokens_requested=prompt_tokens,
        max_tokens=max_tokens,
        phases=phases,
        device_total_vram_bytes=vram.total,
    )


__all__ = ["build_prompt", "run_deployment_benchmark"]
