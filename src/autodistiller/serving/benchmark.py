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

from ..metadata.hardware import device_vram_bytes
from ..results import ConcurrencyResult, DeploymentBenchmark, LatencyStats
from .client import RequestMetrics, probe_endpoint, stream_request

VRAM_POLL_INTERVAL_S = 0.25

WARMUP_MAX_REQUESTS = 10
WARMUP_STABLE_RUN = 3
WARMUP_TOLERANCE = 0.25
"""How warm is warm enough.

A server answers ``/v1/models`` before it is ready to be measured: vLLM captures
CUDA graphs and compiles kernels lazily, on real requests, after it starts
reporting healthy. A fixed warmup count is a guess at how long that takes, and
guessing low is not visible in the result -- it just moves the cost into the
first measured phase.

So warm up until the server proves it is warm: stop once ``WARMUP_STABLE_RUN``
consecutive requests land within ``WARMUP_TOLERANCE`` of the fastest seen.
Bounded by ``WARMUP_MAX_REQUESTS``, because a server that never stabilizes is
telling us something a longer wait will not fix.
"""

MIN_THROUGHPUT_EFFICIENCY = 0.5
"""Below this, a phase's throughput is not describing steady-state serving.

Wall-clock throughput is total tokens over wall-clock time, so it absorbs any
stall. Per-token latency does not: a median is unmoved by one outlier. When the
two disagree badly the wall-clock number is measuring a hiccup rather than the
server, and reporting it as throughput would corrupt the ranking that depends
on it.
"""

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

    Reads through NVML rather than torch, which reports only the caller's own
    CUDA context and therefore cannot see a server running in another process.
    """

    def __init__(self, device_index: int = 0) -> None:
        self.device_index = device_index
        self.peak_used: int | None = None
        self.total: int | None = None
        self._task: asyncio.Task | None = None
        self._stop = asyncio.Event()

    def _sample(self) -> None:
        reading = device_vram_bytes(self.device_index)
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


async def _warm_until_stable(
    client: httpx.AsyncClient,
    *,
    url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    use_chat: bool,
    ignore_eos: bool,
    minimum: int,
    maximum: int,
) -> tuple[bool, list[float]]:
    """Send single requests until the server settles. Returns (settled, latencies).

    Settled means ``WARMUP_STABLE_RUN`` consecutive requests all landed within
    ``WARMUP_TOLERANCE`` of the fastest seen so far. The fastest is the right
    reference: a cold-start stall is slow against a floor the server reaches and
    then holds, so comparing against the floor detects it where comparing
    consecutive pairs would accept a uniformly slow run.
    """
    latencies: list[float] = []

    for _ in range(max(maximum, minimum)):
        result = await stream_request(
            client,
            url=url,
            model=model,
            prompt=prompt,
            max_tokens=max_tokens,
            use_chat=use_chat,
            ignore_eos=ignore_eos,
        )
        if not result.ok:
            # A failing warmup is the server's problem to report; the phases
            # below will record the failures properly.
            break
        latencies.append(result.total_s)

        if len(latencies) < max(minimum, WARMUP_STABLE_RUN):
            continue
        recent = latencies[-WARMUP_STABLE_RUN:]
        if max(recent) <= min(latencies) * (1 + WARMUP_TOLERANCE):
            return True, latencies

    return False, latencies


def _throughput_efficiency(phase: ConcurrencyResult) -> float | None:
    """Measured throughput against what the per-token timings imply.

    At concurrency C each in-flight stream emits a token every TPOT seconds, so
    C/TPOT is the decode ceiling. A healthy phase lands a little under it -- the
    gap is prefill, which produces no output tokens. A phase that lands far under
    it spent its wall-clock time on something other than serving.
    """
    if phase.tpot is None or phase.tpot.p50 <= 0 or phase.output_tokens_per_s <= 0:
        return None
    implied = phase.concurrency / phase.tpot.p50
    return phase.output_tokens_per_s / implied if implied > 0 else None


def _aggregate(
    results: list[RequestMetrics], duration_s: float, concurrency: int
) -> ConcurrencyResult:
    ok = [r for r in results if r.ok]
    failed = [r for r in results if not r.ok]

    ttfts = [r.ttft_s for r in ok if r.ttft_s is not None]
    tpots = [r.tpot_s for r in ok if r.tpot_s is not None]
    latencies = [r.total_s for r in ok]
    total_output_tokens = sum(r.n_output_tokens for r in ok)

    phase = ConcurrencyResult(
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

    efficiency = _throughput_efficiency(phase)
    phase.throughput_efficiency = efficiency

    if efficiency is not None and efficiency < MIN_THROUGHPUT_EFFICIENCY and phase.tpot is not None:
        slowest = phase.request_latency.max if phase.request_latency else 0.0
        median = phase.request_latency.p50 if phase.request_latency else 0.0
        phase.warnings.append(
            f"throughput is {efficiency * 100:.0f}% of what the per-token timings imply "
            f"({phase.output_tokens_per_s:.0f} vs "
            f"{phase.concurrency / phase.tpot.p50:.0f} tok/s); "
            f"slowest request {slowest:.2f}s against a median of {median:.2f}s. "
            f"A stall inflated this measurement -- treat it as unreliable."
        )

    return phase


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
    warmup_max_requests: int = WARMUP_MAX_REQUESTS,
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
        # The first requests pay for CUDA graph capture and kernel compilation,
        # which no steady-state workload repeats. Warm until the server proves
        # it is warm rather than for a fixed count; see WARMUP_TOLERANCE.
        if warmup_requests:
            say(f"warmup (until stable, max {warmup_max_requests})")
            warmed, latencies = await _warm_until_stable(
                client,
                url=url,
                model=served_model,
                prompt=prompt,
                max_tokens=max_tokens,
                use_chat=use_chat,
                ignore_eos=ignore_eos,
                minimum=warmup_requests,
                maximum=warmup_max_requests,
            )
            if latencies:
                detail = f"  warm after {len(latencies)} requests, {latencies[-1]:.2f}s"
                if not warmed:
                    detail += " (never stabilized; measurements may be noisy)"
                say(detail)

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
                for warning in phase.warnings:
                    say(f"  warning: {warning}")

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
