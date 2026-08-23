"""Deployment benchmarking.

Driven against a fake OpenAI-compatible server, so the whole Phase 2 path is
covered without a GPU or an installed serving runtime. The fake streams on a
controlled clock, which is what makes the timing assertions meaningful: if TTFT
accounting were wrong, a server that stalls 200ms before its first token would
not show up as 200ms.
"""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from autodistiller.metadata.hardware import GPUInfo
from autodistiller.metadata.profiles import (
    PROFILES,
    architecture_for,
    capabilities_for,
    profile_from_gpu,
    resolve_profile,
)
from autodistiller.serving.backends import BACKENDS, resolve_backend
from autodistiller.serving.benchmark import _percentile, build_prompt, run_deployment_benchmark
from autodistiller.serving.client import probe_endpoint, stream_request

MODEL = "fake/model"


async def _sse_body(
    *,
    n_tokens: int,
    ttft_delay: float,
    token_delay: float,
    include_usage: bool,
    prompt_tokens: int,
    chat: bool,
):
    await asyncio.sleep(ttft_delay)
    for index in range(n_tokens):
        if index:
            await asyncio.sleep(token_delay)
        choice = (
            {"delta": {"content": f"t{index}"}, "index": 0}
            if chat
            else {"text": f"t{index}", "index": 0}
        )
        yield f"data: {json.dumps({'choices': [choice]})}\n\n".encode()

    if include_usage:
        usage = {"prompt_tokens": prompt_tokens, "completion_tokens": n_tokens}
        yield f"data: {json.dumps({'choices': [], 'usage': usage})}\n\n".encode()
    yield b"data: [DONE]\n\n"


def fake_server(
    *,
    n_tokens: int = 8,
    ttft_delay: float = 0.05,
    token_delay: float = 0.01,
    include_usage: bool = True,
    prompt_tokens: int = 123,
    chat: bool = False,
    status: int = 200,
    models: list[str] | None = None,
) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/models":
            listed = [{"id": m} for m in (models if models is not None else [MODEL])]
            return httpx.Response(200, json={"data": listed})
        if status != 200:
            return httpx.Response(status, text="upstream exploded")
        return httpx.Response(
            200,
            content=_sse_body(
                n_tokens=n_tokens,
                ttft_delay=ttft_delay,
                token_delay=token_delay,
                include_usage=include_usage,
                prompt_tokens=prompt_tokens,
                chat=chat,
            ),
        )

    return httpx.MockTransport(handler)


async def _one_request(transport: httpx.MockTransport, **kwargs):
    async with httpx.AsyncClient(transport=transport) as client:
        return await stream_request(
            client, url="http://fake", model=MODEL, prompt="hi", max_tokens=8, **kwargs
        )


# --- endpoint probing ---------------------------------------------------


def test_probe_reports_served_models():
    info = asyncio.run(probe_endpoint("http://fake", transport=fake_server()))
    assert info.reachable
    assert info.models == [MODEL]
    assert info.default_model == MODEL


def test_probe_records_failure_instead_of_raising():
    def boom(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    info = asyncio.run(probe_endpoint("http://fake", transport=httpx.MockTransport(boom)))
    assert not info.reachable
    assert info.error and "ConnectError" in info.error


# --- single request timing ----------------------------------------------


def test_ttft_measures_time_to_first_token():
    metrics = asyncio.run(_one_request(fake_server(ttft_delay=0.20, token_delay=0.001)))
    assert metrics.ok
    assert metrics.ttft_s == pytest.approx(0.20, abs=0.15)


def test_tpot_excludes_prefill():
    """A slow first token must not inflate per-token decode cost."""
    metrics = asyncio.run(_one_request(fake_server(n_tokens=11, ttft_delay=0.30, token_delay=0.01)))
    assert metrics.tpot_s == pytest.approx(0.01, abs=0.02)
    # TPOT is decode-only, so it stays far below the average over the whole request.
    assert metrics.tpot_s < metrics.total_s / metrics.n_output_tokens


def test_token_counts_come_from_the_usage_chunk():
    metrics = asyncio.run(_one_request(fake_server(n_tokens=5, prompt_tokens=321)))
    assert metrics.n_output_tokens == 5
    assert metrics.n_prompt_tokens == 321


def test_token_count_falls_back_to_chunk_count():
    """Servers that omit usage still stream one token per chunk."""
    metrics = asyncio.run(_one_request(fake_server(n_tokens=7, include_usage=False)))
    assert metrics.n_output_tokens == 7
    assert metrics.n_prompt_tokens == 0


def test_chat_completions_shape_is_parsed():
    metrics = asyncio.run(_one_request(fake_server(n_tokens=4, chat=True), use_chat=True))
    assert metrics.ok
    assert metrics.n_output_tokens == 4


def test_http_error_is_captured_not_raised():
    metrics = asyncio.run(_one_request(fake_server(status=500)))
    assert not metrics.ok
    assert "500" in metrics.error


def test_a_stream_with_no_tokens_is_a_failure():
    metrics = asyncio.run(_one_request(fake_server(n_tokens=0, include_usage=False)))
    assert not metrics.ok
    assert metrics.ttft_s is None


# --- aggregation --------------------------------------------------------


def test_percentile_is_nearest_rank():
    values = [float(v) for v in range(1, 101)]
    assert _percentile(values, 0.50) == 51.0
    assert _percentile(values, 0.90) == 91.0
    assert _percentile(values, 0.99) == 100.0


def test_percentile_handles_a_single_sample():
    assert _percentile([4.2], 0.99) == 4.2


def test_percentile_of_nothing_is_zero():
    assert _percentile([], 0.5) == 0.0


def test_build_prompt_scales_with_target():
    short = build_prompt(32)
    long = build_prompt(512)
    assert len(long) > len(short) > 0


# --- full sweep ---------------------------------------------------------


def _run_sweep(transport, **kwargs):
    return asyncio.run(
        run_deployment_benchmark(
            url="http://fake",
            backend="vllm",
            transport=transport,
            warmup_requests=0,
            **kwargs,
        )
    )


def test_sweep_produces_a_phase_per_concurrency_level():
    result = _run_sweep(
        fake_server(n_tokens=4, ttft_delay=0.01, token_delay=0.001),
        concurrency_levels=(1, 2),
        requests_per_level=4,
    )
    assert [p.concurrency for p in result.phases] == [1, 2]
    assert all(p.n_requests == 4 for p in result.phases)
    assert all(p.n_failed == 0 for p in result.phases)


def test_sweep_result_is_marked_a_deployment_claim():
    """The contrast with Phase 1's Transformers smoke test has to be explicit."""
    result = _run_sweep(fake_server(), concurrency_levels=(1,), requests_per_level=2)
    assert result.is_deployment_claim is True
    assert result.backend == "vllm"
    assert result.served_model == MODEL


def test_concurrency_actually_overlaps_requests():
    """Eight serial requests at 100ms each take ~800ms; run 8-wide they take ~100ms.
    If the semaphore or gather were wrong, both phases would cost the same."""
    result = _run_sweep(
        fake_server(n_tokens=1, ttft_delay=0.1, token_delay=0.0, include_usage=False),
        concurrency_levels=(1, 8),
        requests_per_level=8,
    )
    serial, parallel = result.phases
    assert parallel.duration_s < serial.duration_s / 2


def test_throughput_is_reported_per_phase():
    result = _run_sweep(
        fake_server(n_tokens=10, ttft_delay=0.01, token_delay=0.001),
        concurrency_levels=(1, 4),
        requests_per_level=4,
    )
    assert all(p.output_tokens_per_s > 0 for p in result.phases)
    assert result.best_throughput is not None
    assert result.single_stream is not None and result.single_stream.concurrency == 1


def test_failures_are_counted_and_reported():
    result = _run_sweep(fake_server(status=503), concurrency_levels=(1,), requests_per_level=3)
    phase = result.phases[0]
    assert phase.n_failed == 3
    assert phase.errors and "503" in phase.errors[0]


def test_unreachable_endpoint_fails_loudly():
    def boom(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    with pytest.raises(ConnectionError, match="no serving endpoint"):
        _run_sweep(httpx.MockTransport(boom), concurrency_levels=(1,))


def test_server_with_no_models_needs_an_explicit_name():
    with pytest.raises(ValueError, match="served-model"):
        _run_sweep(fake_server(models=[]), concurrency_levels=(1,))


# --- hardware profiles --------------------------------------------------


@pytest.mark.parametrize(
    ("compute_capability", "expected"),
    [
        ("7.5", {"fp32", "fp16", "int8", "int4"}),
        ("8.0", {"fp32", "fp16", "int8", "int4", "bf16"}),
        ("8.9", {"fp32", "fp16", "int8", "int4", "bf16", "fp8"}),
        ("12.0", {"fp32", "fp16", "int8", "int4", "bf16", "fp8", "fp4"}),
    ],
)
def test_capabilities_follow_compute_capability(compute_capability, expected):
    assert capabilities_for(compute_capability) == expected


def test_turing_has_no_bf16():
    """The rule that matters most: bf16 arrived with Ampere."""
    assert "bf16" not in capabilities_for("7.5")
    assert "bf16" in capabilities_for("8.0")


def test_architecture_names():
    assert architecture_for("12.0").startswith("Blackwell")
    assert architecture_for("8.9") == "Ada Lovelace"
    assert architecture_for("8.0") == "Ampere"


def test_unknown_gpu_still_profiles_from_detection():
    """An unrecognized card is not an error; its measured facts are enough."""
    gpu = GPUInfo(
        index=0,
        name="NVIDIA Some Future Card",
        total_memory_bytes=16 * 1024**3,
        compute_capability="12.0",
    )
    profile = profile_from_gpu(gpu)
    assert profile.vram_gib == pytest.approx(16.0)
    assert "fp4" in profile.capabilities


def test_known_gpu_matches_its_profile():
    gpu = GPUInfo(
        index=0,
        name="NVIDIA GeForce RTX 5070 Laptop GPU",
        total_memory_bytes=8151 * 1024**2,
        compute_capability="12.0",
    )
    assert profile_from_gpu(gpu).name == "rtx-5070"


def test_profiles_resolve_by_name():
    assert resolve_profile("A100-80GB").vram_gib == 80
    with pytest.raises(KeyError, match="available"):
        resolve_profile("nope")


def test_every_profile_has_coherent_capabilities():
    for profile in PROFILES.values():
        assert profile.capabilities
        assert profile.vram_gib > 0


# --- backends -----------------------------------------------------------


def test_vllm_launch_command():
    command = resolve_backend("vllm").launch_command(
        "Qwen/Qwen3-0.6B", port=8000, max_model_len=4096, gpu_memory_utilization=0.8
    )
    assert command.startswith("vllm serve Qwen/Qwen3-0.6B")
    assert "--max-model-len 4096" in command
    assert "--gpu-memory-utilization 0.8" in command


def test_llama_cpp_is_registered_for_phase_9():
    backend = resolve_backend("llama.cpp")
    assert backend.default_port == 8080
    assert backend.supports_ignore_eos is False


def test_unknown_backend_lists_the_options():
    with pytest.raises(KeyError, match="vllm"):
        resolve_backend("tensorrt")


def test_all_backends_can_build_a_launch_command():
    for backend in BACKENDS.values():
        assert backend.launch_command("some/model")
