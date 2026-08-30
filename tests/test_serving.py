"""Deployment benchmarking.

Driven against a fake OpenAI-compatible server, so the whole Phase 2 path is
covered without a GPU or an installed serving runtime. The fake streams on a
controlled clock, which is what makes the timing assertions meaningful: if TTFT
accounting were wrong, a server that stalls 200ms before its first token would
not show up as 200ms.
"""

from __future__ import annotations

import asyncio
import contextlib as _contextlib
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
from autodistiller.serving.benchmark import (
    _aggregate,
    _percentile,
    build_prompt,
    run_deployment_benchmark,
)
from autodistiller.serving.client import RequestMetrics, probe_endpoint, stream_request

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
    # Warmup off unless a test is about warmup: it costs requests the other
    # tests would then have to account for.
    kwargs.setdefault("warmup_requests", 0)
    return asyncio.run(
        run_deployment_benchmark(
            url="http://fake",
            backend="vllm",
            transport=transport,
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


# --- VRAM sampling ------------------------------------------------------


def test_device_vram_is_consistent_or_absent():
    """Whatever the source, free must never exceed total.

    Returns None on a machine with no NVIDIA device, which is the CI case.
    """
    from autodistiller.metadata.hardware import device_vram_bytes

    reading = device_vram_bytes(0)
    if reading is None:
        pytest.skip("no NVIDIA device")
    free, total = reading
    assert 0 <= free <= total
    assert total > 0


@pytest.mark.gpu
def test_nvml_tracks_real_allocations():
    """The sampler must actually move when memory is taken.

    A reading that never changes is worse than no reading, because it looks
    like a measurement. (An earlier version sampled through torch, which cannot
    see memory held by a server in another process and reported a flat 1.11 GiB
    at every concurrency level.)
    """
    import torch

    from autodistiller.metadata.hardware import device_vram_bytes

    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")

    before = device_vram_bytes(0)
    assert before is not None

    block = torch.zeros(256 * 1024 * 1024 // 4, dtype=torch.float32, device="cuda")  # 256 MiB
    torch.cuda.synchronize()
    try:
        after = device_vram_bytes(0)
        assert after is not None
        used_before, used_after = before[1] - before[0], after[1] - after[0]
        assert used_after > used_before
    finally:
        del block
        torch.cuda.empty_cache()


# --- server lifecycle ---------------------------------------------------

FAKE_SERVER = """
import json, sys
from http.server import BaseHTTPRequestHandler, HTTPServer

port = int(sys.argv[1])

class H(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/v1/models":
            body = json.dumps({"data": [{"id": "fake"}]}).encode()
            self.send_response(200)
        else:
            body = b"{}"
            self.send_response(404)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
    def log_message(self, *a):
        pass

HTTPServer(("127.0.0.1", port), H).serve_forever()
"""


def _free_port() -> int:
    import socket

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_launcher_starts_waits_and_stops(tmp_path):
    """The whole lifecycle against a real process."""
    import sys

    from autodistiller.serving.launcher import LaunchSpec, serving

    script = tmp_path / "fake_server.py"
    script.write_text(FAKE_SERVER, encoding="utf-8")
    port = _free_port()

    spec = LaunchSpec(
        template=f'"{sys.executable}" "{script}" {port}',
        url=f"http://127.0.0.1:{port}",
        port=port,
        ready_timeout_s=30,
    )

    with serving(spec, "fake/model") as url:
        response = httpx.get(f"{url}/v1/models", timeout=5)
        assert response.status_code == 200
        assert response.json()["data"][0]["id"] == "fake"

    # The server runs under a shell, so it is a grandchild: terminating the
    # process we hold is not enough, and the tree has to be killed.
    with pytest.raises(httpx.HTTPError):
        httpx.get(f"http://127.0.0.1:{port}/v1/models", timeout=2)


def test_launcher_gives_up_when_the_process_dies(tmp_path):
    """A server that already exited will never become ready, and waiting the
    full timeout for that is wasted minutes."""
    import sys

    from autodistiller.serving.launcher import LaunchSpec, ServerError, serving

    spec = LaunchSpec(
        template=f'"{sys.executable}" -c "raise SystemExit(3)"',
        url=f"http://127.0.0.1:{_free_port()}",
        ready_timeout_s=60,
    )
    with pytest.raises(ServerError, match="exited with code"), serving(spec, "m"):
        pass


def test_stopped_means_not_serving_models():
    """The port outlives the engine: after vLLM exits something still answers
    with a 404, so waiting for silence would wait forever."""
    from autodistiller.serving.launcher import wait_until_stopped

    assert wait_until_stopped(f"http://127.0.0.1:{_free_port()}", timeout_s=5)


def test_launch_template_formats_the_candidate():
    from autodistiller.serving.launcher import LaunchSpec

    spec = LaunchSpec(template="serve {model} -p {port} -c {max_model_len} {kv_flag}", port=9001)
    assert spec.command_for("m", max_model_len=4096, kv_dtype="fp8") == (
        "serve m -p 9001 -c 4096 --kv-cache-dtype fp8"
    )
    assert spec.command_for("m", max_model_len=2048).strip() == "serve m -p 9001 -c 2048"


def test_wsl_path_translates_local_artifacts(tmp_path):
    """A launch that crosses into WSL carries the model path with it, and a
    Windows path means nothing on the other side."""
    import sys

    from autodistiller.serving.launcher import wsl_path

    artifact = tmp_path / "model-fp8"
    artifact.mkdir()
    translated = wsl_path(str(artifact))

    if sys.platform == "win32":
        assert translated.startswith("/mnt/")
        assert "\\" not in translated
    assert translated.endswith("model-fp8")


def test_wsl_path_leaves_hub_ids_alone():
    """Repo ids are not paths and must survive untouched."""
    from autodistiller.serving.launcher import wsl_path

    assert wsl_path("Qwen/Qwen3-0.6B") == "Qwen/Qwen3-0.6B"


def test_launch_spec_applies_the_translator():
    from autodistiller.serving.launcher import LaunchSpec

    spec = LaunchSpec(template="serve {model}", path_translator=lambda p: f"/mnt/{p}")
    assert spec.command_for("x") == "serve /mnt/x"


# --- cold-start stalls --------------------------------------------------


def _request(total_s: float, *, ttft_s: float = 0.058, tokens: int = 128) -> RequestMetrics:
    """One streamed request. Defaults mirror a real measured request."""
    return RequestMetrics(
        ok=True, total_s=total_s, ttft_s=ttft_s, n_prompt_tokens=256, n_output_tokens=tokens
    )


def test_a_stall_is_reported_rather_than_measured():
    """The numbers here are the ones a real run produced: seven requests at
    0.67s, one at 9.33s, and a phase duration of 21.1s. Wall-clock throughput
    absorbed the stall and reported 48 tok/s for a server doing 209."""
    results = [_request(0.67) for _ in range(7)] + [_request(9.33)]
    phase = _aggregate(results, duration_s=21.14, concurrency=1)

    assert phase.output_tokens_per_s == pytest.approx(1024 / 21.14, rel=0.01)
    assert phase.throughput_efficiency < 0.5
    assert phase.warnings
    assert "stall" in phase.warnings[0]
    assert "unreliable" in phase.warnings[0]


def test_a_clean_phase_carries_no_warning():
    """The second server in the same run: eight requests within 0.03s of each
    other. Nothing to flag, and flagging it would make the warning worthless."""
    results = [_request(0.85, ttft_s=0.060) for _ in range(8)]
    phase = _aggregate(results, duration_s=6.79, concurrency=1)

    assert phase.throughput_efficiency > 0.8
    assert not phase.warnings


def test_queueing_at_higher_concurrency_is_not_a_stall():
    """Latency legitimately spreads under load, so the check has to compare
    against the per-token ceiling rather than against latency spread."""
    results = [_request(0.78) for _ in range(28)] + [_request(1.08) for _ in range(4)]
    phase = _aggregate(results, duration_s=3.47, concurrency=8)

    assert not phase.warnings


def test_efficiency_is_absent_when_there_is_nothing_to_compare():
    phase = _aggregate([RequestMetrics(ok=False, total_s=0.0, error="boom")], 1.0, 1)
    assert phase.throughput_efficiency is None
    assert not phase.warnings


# --- warmup -------------------------------------------------------------


WARM_DELAY = 0.05
"""How long a "warm" fake request takes.

Deliberately not smaller. Warmup settles when three consecutive requests land
within 25% of the fastest seen, and at a 5ms delay ordinary scheduling jitter is
comfortably more than 25% -- so the fixture, not the code, decided whether the
test passed. At 50ms the same jitter is under 10%.
"""


def _stalling_server(*, slow_requests: int, slow_delay: float = 0.40):
    """A server that answers /v1/models before it is warm.

    Which is what vLLM does: it reports healthy, then pays for CUDA graph
    capture on the first real requests.
    """
    state = {"seen": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={"data": [{"id": MODEL}]})
        state["seen"] += 1
        cold = state["seen"] <= slow_requests
        return httpx.Response(
            200,
            content=_sse_body(
                n_tokens=4,
                ttft_delay=slow_delay if cold else WARM_DELAY,
                token_delay=0.001,
                include_usage=True,
                prompt_tokens=16,
                chat=False,
            ),
        )

    return httpx.MockTransport(handler), state


def test_warmup_absorbs_the_cold_start_so_the_phase_does_not():
    """The whole point: a stall that used to land in the first measured phase
    should be paid for during warmup instead."""
    transport, state = _stalling_server(slow_requests=3)
    result = _run_sweep(transport, concurrency_levels=(1,), warmup_requests=2)

    assert state["seen"] > 3, "warmup stopped before the server was warm"
    phase = result.phases[0]
    assert not phase.warnings
    assert phase.request_latency.max < 0.20  # no cold request (0.40s) reached it


def _warmup_calls(latencies: list[float], *, minimum: int = 2, maximum: int = 10) -> int:
    """How many requests warmup uses against a scripted sequence of latencies.

    Drives the decision directly with no wall clock involved. The earlier
    version of this test ran a fake server on real sleeps and failed whenever
    the machine was busy -- it was measuring the scheduler, not the code.
    """
    import autodistiller.serving.benchmark as bm

    calls = {"n": 0}

    async def fake_request(client, **kwargs):
        index = min(calls["n"], len(latencies) - 1)
        calls["n"] += 1
        return RequestMetrics(ok=True, total_s=latencies[index], ttft_s=0.01, n_output_tokens=4)

    original = bm.stream_request
    bm.stream_request = fake_request
    try:
        asyncio.run(
            bm._warm_until_stable(
                None,
                url="http://fake",
                model=MODEL,
                prompt="hi",
                max_tokens=4,
                use_chat=False,
                ignore_eos=True,
                minimum=minimum,
                maximum=maximum,
            )
        )
    finally:
        bm.stream_request = original
    return calls["n"]


def test_a_warm_server_settles_immediately():
    """Steady latencies from the first request: nothing to wait out."""
    assert _warmup_calls([0.50] * 12) == 3


def test_a_cold_start_is_waited_out_but_not_forever():
    """Three slow requests, then steady. Warmup must absorb the slow ones and
    then stop -- not stop early, and not run to the cap."""
    assert _warmup_calls([9.0, 4.0, 1.0] + [0.50] * 12) == 6


def test_a_server_that_never_settles_stops_at_the_cap():
    """Alternating fast and slow never satisfies the stability rule, and a
    longer wait would not fix it."""
    assert _warmup_calls([0.5, 5.0] * 20, maximum=10) == 10


def test_warmup_is_bounded_when_a_server_never_settles():
    """A server that stays erratic is telling us something a longer wait will
    not fix, so the cap has to hold."""
    transport, state = _stalling_server(slow_requests=1000)
    _run_sweep(transport, concurrency_levels=(1,), warmup_requests=2, warmup_max_requests=4)

    assert state["seen"] <= 4 + 8


def test_warmup_stops_when_the_server_is_failing():
    """Ten failing warmup requests tell us nothing the phase will not report
    properly."""
    transport = fake_server(status=500)
    result = _run_sweep(transport, concurrency_levels=(1,), warmup_requests=2)
    assert result.phases[0].n_failed == 8


@_contextlib.contextmanager
def _patched_get(*, status: int = 200, payload=None, raises=None):
    """Stand in for httpx.get inside the launcher."""
    import autodistiller.serving.launcher as launcher

    class _Resp:
        status_code = status

        def json(self):
            if payload is None:
                raise ValueError("no json")
            return payload

    def fake_get(url, **kwargs):
        if raises is not None:
            raise raises
        return _Resp()

    original = launcher.httpx.get
    launcher.httpx.get = fake_get
    try:
        yield
    finally:
        launcher.httpx.get = original


# --- launching into an occupied port --------------------------------------


def test_a_port_already_serving_something_is_refused():
    """A squatter that answers 200 is the dangerous case: the benchmark would
    measure it and report the numbers as the model's."""
    from autodistiller.serving.launcher import ServerError, check_port_is_free

    squatter = _patched_get(status=200, payload={"data": [{"id": "some-other-service"}]})
    with pytest.raises(ServerError, match="already in use"), squatter:
        check_port_is_free("http://fake")


def test_a_port_answering_anything_at_all_is_refused():
    """404 is occupied too. On WSL2 a Windows process holding the port stops
    localhost forwarding, so the server we launch is unreachable and readiness
    would time out blaming the wrong thing."""
    from autodistiller.serving.launcher import ServerError, check_port_is_free

    with pytest.raises(ServerError, match="HTTP 404"), _patched_get(status=404, payload=None):
        check_port_is_free("http://fake")


def test_a_refused_connection_means_the_port_is_free():
    from autodistiller.serving.launcher import check_port_is_free

    with _patched_get(raises=httpx.ConnectError("refused")):
        check_port_is_free("http://fake")  # must not raise


def test_the_refusal_names_what_is_squatting():
    from autodistiller.serving.launcher import ServerError, check_port_is_free

    squatter = _patched_get(status=200, payload={"data": [{"id": "stockedge-api"}]})
    with pytest.raises(ServerError) as excinfo, squatter:
        check_port_is_free("http://fake")
    assert "stockedge-api" in str(excinfo.value)


def test_a_short_generation_is_not_mistaken_for_a_stall():
    """Prefill produces no output tokens, so on a 4-token generation it dominates
    the wall clock. A decode-only ceiling called that a stall; it is not one."""
    results = [_request(0.053, ttft_s=0.050, tokens=4) for _ in range(8)]
    phase = _aggregate(results, duration_s=0.62, concurrency=1)

    assert phase.throughput_efficiency > 0.5
    assert not phase.warnings


def test_a_dying_server_reports_its_own_last_words():
    """ "exited with code 1" is true and useless. The runtime printed the reason
    and it has to survive to the error that reports the death."""
    import subprocess
    import tempfile

    from autodistiller.serving.launcher import _last_words

    with tempfile.TemporaryFile(mode="w+") as log:
        log.write("loading weights\nValueError: not enough memory for KV cache\n")
        assert "not enough memory" in _last_words(log)
        assert subprocess  # imported for symmetry with the caller


def test_a_server_that_printed_nothing_adds_nothing():
    import tempfile

    from autodistiller.serving.launcher import _last_words

    with tempfile.TemporaryFile(mode="w+") as log:
        assert _last_words(log) == ""
    assert _last_words(None) == ""


def test_capturing_stderr_cannot_block_the_server():
    """A pipe holds ~64 KB before the writer blocks, and nothing drains it while
    readiness is polled -- so PIPE deadlocks the process being waited on. A file
    has no such limit, and this is the regression that proves it stays a file."""
    import inspect
    import subprocess as sp

    from autodistiller.serving import launcher

    source = inspect.getsource(launcher.serving)
    assert "stderr=log" in source
    assert f"stderr={sp.PIPE}" not in source and "stderr=subprocess.PIPE" not in source


def test_a_long_startup_log_is_truncated_to_its_tail():
    """The reason a server stopped is at the end, and a startup log is long."""
    import tempfile

    from autodistiller.serving.launcher import _last_words

    with tempfile.TemporaryFile(mode="w+") as log:
        log.write("\n".join(f"line {n}" for n in range(500)) + "\nfinal cause\n")
        tail = _last_words(log, lines=5)
        assert "final cause" in tail
        assert "line 0" not in tail
        assert tail.count("\n") <= 5


def test_prompt_file_replaces_the_filler_and_moves_the_cache_key(tmp_path) -> None:
    """Acceptance rate is a property of the text, so the prompt is part of the key.

    A default run must keep the key it already has on disk, though: no prompt
    file means the filler, which is what every cached benchmark was measured
    with.
    """
    from autodistiller.optimize.command import (
        BENCHMARK_CONCURRENCY,
        BENCHMARK_MAX_TOKENS,
        BENCHMARK_PROMPT_TOKENS,
    )
    from autodistiller.serving.benchmark import prompt_fingerprint, read_prompt

    path = tmp_path / "traffic.txt"
    path.write_text("  Summarize the following support ticket:\n\n", encoding="utf-8")
    assert read_prompt(path) == "Summarize the following support ticket:"

    empty = tmp_path / "empty.txt"
    empty.write_text("   \n", encoding="utf-8")
    with pytest.raises(ValueError, match="empty"):
        read_prompt(empty)

    def settings(prompt: str | None) -> dict:
        return {
            "prompt_tokens": BENCHMARK_PROMPT_TOKENS,
            "max_tokens": BENCHMARK_MAX_TOKENS,
            "concurrency_levels": list(BENCHMARK_CONCURRENCY),
            **({"prompt": prompt_fingerprint(prompt)} if prompt else {}),
        }

    assert settings("ticket text") != settings("code review text")
    assert "prompt" not in settings(None)


def test_speculative_json_survives_a_nested_shell() -> None:
    """A WSL launch reaches bash as the contents of a double-quoted string.

    Speculative config is JSON, which is nothing but double quotes. Unescaped,
    bash strips them and vLLM is handed `{method: dflash}` -- not JSON, and it
    fails with an error naming neither the cause nor the quoting. Verified
    against the real cmd.exe -> wsl -> bash chain, not reasoned about.
    """
    from autodistiller.candidates.speculative import SpeculativeSpec
    from autodistiller.serving.launcher import LaunchSpec

    spec = SpeculativeSpec(method="dflash", model="z-lab/X-DFlash", n_tokens=15)
    template = 'wsl -e bash -lc "vllm serve {model} --port {port} {spec_flag}"'

    nested = LaunchSpec(template=template, nested_shell=True)
    assert '\\"method\\"' in nested.command_for("m", speculative_config=spec.as_config())

    direct = LaunchSpec(template="vllm serve {model} {spec_flag}", nested_shell=False)
    command = direct.command_for("m", speculative_config=spec.as_config())
    assert '"method": "dflash"' in command
    assert "\\" not in command

    # Nothing speculating leaves the command exactly as it was.
    assert direct.command_for("m").strip() == "vllm serve m"


def test_speculative_launch_leaves_room_to_schedule() -> None:
    """Draft slots come out of the batched-token budget.

    Every sequence reserves one slot per draft token plus one for the verified
    token. vLLM refuses to start when that leaves nothing to schedule -- observed
    as "max_num_scheduled_tokens is set to -1536", which is 512 - (15+1)*128
    under its own defaults.
    """
    from autodistiller.candidates.speculative import SpeculativeSpec
    from autodistiller.serving.launcher import LaunchSpec

    spec = SpeculativeSpec(method="dflash", model="d", n_tokens=15)
    launch = LaunchSpec(
        template="vllm serve {model} {seqs_flag} {util_flag} {spec_flag}",
        max_num_seqs=8,
        gpu_memory_utilization=0.87,
    )

    command = launch.command_for(
        "m", max_model_len=2048, speculative_config=spec.as_config(), speculative_tokens=15
    )
    assert "--max-num-seqs 8" in command
    assert "--gpu-memory-utilization 0.870" in command

    batched = int(command.split("--max-num-batched-tokens ")[1].split()[0])
    assert batched - (15 + 1) * 8 > 0, "vLLM would refuse to start"

    # Not speculating: no draft budget, and the other flags stand alone.
    plain = launch.command_for("m", max_model_len=2048)
    assert "--max-num-batched-tokens" not in plain
    assert "--max-num-seqs 8" in plain

    # Unset means the runtime keeps its own defaults.
    bare = LaunchSpec(template="vllm serve {model} {seqs_flag} {util_flag}")
    assert bare.command_for("m").split() == ["vllm", "serve", "m"]
