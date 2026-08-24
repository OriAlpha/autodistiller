"""Wiring for the ``optimize`` command.

Assembles the pieces the earlier phases built into the single call the roadmap
specifies. Kept out of ``cli.py`` because the wiring -- how a candidate becomes
an evaluation, how a benchmark gets a server to talk to -- is real behaviour
worth reading on its own, not argument parsing.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from pathlib import Path

from ..candidates.generator import Candidate, generate_candidates
from ..candidates.shape import load_shape
from ..config import BaselineInferenceSpec, DatasetSpec, ModelSpec, RunConfig, TaskSpec
from ..metadata.profiles import GPUProfile
from ..results import DeploymentBenchmark, RunRecord
from ..runner import run_evaluation
from ..serving.backends import resolve_backend
from ..serving.benchmark import run_deployment_benchmark
from ..serving.launcher import LaunchSpec, serving
from ..store import RunStore
from .constraints import Constraints, Objective
from .pipeline import CandidateOutcome, OptimizationResult, Optimizer

ProgressFn = Callable[[str], None]

BENCHMARK_PROMPT_TOKENS = 256
BENCHMARK_MAX_TOKENS = 128
BENCHMARK_CONCURRENCY = (1, 8)
"""The request shape the optimizer benchmarks candidates with.

Defined once because two things need to agree on it: the benchmarker that
drives the server, and the cache key that decides whether an earlier benchmark
still describes the same measurement.
"""

WSL_VLLM_TEMPLATE = (
    'wsl -d Ubuntu -e bash -lc "'
    "CU=$HOME/vllm-env/lib/python3.12/site-packages/nvidia/cu13; "
    "VLLM_WSL2_ENABLE_PIN_MEMORY=1 CUDA_HOME=$CU "
    "LIBRARY_PATH=$HOME/cudalibs:$CU/lib:/usr/lib/wsl/lib "
    "PATH=$HOME/vllm-env/bin:$CU/bin:$PATH "
    "$HOME/vllm-env/bin/vllm serve {model} --port {port} "
    '--max-model-len {max_model_len} --gpu-memory-utilization 0.85 {kv_flag}"'
)
"""Launching vLLM from Windows means going through WSL, with the environment
Phase 2 had to discover. See docs/vllm-on-wsl.md.

The ``$VAR`` references are deliberately unescaped. The outer shell here is
cmd.exe, which does not expand them, so they reach bash intact. Escaping them
as ``\\$VAR`` for a POSIX outer shell makes bash treat them as literal text and
the command exits 127.
"""

NATIVE_VLLM_TEMPLATE = "vllm serve {model} --port {port} --max-model-len {max_model_len} {kv_flag}"

NATIVE_LLAMACPP_TEMPLATE = (
    "llama-server -m {model} --port {port} -c {max_model_len} -ngl 999 {kv_flag}"
)
"""llama.cpp builds natively on Windows, so unlike vLLM it needs no WSL wrapper.

``-ngl 999`` offloads every layer it can to the GPU. llama.cpp defaults to CPU,
and a CPU-served candidate benchmarked against a GPU-served baseline is not a
comparison, it is a mistake.
"""

WSL_VLLM_STOP = (
    "wsl -d Ubuntu -e bash -lc "
    '"pkill -f vllm-env/bin/vllm; sleep 8; pkill -9 -f vllm-env/bin/vllm; true"'
)
"""Terminating wsl.exe does not reach the server inside the VM.

Escalates because vLLM's own shutdown is slow: on SIGTERM it takes the engine
core down first and can hold the port for the better part of a minute, which is
long enough for the next candidate to collide with it.
"""


def build_evaluator(
    tasks: list[TaskSpec],
    *,
    base_model: ModelSpec,
    output_dir: Path,
    seed: int = 1234,
    reuse: bool = True,
) -> Callable[[str, Candidate], RunRecord]:
    """Screen quality with the Phase 1 engine, unchanged.

    Candidates are evaluated exactly the way the baseline was, which is the
    only reason their retention figures mean anything.
    """

    def evaluate(target: str, candidate: Candidate) -> RunRecord:
        config = RunConfig(
            model=ModelSpec(
                id=target,
                dtype=base_model.dtype,
                device=base_model.device,
                trust_remote_code=base_model.trust_remote_code,
            ),
            tasks=tasks,
            baseline_inference=BaselineInferenceSpec(enabled=False),
            seed=seed,
            label=candidate.id,
            output_dir=output_dir,
        )
        return run_evaluation(config, save=True, reuse=reuse)

    return evaluate


def build_benchmarker(
    launch: LaunchSpec,
    *,
    base_model: ModelSpec,
    backend: str = "vllm",
    prompt_tokens: int = BENCHMARK_PROMPT_TOKENS,
    max_tokens: int = BENCHMARK_MAX_TOKENS,
    concurrency_levels: tuple[int, ...] = BENCHMARK_CONCURRENCY,
    progress: ProgressFn | None = None,
) -> Callable[[CandidateOutcome], DeploymentBenchmark]:
    """Start a server for the candidate, measure it, and shut it down."""
    backend_spec = resolve_backend(backend)

    def benchmark(outcome: CandidateOutcome) -> DeploymentBenchmark:
        # The same artifact is named differently depending on who serves it:
        # vLLM takes the directory, llama-server the .gguf inside it.
        model = (
            backend_spec.model_path(outcome.served_model) if outcome.served_model else base_model.id
        )
        candidate = outcome.candidate

        with serving(
            launch,
            model,
            max_model_len=candidate.max_model_len,
            kv_dtype=candidate.kv_dtype,
            progress=progress,
        ) as url:
            return asyncio.run(
                run_deployment_benchmark(
                    url=url,
                    backend=backend,
                    prompt_tokens=prompt_tokens,
                    max_tokens=max_tokens,
                    concurrency_levels=concurrency_levels,
                    ignore_eos=backend_spec.supports_ignore_eos,
                    progress=progress,
                )
            )

    return benchmark


def optimize(
    *,
    model: ModelSpec,
    tasks: list[TaskSpec],
    constraints: Constraints,
    objective: Objective = Objective.BALANCED,
    backend: str = "vllm",
    profile: GPUProfile | None = None,
    calibration: DatasetSpec | None = None,
    llama_cpp_dir: str | None = None,
    launch: LaunchSpec | None = None,
    artifacts_root: Path = Path("artifacts"),
    runs_dir: Path = Path("runs"),
    methods: tuple[str, ...] | None = None,
    context_lengths: tuple[int, ...] | None = None,
    concurrency: int = 8,
    max_candidates: int = 12,
    stop_early: bool = True,
    skip_benchmark: bool = False,
    reuse: bool = True,
    progress: ProgressFn | None = None,
) -> OptimizationResult:
    """Run the whole search: generate, screen, compress, evaluate, benchmark, rank.

    ``reuse`` turns the Phase 6 cache off. Everything measured is written to
    ``runs_dir`` and looked up there, so a repeated search costs only the stages
    whose inputs actually changed.
    """
    shape = load_shape(model.id, revision=model.revision, trust_remote_code=model.trust_remote_code)
    if progress is not None:
        progress(f"{shape.describe()}")

    backend_spec = resolve_backend(backend)
    candidate_set = generate_candidates(
        shape,
        backend=backend,
        profile=profile,
        budget_bytes=constraints.max_vram_bytes,
        methods=methods,
        context_lengths=context_lengths,
        kv_dtypes=backend_spec.kv_dtypes,
        concurrency=concurrency,
        max_candidates=max_candidates,
    )
    if progress is not None:
        progress(
            f"{len(candidate_set.accepted)} candidates survived screening "
            f"of {candidate_set.n_considered} considered"
        )

    benchmark_fn = None
    if launch is not None and not skip_benchmark:
        benchmark_fn = build_benchmarker(
            launch,
            base_model=model,
            backend=backend,
            progress=progress,
        )

    optimizer = Optimizer(
        model=model,
        constraints=constraints,
        objective=objective,
        backend=backend,
        artifacts_root=artifacts_root,
        evaluate_fn=build_evaluator(tasks, base_model=model, output_dir=runs_dir, reuse=reuse),
        benchmark_fn=benchmark_fn,
        calibration=calibration,
        llama_cpp_dir=llama_cpp_dir,
        stop_early=stop_early,
        store=RunStore(runs_dir),
        reuse=reuse,
        benchmark_settings={
            "prompt_tokens": BENCHMARK_PROMPT_TOKENS,
            "max_tokens": BENCHMARK_MAX_TOKENS,
            "concurrency_levels": list(BENCHMARK_CONCURRENCY),
        },
        progress=progress,
    )
    return optimizer.run(candidate_set)


__all__ = [
    "NATIVE_LLAMACPP_TEMPLATE",
    "NATIVE_VLLM_TEMPLATE",
    "WSL_VLLM_STOP",
    "WSL_VLLM_TEMPLATE",
    "build_benchmarker",
    "build_evaluator",
    "optimize",
]
