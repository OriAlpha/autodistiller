"""AutoDistiller command line interface.

    env        what hardware and software stack will be recorded
    tasks      evaluation presets
    evaluate   measure a model and save a run record
    compare    check a candidate against a baseline
    runs/show  browse past runs
    export     make a measured result deployable and reproducible
    history    the experiment cache: what has been measured, what can be reused
    methods    compression methods, and whether they are usable here
    compress   produce a compressed artifact
    candidates the search space for a model
    profiles   GPU capability profiles
    backends   deployment backends
    benchmark  measure a running serving endpoint
    optimize   the whole search, under constraints

Anything measured is cached. ``evaluate``, ``compress`` and ``optimize`` reuse
an identical earlier experiment rather than repeating it, and each takes
``--refresh`` to measure again anyway.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import typer
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskID, TextColumn, TimeElapsedColumn

from .config import (
    BaselineInferenceSpec,
    CompressionSpec,
    DatasetSpec,
    DeploymentSpec,
    ModelSpec,
    MultipleChoiceTask,
    PerplexityTask,
    RunConfig,
    TaskSpec,
)
from .evaluation.registry import PRESETS, resolve_tasks
from .metadata.environment import collect_environment
from .metadata.hardware import detect_hardware
from .metadata.profiles import PROFILES, GPUProfile, profile_from_gpu, resolve_profile
from .regression import DEFAULT_MIN_RETENTION, compare_runs
from .reporting.console import (
    console,
    render_candidates,
    render_compression,
    render_deployment,
    render_environment,
    render_export,
    render_hardware,
    render_optimization,
    render_pareto,
    render_recommendations,
    render_regression,
    render_run,
)
from .results import ModelInfo, RunRecord
from .runner import RunObserver, preflight, run_evaluation
from .store import RunStore, make_run_id

app = typer.Typer(
    name="autodistiller",
    help="Automatically find the best LLM deployment configuration "
    "for your hardware and quality constraints.",
    no_args_is_help=True,
    add_completion=False,
)


class ConsoleObserver(RunObserver):
    """Renders run progress as a live rich progress bar."""

    def __init__(self, progress: Progress) -> None:
        self.progress = progress
        self.task_id: TaskID | None = None

    def stage(self, message: str) -> None:
        console.print(f"[dim]|[/dim] {message}")

    def task_started(self, task: TaskSpec) -> None:
        self.task_id = self.progress.add_task(f"{task.name} ({task.kind})", total=None)

    def task_progress(self, done: int, total: int) -> None:
        if self.task_id is not None:
            self.progress.update(self.task_id, completed=done, total=total)

    def task_finished(self, result) -> None:
        if self.task_id is not None:
            self.progress.remove_task(self.task_id)
            self.task_id = None
        primary = result.primary_metric
        summary = f"{primary.name}={primary.format()}" if primary else "failed"
        console.print(f"[green]+[/green] {result.name}: {summary} ({result.duration_s:.1f}s)")


def _configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )


def _apply_task_overrides(
    tasks: list[TaskSpec],
    *,
    batch_size: int | None,
    max_length: int | None,
    stride: int | None,
) -> list[TaskSpec]:
    """Apply CLI-level overrides to every resolved task."""
    for task in tasks:
        if batch_size is not None:
            task.batch_size = batch_size
        if isinstance(task, PerplexityTask):
            if max_length is not None:
                task.max_length = max_length
            if stride is not None:
                task.stride = stride
        elif isinstance(task, MultipleChoiceTask) and (max_length or stride):
            console.print(
                f"[yellow]note:[/yellow] --max-length/--stride do not apply to "
                f"multiple-choice task {task.name!r}; ignored."
            )
    return tasks


@app.command()
def env(
    as_json: bool = typer.Option(False, "--json", help="Emit machine-readable output"),
) -> None:
    """Show the hardware and software stack that runs will be tagged with."""
    hardware = detect_hardware()
    environment = collect_environment()

    if as_json:
        typer.echo(
            json.dumps(
                {
                    "hardware": hardware.model_dump(mode="json"),
                    "hardware_fingerprint": hardware.fingerprint,
                    "environment": environment.model_dump(mode="json"),
                    "environment_fingerprint": environment.fingerprint,
                },
                indent=2,
            )
        )
        return

    console.print(render_hardware(hardware))
    console.print()

    record_stub = RunRecord.model_construct(environment=environment)
    console.print(render_environment(record_stub))

    if not hardware.gpus:
        console.print(
            "\n[yellow]No CUDA device detected.[/yellow] Evaluation will run on CPU, "
            "which is slow but correct. Deployment profiling (Phase 2) needs an NVIDIA GPU."
        )


@app.command()
def tasks() -> None:
    """List the built-in evaluation tasks."""
    console.print("[bold]Built-in presets[/bold]")
    for name in sorted(PRESETS):
        spec = PRESETS[name](None)
        console.print(f"  [cyan]{name:<16}[/cyan] {spec.kind:<16} {spec.dataset.path}")

    console.print("\n[bold]Custom datasets[/bold]")
    console.print("  [cyan]ppl:PATH        [/cyan] perplexity over a local .txt or .jsonl corpus")
    console.print("  [cyan]mc:PATH         [/cyan] multiple choice over a local .jsonl file")
    console.print(
        "\n  JSONL schema for [cyan]mc:[/cyan] - one object per line:\n"
        '    {"id": "q1", "context": "Question: ...\\nAnswer:", '
        '"choices": [" yes", " no"], "answer_index": 0}'
    )


@app.command()
def evaluate(
    model: str | None = typer.Option(None, "--model", "-m", help="Hugging Face id or local path"),
    task: list[str] | None = typer.Option(
        None, "--task", "-t", help="Preset name, or ppl:PATH / mc:PATH. Repeatable."
    ),
    config_path: Path | None = typer.Option(
        None, "--config", "-c", help="Load a full RunConfig from YAML"
    ),
    revision: str | None = typer.Option(None, help="Model branch, tag or commit"),
    dtype: str = typer.Option("auto", help="auto | float32 | float16 | bfloat16"),
    device: str = typer.Option("auto", help="auto | cpu | cuda | cuda:1"),
    limit: int | None = typer.Option(None, help="Cap documents/examples per task"),
    batch_size: int | None = typer.Option(None, "--batch-size", "-b"),
    max_length: int | None = typer.Option(None, help="Perplexity window (default: model context)"),
    stride: int | None = typer.Option(None, help="Perplexity window step (default: max-length)"),
    seed: int = typer.Option(1234),
    trust_remote_code: bool = typer.Option(False, help="Allow custom modelling code from the repo"),
    no_inference: bool = typer.Option(
        False, "--no-inference", help="Skip the generation smoke test"
    ),
    output_dir: Path = typer.Option(Path("runs"), "--output-dir", "-o"),
    label: str | None = typer.Option(None, help="Human label; excluded from the config hash"),
    save_config: Path | None = typer.Option(
        None, help="Write the resolved config to YAML for reproduction"
    ),
    refresh: bool = typer.Option(
        False, "--refresh", help="Measure again even if this exact experiment is cached"
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Establish a baseline: load a model, evaluate it, and record everything."""
    _configure_logging(verbose)

    # A record older than this call is one the cache handed back rather than one
    # this call measured.
    started_at = datetime.now(timezone.utc)

    if config_path is not None:
        run_config = RunConfig.from_yaml(config_path)
        if model:
            run_config.model.id = model
        run_config.output_dir = output_dir
    else:
        if not model:
            raise typer.BadParameter("provide --model, or --config with a saved RunConfig")
        try:
            resolved = _apply_task_overrides(
                resolve_tasks(task, limit=limit),
                batch_size=batch_size,
                max_length=max_length,
                stride=stride,
            )
        except (ValueError, FileNotFoundError) as exc:
            raise typer.BadParameter(str(exc)) from exc

        run_config = RunConfig(
            model=ModelSpec(
                id=model,
                revision=revision,
                dtype=dtype,  # type: ignore[arg-type]
                device=device,
                trust_remote_code=trust_remote_code,
            ),
            tasks=resolved,
            baseline_inference=BaselineInferenceSpec(enabled=not no_inference),
            seed=seed,
            label=label,
            output_dir=output_dir,
        )

    if save_config is not None:
        run_config.save(save_config)
        console.print(f"[dim]|[/dim] Wrote config to {save_config}")

    store = RunStore(run_config.output_dir)

    if problems := preflight(run_config):
        raise typer.BadParameter("\n".join(problems))

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        console=console,
        transient=True,
    ) as progress:
        record = run_evaluation(
            run_config,
            observer=ConsoleObserver(progress),
            store=store,
            reuse=not refresh,
        )

    console.print()
    if record.created_at < started_at:
        console.print(
            f"[green]cached[/green] {record.run_id} — identical config, hardware and stack. "
            f"[dim]Pass --refresh to measure again.[/dim]"
        )
    render_run(record, verbose=verbose)

    if record.status != "ok":
        raise typer.Exit(code=1)


@app.command()
def compare(
    baseline: str = typer.Argument(..., help="Run id, run directory, or path to record.json"),
    candidate: str = typer.Argument(..., help="Run id, run directory, or path to record.json"),
    min_retention: float = typer.Option(
        DEFAULT_MIN_RETENTION,
        "--min-retention",
        help="Fraction of baseline quality that must be retained (0.95 = 95%)",
    ),
    metric: list[str] | None = typer.Option(
        None, "--metric", help="Restrict the check to these metric names. Repeatable."
    ),
    runs_dir: Path = typer.Option(Path("runs"), "--runs-dir"),
    as_json: bool = typer.Option(False, "--json"),
) -> None:
    """Quality regression check. Exits non-zero when quality did not hold."""
    if not 0 < min_retention <= 2:
        raise typer.BadParameter("--min-retention is a fraction, e.g. 0.95 for 95%")

    store = RunStore(runs_dir)
    try:
        baseline_record = store.resolve(baseline)
        candidate_record = store.resolve(candidate)
    except FileNotFoundError as exc:
        raise typer.BadParameter(str(exc)) from exc

    report = compare_runs(
        baseline_record,
        candidate_record,
        min_retention=min_retention,
        metrics=list(metric) if metric else None,
    )

    if as_json:
        typer.echo(json.dumps(report.model_dump(mode="json") | {"passed": report.passed}, indent=2))
    else:
        render_regression(report)

    if not report.passed:
        raise typer.Exit(code=1)


@app.command(name="runs")
def list_runs(
    runs_dir: Path = typer.Option(Path("runs"), "--runs-dir"),
    limit: int = typer.Option(20, "--limit", "-n"),
    model: str | None = typer.Option(None, "--model", "-m", help="Only runs of this model"),
) -> None:
    """List saved runs, newest first."""
    from rich.table import Table

    records = RunStore(runs_dir).list_records(limit=None if model else limit)
    if model:
        records = [r for r in records if model in r.model.id][:limit]
    if not records:
        console.print(f"[dim]No runs found under {runs_dir}[/dim]")
        return

    table = Table(header_style="bold")
    table.add_column("Run id")
    table.add_column("Model")
    table.add_column("Status")
    table.add_column("Primary metrics")
    table.add_column("Config", style="dim")

    for record in records:
        summaries = []
        for result in record.tasks:
            if (primary := result.primary_metric) is not None:
                summaries.append(f"{result.name}/{primary.name}={primary.value:.4f}")
        table.add_row(
            record.run_id,
            record.model.id,
            f"[green]{record.status}[/green]"
            if record.status == "ok"
            else f"[red]{record.status}[/red]",
            ", ".join(summaries) or "-",
            record.config_fingerprint,
        )

    console.print(table)


@app.command()
def show(
    run: str = typer.Argument(..., help="Run id, run directory, or path to record.json"),
    runs_dir: Path = typer.Option(Path("runs"), "--runs-dir"),
    as_json: bool = typer.Option(False, "--json"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Include hardware and environment"),
) -> None:
    """Show a saved run record."""
    try:
        record = RunStore(runs_dir).resolve(run)
    except FileNotFoundError as exc:
        raise typer.BadParameter(str(exc)) from exc

    if as_json:
        typer.echo(record.to_json())
    else:
        render_run(record, verbose=verbose)


@app.command(name="export")
def export_run(
    run: str = typer.Argument(..., help="Run id, run directory, or path to record.json"),
    runs_dir: Path = typer.Option(Path("runs"), "--runs-dir"),
    output_dir: Path | None = typer.Option(
        None,
        "--output-dir",
        "-o",
        help="Assemble a standalone bundle here. Default: write beside the weights.",
    ),
    backend: str = typer.Option("vllm", "--backend", "-b", help="Runtime the result targets"),
    copy_weights: bool = typer.Option(
        False, "--copy-weights", help="Copy the weights into the bundle rather than referring to it"
    ),
    as_json: bool = typer.Option(False, "--json", help="Emit the manifest instead of a report"),
) -> None:
    """Make a measured result deployable and reproducible.

    Writes the manifest, a deployment note and the exact config beside the
    weights, so the directory you would serve is the one that explains itself.
    Exits non-zero if the artifact would not actually load.
    """
    from .export import export as export_result

    try:
        record = RunStore(runs_dir).resolve(run)
    except FileNotFoundError as exc:
        raise typer.BadParameter(str(exc)) from exc

    try:
        manifest, destination = export_result(
            record,
            backend=backend,
            output_dir=output_dir,
            copy_weights=copy_weights,
        )
    except (KeyError, ValueError, OSError) as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc

    if as_json:
        typer.echo(manifest.to_json())
    else:
        console.print(render_export(manifest, destination))

    if not manifest.deployable:
        raise typer.Exit(code=1)


@app.command()
def history(
    runs_dir: Path = typer.Option(Path("runs"), "--runs-dir"),
    limit: int = typer.Option(30, "--limit", "-n"),
    model: str | None = typer.Option(None, "--model", "-m", help="Only runs of this model"),
    rebuild: bool = typer.Option(
        False, "--rebuild", help="Re-derive the index from the run records on disk"
    ),
    as_json: bool = typer.Option(False, "--json", help="Emit index rows, one JSON object per line"),
) -> None:
    """Show the experiment cache: what has been measured, and what it can reuse.

    Reads the index rather than the records, so it stays fast however many runs
    have accumulated. A row with an experiment key can serve a repeated
    evaluation; one with a benchmark key can serve a repeated deployment
    benchmark. Rows with neither predate the cache and are history only.
    """
    from rich.table import Table

    store = RunStore(runs_dir)
    if rebuild:
        rows = store.rebuild_index()
        console.print(f"[dim]Rebuilt {runs_dir / 'index.jsonl'} from {len(rows)} records[/dim]")

    rows = store.summaries(limit=None if model else limit)
    if model:
        rows = [row for row in rows if model in (row.get("model") or "")][:limit]

    if not rows:
        console.print(f"[dim]No experiments recorded under {runs_dir}[/dim]")
        return

    if as_json:
        for row in rows:
            typer.echo(json.dumps(row, ensure_ascii=False))
        return

    table = Table(header_style="bold")
    table.add_column("Run id")
    table.add_column("Model")
    table.add_column("Candidate")
    table.add_column("Holds")
    table.add_column("Experiment key", style="dim")
    table.add_column("Benchmark key", style="dim")

    for row in rows:
        holds = [
            name
            for name, present in (
                ("eval", row.get("has_tasks")),
                ("benchmark", row.get("has_deployment")),
                ("compression", row.get("has_compression")),
            )
            if present
        ]
        status = row.get("status")
        table.add_row(
            row.get("run_id", "?"),
            row.get("model") or "-",
            row.get("candidate_id") or "-",
            ", ".join(holds) if status == "ok" else f"[red]{status}[/red]",
            row.get("experiment_key") or "-",
            row.get("benchmark_key") or "-",
        )

    console.print(table)
    reusable = sum(1 for row in rows if row.get("experiment_key") or row.get("benchmark_key"))
    console.print(f"[dim]{reusable} of {len(rows)} shown are reusable by the cache[/dim]")


@app.command()
def methods(
    backend: str = typer.Option("vllm", "--backend", "-b", help="Serving backend to check against"),
    all_hardware: bool = typer.Option(
        False, "--all", help="Ignore the detected GPU and list every method"
    ),
) -> None:
    """List compression methods and whether they are usable here."""
    from rich.table import Table

    from .compression.methods import available_methods

    hardware = detect_hardware()
    profile = None
    if not all_hardware and hardware.gpus:
        profile = profile_from_gpu(hardware.gpus[0])

    table = Table(header_style="bold")
    table.add_column("Method")
    table.add_column("Bits")
    table.add_column("Scheme")
    table.add_column("Algorithm")
    table.add_column("Calibration")
    table.add_column("Usable here")

    for entry in available_methods(profile=profile, backend=backend):
        method = entry.method
        table.add_row(
            method.name,
            method.describe_size(),
            method.scheme,
            method.algorithm,
            "required" if method.needs_calibration else "-",
            "[green]yes[/green]"
            if entry.available
            else f"[red]no[/red] [dim]({'; '.join(entry.reasons)})[/dim]",
        )

    console.print(table)
    if profile is not None:
        console.print()
        console.print(
            f"Checked against [cyan]{profile.name}[/cyan] "
            f"({profile.architecture}) and backend [cyan]{backend}[/cyan]."
        )


@app.command()
def compress(
    model: str = typer.Option(..., "--model", "-m", help="Hugging Face id or local path"),
    method: str = typer.Option(..., "--method", help="See `autodistiller methods`"),
    calibration: str | None = typer.Option(
        None, "--calibration", help="Calibration corpus: a preset name, ppl:PATH, or a hub id"
    ),
    samples: int = typer.Option(128, "--samples", help="Calibration samples"),
    max_seq_length: int = typer.Option(2048, help="Calibration sequence length"),
    output_dir: Path | None = typer.Option(None, "--output-dir", "-o"),
    artifacts_root: Path = typer.Option(Path("artifacts"), "--artifacts-root"),
    compress_backend: str | None = typer.Option(
        None,
        "--compress-backend",
        help="Override the toolchain. Defaults to whichever produces the method.",
    ),
    serving_backend: str | None = typer.Option(
        None,
        "--serving-backend",
        help="Runtime this artifact must be servable by. Defaults to the method's own.",
    ),
    compress_python: str | None = typer.Option(
        None, "--compress-python", help="Reuse an interpreter that already has the backend"
    ),
    llama_cpp_dir: str | None = typer.Option(
        None,
        "--llama-cpp",
        envvar="LLAMA_CPP_DIR",
        help="llama.cpp checkout, for GGUF methods. Also read from LLAMA_CPP_DIR.",
    ),
    trust_remote_code: bool = typer.Option(False),
    refresh: bool = typer.Option(
        False, "--refresh", help="Compress again even if this exact artifact already exists"
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Produce a compressed artifact through an existing compression backend."""
    _configure_logging(verbose)

    from .compression.backend import CompressionError
    from .compression.pipeline import run_compression

    calibration_spec: DatasetSpec | None = None
    if calibration:
        try:
            resolved = resolve_tasks([calibration], limit=samples)[0]
            calibration_spec = resolved.dataset
        except (ValueError, FileNotFoundError) as exc:
            raise typer.BadParameter(f"--calibration: {exc}") from exc

    spec = CompressionSpec(
        method=method,
        backend=compress_backend or None,
        calibration=calibration_spec,
        num_calibration_samples=samples,
        max_seq_length=max_seq_length,
        output_dir=output_dir,
        python_executable=compress_python,
        llama_cpp_dir=llama_cpp_dir,
    )

    hardware = detect_hardware()
    profile = profile_from_gpu(hardware.gpus[0]) if hardware.gpus else None

    try:
        artifact = run_compression(
            ModelSpec(id=model, trust_remote_code=trust_remote_code),
            spec,
            output_root=artifacts_root,
            profile=profile,
            serving_backend=serving_backend,
            reuse=not refresh,
            progress=lambda message: console.print(f"[dim]|[/dim] {message}"),
        )
    except (KeyError, ValueError, RuntimeError, CompressionError) as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc

    console.print()
    console.print(render_compression(artifact))
    console.print()
    console.print(
        f"[dim]Serve it with:[/dim] vllm serve {artifact.output_dir} "
        f"--port 8000 --max-model-len 4096"
    )


@app.command()
def candidates(
    model: str = typer.Option(..., "--model", "-m", help="Hugging Face id or local path"),
    backend: str = typer.Option("vllm", "--backend", "-b"),
    max_vram: str | None = typer.Option(
        None, "--max-vram", help="Memory budget, e.g. 8GiB. Defaults to the detected GPU."
    ),
    concurrency: int = typer.Option(
        1, "--concurrency", help="Concurrent sequences the KV cache must hold"
    ),
    context: str | None = typer.Option(
        None, "--context", help="Comma-separated context lengths (default 2048,4096,8192)"
    ),
    method: list[str] | None = typer.Option(
        None, "--method", help="Restrict to these methods. Repeatable."
    ),
    max_candidates: int = typer.Option(25, "--max-candidates"),
    no_baseline: bool = typer.Option(False, "--no-baseline", help="Skip the uncompressed entry"),
    hide_rejected: bool = typer.Option(False, "--hide-rejected"),
    profile_name: str | None = typer.Option(
        None, "--profile", help="Target a GPU you do not have, e.g. a100-80gb"
    ),
    as_json: bool = typer.Option(False, "--json"),
) -> None:
    """Generate the search space: what is worth measuring, and what is not."""
    from .candidates import generate_candidates, load_shape
    from .candidates.memory import parse_size
    from .serving.backends import resolve_backend

    try:
        shape = load_shape(model)
    except Exception as exc:
        raise typer.BadParameter(f"could not read the config for {model!r}: {exc}") from exc

    hardware = detect_hardware()
    profile: GPUProfile | None
    if profile_name:
        try:
            profile = resolve_profile(profile_name)
        except KeyError as exc:
            raise typer.BadParameter(str(exc)) from exc
    else:
        profile = profile_from_gpu(hardware.gpus[0]) if hardware.gpus else None

    budget = None
    if max_vram:
        try:
            budget = parse_size(max_vram)
        except ValueError as exc:
            raise typer.BadParameter(str(exc)) from exc

    context_lengths = None
    if context:
        try:
            context_lengths = tuple(int(p) for p in context.split(",") if p.strip())
        except ValueError as exc:
            raise typer.BadParameter(f"--context must be integers: {exc}") from exc

    try:
        result = generate_candidates(
            shape,
            backend=backend,
            profile=profile,
            budget_bytes=budget,
            methods=tuple(method) if method else None,
            context_lengths=context_lengths,
            # KV cache types are backend-specific; llama.cpp has no fp8.
            kv_dtypes=resolve_backend(backend).kv_dtypes,
            concurrency=concurrency,
            include_baseline=not no_baseline,
            max_candidates=max_candidates,
        )
    except KeyError as exc:
        raise typer.BadParameter(str(exc)) from exc

    if as_json:
        typer.echo(
            json.dumps(
                {
                    "model": result.model_id,
                    "backend": result.backend,
                    "budget_bytes": result.budget_bytes,
                    "concurrency": result.concurrency,
                    "accepted": [
                        {
                            "id": c.id,
                            "method": c.method,
                            "max_model_len": c.max_model_len,
                            "kv_dtype": c.kv_dtype,
                            "estimated_bytes": c.estimate.total_bytes,
                        }
                        for c in result.accepted
                    ],
                    "rejected": [
                        {"id": r.candidate.id, "reasons": list(r.reasons)} for r in result.rejected
                    ],
                },
                indent=2,
            )
        )
        return

    console.print(f"[dim]|[/dim] {shape.describe()}")
    console.print()
    console.print(render_candidates(result, show_rejected=not hide_rejected))
    console.print()
    console.print(
        f"{len(result.accepted)} of {result.n_considered} configurations fit"
        + (f" in {result.budget_bytes / 1024**3:.1f} GiB" if result.budget_bytes else "")
        + f" at concurrency {result.concurrency}."
    )
    if result.rejected:
        summary = ", ".join(f"{k} ({v})" for k, v in result.rejection_summary().items())
        console.print(f"[dim]Rejected for: {summary}[/dim]")
    console.print(
        "[dim]Memory figures are estimates from the model config, "
        "used to screen before expensive runs.[/dim]"
    )


@app.command()
def optimize(
    model: str = typer.Option(..., "--model", "-m", help="Hugging Face id or local path"),
    backend: str = typer.Option("vllm", "--backend", "-b"),
    objective: str = typer.Option(
        "balanced", "--objective", help="throughput | latency | size | quality | balanced"
    ),
    max_vram: str | None = typer.Option(None, "--max-vram", help="e.g. 8GiB"),
    min_quality: float | None = typer.Option(
        None, "--min-quality", help="Percent of baseline quality to retain, e.g. 95"
    ),
    max_ttft_ms: float | None = typer.Option(None, "--max-ttft-ms"),
    min_throughput: float | None = typer.Option(
        None, "--min-throughput", help="Tokens per second at peak concurrency"
    ),
    task: list[str] | None = typer.Option(
        None, "--task", "-t", help="Quality screening tasks (default wikitext2)"
    ),
    limit: int = typer.Option(128, help="Documents per screening task"),
    calibration: str | None = typer.Option(
        None, "--calibration", help="Calibration corpus for methods that need it"
    ),
    method: list[str] | None = typer.Option(None, "--method", help="Restrict the search space"),
    max_candidates: int = typer.Option(12, "--max-candidates"),
    llama_cpp_dir: str | None = typer.Option(
        None,
        "--llama-cpp",
        envvar="LLAMA_CPP_DIR",
        help="llama.cpp checkout, for GGUF methods. Also read from LLAMA_CPP_DIR.",
    ),
    concurrency: int = typer.Option(8, "--concurrency", help="Sequences the KV cache must hold"),
    launch_template: str | None = typer.Option(
        None, "--launch", help="Command template to start a server. See --launch-preset."
    ),
    launch_preset: str = typer.Option(
        "none",
        "--launch-preset",
        help="none | wsl-vllm | native-vllm | native-llamacpp",
    ),
    stop_early: bool = typer.Option(True, "--stop-early/--no-stop-early"),
    artifacts_root: Path = typer.Option(Path("artifacts"), "--artifacts-root"),
    output_dir: Path = typer.Option(Path("runs"), "--output-dir", "-o"),
    refresh: bool = typer.Option(
        False, "--refresh", help="Ignore cached experiments and measure everything again"
    ),
    no_pareto: bool = typer.Option(
        False, "--no-pareto", help="Skip the trade-off analysis and print only the winner"
    ),
    export_dir: Path | None = typer.Option(
        None,
        "--export",
        help="Export the recommended configuration here once the search finishes",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Find the best deployment configuration under explicit constraints."""
    _configure_logging(verbose)

    from .candidates.memory import parse_size
    from .optimize.command import (
        NATIVE_LLAMACPP_TEMPLATE,
        NATIVE_VLLM_TEMPLATE,
        WSL_VLLM_STOP,
        WSL_VLLM_TEMPLATE,
    )
    from .optimize.command import optimize as run_optimize
    from .optimize.constraints import Constraints, Objective
    from .serving.launcher import LaunchSpec, wsl_path

    try:
        chosen_objective = Objective(objective.lower())
    except ValueError as exc:
        raise typer.BadParameter(
            f"unknown objective {objective!r}; choose from {', '.join(o.value for o in Objective)}"
        ) from exc

    budget = None
    if max_vram:
        try:
            budget = parse_size(max_vram)
        except ValueError as exc:
            raise typer.BadParameter(str(exc)) from exc

    if min_quality is not None and not 0 < min_quality <= 100:
        raise typer.BadParameter("--min-quality is a percent, e.g. 95")

    constraints = Constraints(
        min_quality_retention=min_quality / 100 if min_quality is not None else None,
        max_vram_bytes=budget,
        max_ttft_s=max_ttft_ms / 1000 if max_ttft_ms is not None else None,
        min_throughput_tokens_per_s=min_throughput,
    )

    try:
        tasks = resolve_tasks(task, limit=limit)
        calibration_spec = (
            resolve_tasks([calibration], limit=512)[0].dataset if calibration else None
        )
    except (ValueError, FileNotFoundError) as exc:
        raise typer.BadParameter(str(exc)) from exc

    from .serving.backends import resolve_backend

    try:
        backend_spec = resolve_backend(backend)
    except KeyError as exc:
        raise typer.BadParameter(str(exc)) from exc

    template = launch_template
    if template is None and launch_preset != "none":
        template = {
            "wsl-vllm": WSL_VLLM_TEMPLATE,
            "native-vllm": NATIVE_VLLM_TEMPLATE,
            "native-llamacpp": NATIVE_LLAMACPP_TEMPLATE,
        }.get(launch_preset)
        if template is None:
            raise typer.BadParameter(f"unknown --launch-preset {launch_preset!r}")

    # A WSL launch crosses a process boundary, so terminating the launcher does
    # not stop the server; it needs an explicit stop command.
    is_wsl = template == WSL_VLLM_TEMPLATE
    stop = WSL_VLLM_STOP if is_wsl else None
    launch = (
        LaunchSpec(
            template=template,
            stop_template=stop,
            port=backend_spec.default_port,
            url=f"http://localhost:{backend_spec.default_port}",
            # The KV cache flag is not the same word in every runtime.
            kv_flag_template=backend_spec.kv_flag_template,
            # Artifact paths are local; the server is not.
            path_translator=wsl_path if is_wsl else None,
        )
        if template
        else None
    )
    if launch is None and constraints.needs_benchmark:
        raise typer.BadParameter(
            "latency and throughput constraints need a deployment benchmark. "
            "Pass --launch-preset wsl-vllm (or --launch with your own command)."
        )

    hardware = detect_hardware()
    profile = profile_from_gpu(hardware.gpus[0]) if hardware.gpus else None

    try:
        result = run_optimize(
            model=ModelSpec(id=model),
            tasks=tasks,
            constraints=constraints,
            objective=chosen_objective,
            backend=backend,
            profile=profile,
            calibration=calibration_spec,
            llama_cpp_dir=llama_cpp_dir,
            launch=launch,
            artifacts_root=artifacts_root,
            runs_dir=output_dir,
            methods=tuple(method) if method else None,
            concurrency=concurrency,
            max_candidates=max_candidates,
            stop_early=stop_early,
            skip_benchmark=launch is None,
            reuse=not refresh,
            progress=lambda message: console.print(f"[dim]|[/dim] {message}"),
        )
    except (KeyError, ValueError, RuntimeError) as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc

    console.print()
    console.print(render_optimization(result))
    console.print()
    console.print(result.explain())

    if result.qualified and not no_pareto:
        report = result.pareto()
        console.print()
        console.print(render_pareto(report))
        console.print()
        console.print(render_recommendations(report))
        if len(result.qualified) == 1:
            console.print(
                "\n[dim]Only one configuration qualified, so there is nothing to trade off "
                "against. Re-run with --no-stop-early to measure the alternatives.[/dim]"
            )

    if result.recommended is None:
        raise typer.Exit(code=1)

    # Opt-in: a search should not write files into directories the user did not
    # point it at. `autodistiller export <run-id>` covers the in-place case.
    if export_dir is not None:
        _export_recommendation(result, backend=backend, output_dir=export_dir)


def _export_recommendation(result, *, backend: str, output_dir: Path | None) -> None:
    """Make the winner deployable without the user hunting for its run id.

    Only exports when the winner has a stored record behind it. Without one
    there is nothing whose provenance could be written down, and a manifest
    asserting a result it cannot evidence is worse than no manifest.
    """
    from .export import export as export_result

    best = result.recommended
    record = best.record if best is not None else None
    if record is None:
        if output_dir is not None:
            console.print(
                "[yellow]note:[/yellow] the recommended configuration has no stored run "
                "record, so there is nothing to export."
            )
        return

    try:
        manifest, destination = export_result(
            record,
            backend=backend,
            quality_retention=best.quality_retention,
            artifact_dir=best.artifact.output_dir if best.artifact else None,
            output_dir=output_dir,
        )
    except (KeyError, ValueError, OSError) as exc:
        # A search that produced good numbers should not fail because they could
        # not be filed. The result is already printed above.
        console.print(f"[yellow]could not export:[/yellow] {exc}")
        return

    console.print()
    console.print(render_export(manifest, destination))


@app.command()
def backends() -> None:
    """List deployment backends and how to launch them."""
    from .serving.backends import BACKENDS

    for backend in BACKENDS.values():
        console.print(f"[bold cyan]{backend.name}[/bold cyan]  {backend.description}")
        console.print(f"  default port     {backend.default_port}")
        console.print(f"  ignore_eos       {'yes' if backend.supports_ignore_eos else 'no'}")
        console.print(f"  launch           {backend.launch_command('<model>')}")
        if backend.notes:
            console.print(f"  [dim]{backend.notes}[/dim]")
        console.print()


@app.command()
def profiles() -> None:
    """List known NVIDIA hardware profiles and their numeric-format support."""
    from rich.table import Table

    table = Table(header_style="bold")
    table.add_column("Profile")
    table.add_column("VRAM", justify="right")
    table.add_column("Compute")
    table.add_column("Architecture")
    table.add_column("Formats")

    for name in sorted(PROFILES):
        profile = PROFILES[name]
        table.add_row(
            name,
            f"{profile.vram_gib:g} GiB",
            profile.compute_capability,
            profile.architecture,
            ", ".join(sorted(profile.capabilities)),
        )
    console.print(table)

    detected = detect_hardware()
    if detected.gpus:
        found = profile_from_gpu(detected.gpus[0])
        console.print()
        console.print(
            f"Detected: [cyan]{found.name}[/cyan] "
            f"({found.vram_gib:.1f} GiB, {found.architecture}) - "
            f"supports {', '.join(sorted(found.capabilities))}"
        )


@app.command()
def benchmark(
    endpoint: str = typer.Option("http://localhost:8000", "--endpoint", "-e"),
    backend: str = typer.Option("vllm", "--backend", "-b", help="Runtime being measured"),
    served_model: str | None = typer.Option(
        None, "--served-model", help="Defaults to whatever the endpoint reports"
    ),
    prompt_tokens: int = typer.Option(256, help="Approximate prompt length"),
    max_tokens: int = typer.Option(128, help="Output tokens per request"),
    concurrency: str = typer.Option("1,4,16", help="Comma-separated concurrency levels"),
    requests: int | None = typer.Option(
        None, "--requests", "-n", help="Requests per level (default: 4x concurrency, min 8)"
    ),
    warmup: int = typer.Option(2, help="Warmup requests before measuring"),
    use_chat: bool = typer.Option(False, "--chat", help="Use /v1/chat/completions"),
    save: bool = typer.Option(True, help="Write a run record"),
    output_dir: Path = typer.Option(Path("runs"), "--output-dir", "-o"),
    as_json: bool = typer.Option(False, "--json"),
) -> None:
    """Benchmark a running serving endpoint. These numbers are deployment claims."""
    import asyncio

    from .serving.backends import resolve_backend
    from .serving.benchmark import run_deployment_benchmark

    try:
        chosen = resolve_backend(backend)
    except KeyError as exc:
        raise typer.BadParameter(str(exc)) from exc

    try:
        levels = tuple(int(part) for part in concurrency.split(",") if part.strip())
    except ValueError as exc:
        raise typer.BadParameter(f"--concurrency must be integers: {exc}") from exc
    if not levels:
        raise typer.BadParameter("--concurrency needs at least one level")

    spec = DeploymentSpec(
        backend=chosen.name,
        endpoint=endpoint,
        served_model=served_model,
        prompt_tokens=prompt_tokens,
        max_tokens=max_tokens,
        concurrency_levels=list(levels),
        requests_per_level=requests,
        warmup_requests=warmup,
        use_chat=use_chat,
    )

    try:
        result = asyncio.run(
            run_deployment_benchmark(
                url=spec.endpoint,
                backend=spec.backend,
                model=spec.served_model,
                prompt_tokens=spec.prompt_tokens,
                max_tokens=spec.max_tokens,
                concurrency_levels=tuple(spec.concurrency_levels),
                requests_per_level=spec.requests_per_level,
                warmup_requests=spec.warmup_requests,
                use_chat=spec.use_chat,
                ignore_eos=chosen.supports_ignore_eos,
                device_index=spec.device_index,
                progress=lambda message: console.print(f"[dim]|[/dim] {message}"),
            )
        )
    except (ConnectionError, ValueError) as exc:
        console.print(f"[red]{exc}[/red]")
        console.print(
            f"[dim]Start one with:[/dim] {chosen.launch_command(served_model or '<model>')}"
        )
        raise typer.Exit(code=1) from exc

    if as_json:
        typer.echo(json.dumps(result.model_dump(mode="json"), indent=2))
        return

    console.print()
    console.print(render_deployment(result))

    if save:
        config = RunConfig(
            model=ModelSpec(id=result.served_model), deployment=spec, output_dir=output_dir
        )
        record = RunRecord(
            run_id=make_run_id(config),
            config=config,
            config_fingerprint=config.fingerprint,
            model=ModelInfo(id=result.served_model),
            hardware=detect_hardware(),
            environment=collect_environment(),
            deployment=result,
        )
        directory = RunStore(output_dir).save(record)
        console.print(f"[dim]|[/dim] Saved run to {directory}")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
