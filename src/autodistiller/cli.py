"""AutoDistiller command line interface.

Phase 1 exposes the evaluation engine:

* ``autodistiller env`` -- what hardware and software stack will be recorded
* ``autodistiller evaluate`` -- measure a model and save a run record
* ``autodistiller compare`` -- check a candidate against a baseline
* ``autodistiller runs`` / ``show`` -- browse past runs

The ``optimize`` command from the roadmap arrives once Phases 2-5 land; it will
call the same evaluation engine underneath.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import typer
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskID, TextColumn, TimeElapsedColumn

from .config import (
    BaselineInferenceSpec,
    ModelSpec,
    MultipleChoiceTask,
    PerplexityTask,
    RunConfig,
    TaskSpec,
)
from .evaluation.registry import PRESETS, resolve_tasks
from .metadata.environment import collect_environment
from .metadata.hardware import detect_hardware
from .regression import DEFAULT_MIN_RETENTION, compare_runs
from .reporting.console import (
    console,
    render_environment,
    render_hardware,
    render_regression,
    render_run,
)
from .results import RunRecord
from .runner import RunObserver, preflight, run_evaluation
from .store import RunStore

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
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Establish a baseline: load a model, evaluate it, and record everything."""
    _configure_logging(verbose)

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
    if (cached := store.find_by_fingerprint(run_config.fingerprint)) is not None:
        # Phase 6 will skip the work entirely; for now, just say so.
        console.print(
            f"[yellow]note:[/yellow] an identical config was already run "
            f"({cached.run_id}). Re-running; Phase 6 will reuse it instead."
        )

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
        record = run_evaluation(run_config, observer=ConsoleObserver(progress), store=store)

    console.print()
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
) -> None:
    """List saved runs, newest first."""
    from rich.table import Table

    records = RunStore(runs_dir).list_records(limit=limit)
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


def main() -> None:
    app()


if __name__ == "__main__":
    main()
