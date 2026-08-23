"""Rich console rendering.

Kept separate from the evaluation code so results stay plain data: the same
``RunRecord`` renders to a terminal here, to JSON in the store, and to whatever
Phase 7's Pareto view needs later.
"""

from __future__ import annotations

from rich.console import Console, RenderableType
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from ..metadata.hardware import BYTES_PER_GIB, HardwareInfo
from ..regression import RegressionReport
from ..results import RunRecord

console = Console()


def _gib(value: int | None) -> str:
    return f"{value / BYTES_PER_GIB:.2f} GiB" if value else "n/a"


def render_hardware(hardware: HardwareInfo) -> Table:
    table = Table(title="Hardware", title_justify="left", show_header=False, box=None)
    table.add_column(style="cyan", no_wrap=True)
    table.add_column()

    table.add_row("Accelerator", hardware.accelerator)
    for gpu in hardware.gpus:
        detail = f"{gpu.name} - {gpu.total_memory_gib:.1f} GiB VRAM"
        if gpu.compute_capability:
            detail += f", sm_{gpu.compute_capability.replace('.', '')}"
        if gpu.driver_version:
            detail += f", driver {gpu.driver_version}"
        table.add_row(f"GPU {gpu.index}", detail)

    table.add_row("CPU", f"{hardware.cpu} ({hardware.cpu_count} threads)")
    table.add_row("RAM", _gib(hardware.total_ram_bytes))
    table.add_row("OS", hardware.os)
    table.add_row("Fingerprint", hardware.fingerprint)
    return table


def render_environment(record: RunRecord) -> Table:
    table = Table(title="Environment", title_justify="left", show_header=False, box=None)
    table.add_column(style="cyan", no_wrap=True)
    table.add_column()
    table.add_row("Python", record.environment.python_version)
    table.add_row("CUDA (torch)", record.environment.cuda_version or "n/a")
    for name, version in sorted(record.environment.packages.items()):
        table.add_row(name, version)
    table.add_row("Fingerprint", record.environment.fingerprint)
    return table


def render_model(record: RunRecord) -> Table:
    model = record.model
    table = Table(title="Model", title_justify="left", show_header=False, box=None)
    table.add_column(style="cyan", no_wrap=True)
    table.add_column()

    table.add_row("Id", model.id)
    table.add_row("Revision", model.revision or "default")
    if model.resolved_commit:
        table.add_row("Commit", model.resolved_commit)
    table.add_row("Architecture", model.architecture or "n/a")
    params = f"{model.n_parameters_b:.2f}B" if model.n_parameters_b else "n/a"
    table.add_row("Parameters", params)
    table.add_row("Loaded as", f"{model.dtype} on {model.device}")
    table.add_row("Weights in memory", _gib(model.weights_size_bytes))
    table.add_row("Eval context", str(model.context_length))
    table.add_row("Arch fingerprint", model.architecture_fingerprint or "n/a")
    return table


def render_tasks(record: RunRecord) -> Table:
    table = Table(title="Evaluation", title_justify="left", header_style="bold")
    table.add_column("Task", style="bold")
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    table.add_column("Samples", justify="right")
    table.add_column("Tokens", justify="right")
    table.add_column("Time", justify="right")

    for result in record.tasks:
        if not result.metrics:
            table.add_row(
                result.name,
                Text(str(result.details.get("error", "no metrics")), style="red"),
                "-",
                "-",
                "-",
                f"{result.duration_s:.1f}s",
            )
            continue

        for index, metric in enumerate(result.metrics):
            first = index == 0
            table.add_row(
                result.name if first else "",
                metric.name,
                metric.format(),
                f"{result.n_samples:,}" if first else "",
                f"{result.n_tokens:,}" if first else "",
                f"{result.duration_s:.1f}s" if first else "",
            )
    return table


def render_baseline_inference(record: RunRecord) -> Panel | None:
    inference = record.baseline_inference
    if inference is None or not inference.samples:
        return None

    table = Table(show_header=True, header_style="bold", box=None, pad_edge=False)
    table.add_column("Prompt", max_width=32, overflow="ellipsis")
    table.add_column("Output", max_width=48, overflow="ellipsis")
    table.add_column("New tok", justify="right")
    table.add_column("Latency", justify="right")
    table.add_column("tok/s", justify="right")

    for sample in inference.samples:
        table.add_row(
            sample.prompt.replace("\n", " "),
            (sample.output or "").replace("\n", " ") or "-",
            str(sample.n_generated_tokens),
            f"{sample.latency_s:.2f}s",
            f"{sample.tokens_per_second:.1f}",
        )

    footer = Text(
        f"\nmean {inference.mean_tokens_per_second:.1f} tok/s"
        f" | peak VRAM {_gib(inference.peak_vram_bytes)}"
        f"\nMeasured in the {inference.runtime} runtime - a smoke test, NOT a"
        " deployment performance claim.",
        style="dim",
    )

    return Panel(
        _stack(table, footer),
        title="Baseline inference",
        title_align="left",
        border_style="dim",
    )


def _stack(*renderables: RenderableType) -> Table:
    grid = Table.grid()
    grid.add_column()
    for renderable in renderables:
        grid.add_row(renderable)
    return grid


def render_deployment(benchmark) -> Table:
    """Render a concurrency sweep.

    Unlike the Transformers smoke test, these numbers are deployment claims, so
    the table names the runtime that produced them.
    """
    table = Table(
        title=(
            f"Deployment benchmark - {benchmark.backend}"
            + (f" {benchmark.runtime_version}" if benchmark.runtime_version else "")
            + f" @ {benchmark.endpoint}"
        ),
        title_justify="left",
        header_style="bold",
    )
    table.add_column("Concurrency", justify="right")
    table.add_column("TTFT p50", justify="right")
    table.add_column("TTFT p99", justify="right")
    table.add_column("TPOT p50", justify="right")
    table.add_column("Throughput", justify="right")
    table.add_column("Req/s", justify="right")
    table.add_column("Peak VRAM", justify="right")
    table.add_column("Failed", justify="right")

    for phase in benchmark.phases:
        table.add_row(
            str(phase.concurrency),
            f"{phase.ttft.p50 * 1000:.0f}ms" if phase.ttft else "-",
            f"{phase.ttft.p99 * 1000:.0f}ms" if phase.ttft else "-",
            f"{phase.tpot.p50 * 1000:.1f}ms" if phase.tpot else "-",
            f"{phase.output_tokens_per_s:.1f} tok/s",
            f"{phase.requests_per_s:.2f}",
            _gib(phase.peak_vram_bytes),
            Text(str(phase.n_failed), style="red") if phase.n_failed else "0",
        )
    return table


def render_run(record: RunRecord, *, verbose: bool = False) -> None:
    """Print a full run record."""
    status_style = "green" if record.status == "ok" else "red"
    console.print(
        Panel(
            Text(f"{record.run_id}", style="bold"),
            subtitle=Text(
                f"{record.status.upper()} | {record.total_duration_s:.1f}s"
                f" | config {record.config_fingerprint}",
                style=status_style,
            ),
            subtitle_align="left",
            border_style=status_style,
        )
    )

    console.print(render_model(record))
    console.print()
    console.print(render_tasks(record))

    if (panel := render_baseline_inference(record)) is not None:
        console.print()
        console.print(panel)

    if record.deployment is not None:
        console.print()
        console.print(render_deployment(record.deployment))
        total = record.deployment.device_total_vram_bytes
        console.print(
            Text(
                f"Measured in the {record.deployment.backend} runtime."
                f" Peak VRAM is device-wide" + (f" of {_gib(total)} total" if total else "") + ".",
                style="dim",
            )
        )

    if verbose:
        console.print()
        console.print(render_hardware(record.hardware))
        console.print()
        console.print(render_environment(record))

    if record.error:
        console.print()
        console.print(Text(f"error: {record.error}", style="red"))


def render_regression(report: RegressionReport) -> None:
    """Print a baseline-vs-candidate comparison."""
    table = Table(
        title=(
            f"Regression: {report.candidate_run_id} vs {report.baseline_run_id} "
            f"(min retention {report.min_retention * 100:.1f}%)"
        ),
        title_justify="left",
        header_style="bold",
    )
    table.add_column("Task", style="bold")
    table.add_column("Metric")
    table.add_column("Baseline", justify="right")
    table.add_column("Candidate", justify="right")
    table.add_column("Delta", justify="right")
    table.add_column("Retention", justify="right")
    table.add_column("Verdict", justify="center")

    verdict_styles = {"pass": "green", "fail": "red", "not_comparable": "yellow"}

    for comparison in report.comparisons:
        delta = Text(
            f"{comparison.delta:+.4f}",
            style="green" if comparison.improved else "red",
        )
        # A change inside combined sampling noise should not be read as a trend.
        if comparison.significant is False:
            delta.append(" (noise)", style="dim")

        table.add_row(
            comparison.task,
            comparison.metric,
            f"{comparison.baseline:.4f}",
            f"{comparison.candidate:.4f}",
            delta,
            comparison.format_retention(),
            Text(comparison.verdict, style=verdict_styles[comparison.verdict]),
        )

    console.print(table)

    if report.issues:
        console.print()
        issue_styles = {"info": "dim", "warning": "yellow", "error": "red"}
        for issue in report.issues:
            console.print(
                Text(f"  [{issue.level}] {issue.message}", style=issue_styles[issue.level])
            )

    console.print()
    if report.passed:
        console.print(
            Text("PASS - quality held within the configured threshold.", style="bold green")
        )
    else:
        reasons = []
        if report.failures:
            reasons.append(f"{len(report.failures)} metric(s) below threshold")
        if report.blocking_issues:
            reasons.append(f"{len(report.blocking_issues)} blocking comparability issue(s)")
        console.print(Text(f"FAIL - {'; '.join(reasons)}.", style="bold red"))


__all__ = [
    "console",
    "render_deployment",
    "render_environment",
    "render_hardware",
    "render_regression",
    "render_run",
]
