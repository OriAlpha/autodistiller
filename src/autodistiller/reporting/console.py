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

    # A stall that inflated a phase must stay visible in a record read back
    # later, not only in the progress output of the run that produced it.
    for phase in benchmark.phases:
        for warning in phase.warnings:
            note = f"concurrency {phase.concurrency}: {warning}"
            table.caption = f"{table.caption}\n{note}" if table.caption else note
    if table.caption:
        table.caption_style = "yellow"
        table.caption_justify = "left"

    return table


def render_compression(artifact) -> Table:
    """Render a produced artifact and the recipe that made it."""
    recipe = artifact.recipe
    table = Table(
        title=f"Compression artifact - {recipe.label}",
        title_justify="left",
        show_header=False,
        box=None,
    )
    table.add_column(style="cyan", no_wrap=True)
    table.add_column()

    table.add_row("Source model", artifact.source_model)
    table.add_row("Method", f"{recipe.method} ({recipe.describe()})")
    table.add_row("Algorithm", recipe.algorithm)
    table.add_row("Left uncompressed", ", ".join(recipe.ignore) or "-")
    if recipe.needs_calibration:
        table.add_row(
            "Calibration",
            f"{recipe.n_calibration_samples} samples @ {recipe.max_seq_length} tokens"
            f" (fingerprint {recipe.calibration_fingerprint})",
        )
    else:
        table.add_row("Calibration", "not required")
    table.add_row("Artifact", artifact.output_dir)
    table.add_row("Size on disk", _gib(artifact.artifact_bytes))
    table.add_row("Backend", f"{artifact.backend} {artifact.versions.get(artifact.backend, '')}")
    table.add_row("Duration", f"{artifact.duration_s:.1f}s")
    return table


def render_candidates(candidate_set, *, show_rejected: bool = True) -> Table:
    """Render the generated search space.

    Rejected candidates are shown with their reasons: a shorter list with no
    explanation is not an explainable search space.
    """
    table = Table(
        title=f"Candidates for {candidate_set.model_id} on {candidate_set.backend}",
        title_justify="left",
        header_style="bold",
    )
    table.add_column("#", justify="right", style="dim")
    table.add_column("Candidate")
    table.add_column("Context", justify="right")
    table.add_column("KV", justify="right")
    table.add_column("Weights", justify="right")
    table.add_column("KV cache", justify="right")
    table.add_column("Est. total", justify="right")
    table.add_column("Status")

    for index, candidate in enumerate(candidate_set.accepted, start=1):
        estimate = candidate.estimate
        table.add_row(
            str(index),
            Text(candidate.method or "baseline", style="bold" if candidate.is_baseline else ""),
            f"{candidate.max_model_len:,}",
            candidate.kv_dtype,
            _gib(estimate.weights_bytes),
            _gib(estimate.kv_cache_bytes),
            _gib(estimate.total_bytes),
            Text("fits", style="green"),
        )

    if show_rejected:
        for rejection in candidate_set.rejected:
            candidate = rejection.candidate
            table.add_row(
                "-",
                Text(candidate.method or "baseline", style="dim"),
                Text(f"{candidate.max_model_len:,}", style="dim"),
                Text(candidate.kv_dtype, style="dim"),
                Text(_gib(candidate.estimate.weights_bytes), style="dim"),
                Text(_gib(candidate.estimate.kv_cache_bytes), style="dim"),
                Text(_gib(candidate.estimate.total_bytes), style="dim"),
                Text("; ".join(rejection.reasons), style="yellow"),
            )

    return table


def render_optimization(result) -> Table:
    """Render every configuration that was tried, and how far each one got."""
    table = Table(
        title=(
            f"Optimization: {result.model_id} on {result.backend}, "
            f"objective {result.objective.value}"
        ),
        title_justify="left",
        header_style="bold",
    )
    table.add_column("Candidate")
    table.add_column("Stage")
    table.add_column("Quality", justify="right")
    table.add_column("Size", justify="right")
    table.add_column("Throughput", justify="right")
    table.add_column("TTFT", justify="right")
    table.add_column("Cached", justify="center")
    table.add_column("Verdict")

    best = result.recommended
    for outcome in result.outcomes:
        benchmark = outcome.benchmark
        single = benchmark.single_stream if benchmark else None
        peak = benchmark.best_throughput if benchmark else None

        if outcome.error:
            verdict = Text("failed", style="red")
        elif outcome.violations:
            verdict = Text(outcome.violations[0], style="yellow")
        elif best is not None and outcome is best:
            verdict = Text("RECOMMENDED", style="bold green")
        else:
            verdict = Text("qualified", style="green")

        table.add_row(
            outcome.candidate.id,
            outcome.stage,
            f"{outcome.quality_retention * 100:.2f}%" if outcome.quality_retention else "-",
            _gib(outcome.weights_bytes),
            f"{peak.output_tokens_per_s:.0f} tok/s" if peak else "-",
            f"{single.ttft.p50 * 1000:.0f}ms" if single and single.ttft else "-",
            # Which stages this candidate did not have to pay for again.
            Text(",".join(s[:4] for s in outcome.reused), style="cyan") if outcome.reused else "-",
            verdict,
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

    if record.compression is not None:
        console.print()
        console.print(render_compression(record.compression))

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


def render_pareto(report) -> Table:
    """Render the trade-off surface: who is optimal, who is beaten, and by what.

    The point is transparency. A candidate that lost should show the numbers it
    lost on, sitting next to the candidate that beat it.
    """
    table = Table(
        title=f"Pareto frontier - {report.frontier.labels}",
        title_justify="left",
        header_style="bold",
    )
    table.add_column("Candidate")
    for axis in report.axes:
        table.add_column(axis.label, justify="right")
    table.add_column("Verdict")

    frontier = report.frontier
    groups = (
        (frontier.optimal, Text("Pareto-optimal", style="bold green")),
        (frontier.dominated, Text("dominated", style="dim")),
        (frontier.incomparable, Text("not measured on every axis", style="yellow")),
    )

    for outcomes, verdict in groups:
        for outcome in outcomes:
            table.add_row(
                outcome.candidate.id,
                *[axis.format(outcome) for axis in report.axes],
                verdict,
            )

    return table


def render_recommendations(report) -> Table:
    """Render the named options, each with what it costs to choose it."""
    table = Table(
        title="Recommendations",
        title_justify="left",
        header_style="bold",
    )
    table.add_column("Option", style="bold")
    table.add_column("Candidate")
    table.add_column("Wins on")
    table.add_column("Frontier", justify="center")
    table.add_column("Gives up", overflow="fold")

    for recommendation in report.recommendations:
        table.add_row(
            recommendation.label,
            recommendation.outcome.candidate.id,
            f"{recommendation.score.basis} {recommendation.score.detail}".strip(),
            Text("yes", style="green")
            if recommendation.on_frontier
            else Text("no", style="yellow"),
            # An empty note means it was best (or tied best) on every axis.
            recommendation.trade_off or Text("nothing", style="dim"),
        )

    return table


def render_export(manifest, destination) -> Panel:
    """Render what was exported, and whether it would actually serve."""
    table = Table(show_header=False, box=None, pad_edge=False)
    table.add_column(style="cyan", no_wrap=True)
    table.add_column(overflow="fold")

    recipe = manifest.artifact.recipe if manifest.artifact else None
    table.add_row("Source model", manifest.source_model)
    table.add_row("Recipe", recipe.label if recipe else "uncompressed baseline")
    table.add_row("Serving", manifest.served_path)
    if manifest.artifact_bytes:
        table.add_row("Size on disk", _gib(manifest.artifact_bytes))
    if manifest.quality_retention is not None:
        table.add_row("Quality retention", f"{manifest.quality_retention * 100:.2f}%")
    table.add_row("From run", manifest.run_id)
    table.add_row("Written to", str(destination))

    table.add_row("", "")
    for check in manifest.checks:
        mark = Text("PASS", style="green") if check["ok"] else Text("FAIL", style="bold red")
        table.add_row(check["name"], Text.assemble(mark, "  ", check["detail"]))

    table.add_row("", "")
    table.add_row("Serve it", Text(manifest.serve_command, style="bold"))

    ok = manifest.deployable
    return Panel(
        table,
        title=f"Export - {'deployable' if ok else 'NOT deployable'}",
        title_align="left",
        border_style="green" if ok else "red",
    )


__all__ = [
    "console",
    "render_candidates",
    "render_compression",
    "render_deployment",
    "render_environment",
    "render_export",
    "render_hardware",
    "render_optimization",
    "render_pareto",
    "render_recommendations",
    "render_regression",
    "render_run",
]
