"""Regression reporting.

The direction-aware retention rule is the part most likely to be got wrong: a
drop in accuracy and a rise in perplexity are both regressions, and both must
land below 1.0.
"""

from __future__ import annotations

import pytest

from autodistiller.config import DatasetSpec, ModelSpec, PerplexityTask, RunConfig
from autodistiller.metadata.environment import EnvironmentInfo
from autodistiller.metadata.hardware import GPUInfo, HardwareInfo
from autodistiller.regression import compare_runs
from autodistiller.results import MetricValue, ModelInfo, RunRecord, TaskResult


def _hardware(gpu_name: str = "RTX 4090") -> HardwareInfo:
    return HardwareInfo(
        hostname="host",
        os="linux",
        cpu="test-cpu",
        cpu_count=8,
        accelerator="cuda",
        gpus=[
            GPUInfo(
                index=0,
                name=gpu_name,
                total_memory_bytes=24 * 1024**3,
                compute_capability="8.9",
            )
        ],
    )


def _environment(transformers: str = "4.44.0") -> EnvironmentInfo:
    return EnvironmentInfo(
        python_version="3.12.0",
        platform="linux",
        packages={"transformers": transformers, "torch": "2.7.0"},
        torch_version="2.7.0",
    )


def _record(
    run_id: str,
    *,
    perplexity: float = 10.0,
    accuracy: float = 0.80,
    stderr: float | None = 0.01,
    dataset_fingerprint: str = "data-abc",
    architecture_fingerprint: str = "arch-1",
    hardware: HardwareInfo | None = None,
    environment: EnvironmentInfo | None = None,
    status: str = "ok",
) -> RunRecord:
    config = RunConfig(
        model=ModelSpec(id="tiny/model"),
        tasks=[PerplexityTask(name="wikitext2", dataset=DatasetSpec(source="text", path="c.txt"))],
    )
    return RunRecord(
        run_id=run_id,
        status=status,
        config=config,
        config_fingerprint=config.fingerprint,
        model=ModelInfo(id="tiny/model", architecture_fingerprint=architecture_fingerprint),
        hardware=hardware or _hardware(),
        environment=environment or _environment(),
        tasks=[
            TaskResult(
                name="wikitext2",
                kind="perplexity",
                dataset_fingerprint=dataset_fingerprint,
                metrics=[
                    MetricValue(
                        name="perplexity",
                        value=perplexity,
                        higher_is_better=False,
                        stderr=stderr,
                    ),
                    MetricValue(name="acc", value=accuracy, higher_is_better=True, stderr=stderr),
                ],
            )
        ],
    )


def test_identical_runs_retain_everything():
    report = compare_runs(_record("base"), _record("cand"))
    assert report.passed
    assert all(c.retention == pytest.approx(1.0) for c in report.comparisons)


def test_rising_perplexity_is_a_regression():
    """Perplexity is lower-is-better: 10 -> 12.5 keeps 80% of quality."""
    report = compare_runs(_record("base", perplexity=10.0), _record("cand", perplexity=12.5))
    ppl = next(c for c in report.comparisons if c.metric == "perplexity")

    assert ppl.retention == pytest.approx(0.8)
    assert ppl.improved is False
    assert ppl.verdict == "fail"
    assert not report.passed


def test_falling_perplexity_is_an_improvement():
    report = compare_runs(_record("base", perplexity=10.0), _record("cand", perplexity=8.0))
    ppl = next(c for c in report.comparisons if c.metric == "perplexity")
    assert ppl.retention == pytest.approx(1.25)
    assert ppl.improved is True
    assert ppl.verdict == "pass"


def test_falling_accuracy_is_a_regression():
    """Accuracy is higher-is-better; retention must point the same way as
    perplexity's does."""
    report = compare_runs(_record("base", accuracy=0.80), _record("cand", accuracy=0.64))
    acc = next(c for c in report.comparisons if c.metric == "acc")
    assert acc.retention == pytest.approx(0.8)
    assert acc.improved is False
    assert acc.verdict == "fail"


def test_threshold_is_respected():
    baseline = _record("base", perplexity=10.0)
    candidate = _record("cand", perplexity=10.4)  # ~96.2% retention

    assert not compare_runs(baseline, candidate, min_retention=0.99).passed
    assert compare_runs(baseline, candidate, min_retention=0.95).passed


def test_metric_filter_narrows_the_check():
    report = compare_runs(_record("base"), _record("cand", perplexity=99.0), metrics=["acc"])
    assert {c.metric for c in report.comparisons} == {"acc"}
    assert report.passed


def test_different_data_blocks_the_comparison():
    """Scoring on different data is the quiet way to reach a wrong conclusion."""
    report = compare_runs(
        _record("base", dataset_fingerprint="data-abc"),
        _record("cand", dataset_fingerprint="data-xyz"),
    )
    assert not report.passed
    assert any(i.level == "error" for i in report.issues)
    assert all(c.verdict == "not_comparable" for c in report.comparisons)


def test_different_hardware_warns_but_does_not_block():
    report = compare_runs(
        _record("base", hardware=_hardware("RTX 4090")),
        _record("cand", hardware=_hardware("A100")),
    )
    assert report.passed
    assert any(i.level == "warning" and "hardware" in i.message for i in report.issues)


def test_library_upgrade_is_reported():
    report = compare_runs(
        _record("base", environment=_environment("4.44.0")),
        _record("cand", environment=_environment("4.50.0")),
    )
    message = next(i.message for i in report.issues if "software stack" in i.message)
    assert "4.44.0 -> 4.50.0" in message


def test_changed_architecture_is_informational():
    """Expected after compression: worth stating, not worth blocking."""
    report = compare_runs(
        _record("base", architecture_fingerprint="arch-1"),
        _record("cand", architecture_fingerprint="arch-2"),
    )
    assert report.passed
    assert any(i.level == "info" for i in report.issues)


def test_failed_run_blocks_the_comparison():
    report = compare_runs(_record("base"), _record("cand", status="failed"))
    assert not report.passed
    assert report.blocking_issues


def test_small_change_within_noise_is_flagged():
    baseline = _record("base", perplexity=10.0, stderr=0.5)
    candidate = _record("cand", perplexity=10.1, stderr=0.5)
    ppl = next(c for c in compare_runs(baseline, candidate).comparisons if c.metric == "perplexity")
    assert ppl.significant is False


def test_large_change_beyond_noise_is_significant():
    baseline = _record("base", perplexity=10.0, stderr=0.01)
    candidate = _record("cand", perplexity=15.0, stderr=0.01)
    ppl = next(c for c in compare_runs(baseline, candidate).comparisons if c.metric == "perplexity")
    assert ppl.significant is True


def test_missing_stderr_leaves_significance_unknown():
    baseline = _record("base", stderr=None)
    candidate = _record("cand", stderr=None)
    assert all(c.significant is None for c in compare_runs(baseline, candidate).comparisons)


def test_missing_task_in_candidate_is_warned():
    baseline = _record("base")
    candidate = _record("cand")
    candidate.tasks = []
    report = compare_runs(baseline, candidate)
    assert any("missing from candidate" in i.message for i in report.issues)
    assert report.comparisons == []


def test_worst_metric_is_reported():
    baseline = _record("base", perplexity=10.0, accuracy=0.80)
    candidate = _record("cand", perplexity=20.0, accuracy=0.79)
    worst = compare_runs(baseline, candidate).worst
    assert worst is not None and worst.metric == "perplexity"
