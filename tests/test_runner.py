"""End-to-end: the full evaluate -> record -> store -> compare loop, run against
a real (tiny) model with no network access.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from autodistiller.cli import app
from autodistiller.config import (
    BaselineInferenceSpec,
    DatasetSpec,
    ModelSpec,
    MultipleChoiceTask,
    PerplexityTask,
    RunConfig,
)
from autodistiller.regression import compare_runs
from autodistiller.results import RunRecord
from autodistiller.runner import preflight, run_evaluation
from autodistiller.store import RunStore, make_run_id


@pytest.fixture
def run_config(tiny_model_dir: Path, text_corpus_file: Path, mc_dataset_file: Path, tmp_path: Path):
    return RunConfig(
        model=ModelSpec(id=str(tiny_model_dir), device="cpu", dtype="float32"),
        tasks=[
            PerplexityTask(
                name="corpus",
                dataset=DatasetSpec(source="text", path=str(text_corpus_file)),
                max_length=64,
                stride=32,
            ),
            MultipleChoiceTask(
                name="evals", dataset=DatasetSpec(source="jsonl", path=str(mc_dataset_file))
            ),
        ],
        baseline_inference=BaselineInferenceSpec(
            prompts=["The quick brown"], max_new_tokens=4, warmup_runs=0
        ),
        output_dir=tmp_path / "runs",
    )


def test_full_run_produces_a_complete_record(run_config: RunConfig):
    record = run_evaluation(run_config)

    assert record.status == "ok"
    assert record.error is None
    assert len(record.tasks) == 2
    assert record.task("corpus").metric("perplexity") is not None
    assert record.task("evals").metric("acc") is not None
    assert record.config_fingerprint == run_config.fingerprint
    assert record.total_duration_s > 0


def test_record_captures_full_provenance(run_config: RunConfig):
    record = run_evaluation(run_config)

    assert record.model.architecture_fingerprint
    assert record.model.n_parameters and record.model.n_parameters > 0
    assert record.environment.packages.get("torch")
    assert record.environment.fingerprint
    assert record.hardware.fingerprint
    assert all(t.dataset_fingerprint for t in record.tasks)


def test_baseline_inference_is_never_a_deployment_claim(run_config: RunConfig):
    """The roadmap forbids presenting Transformers timings as serving numbers."""
    inference = run_evaluation(run_config).baseline_inference

    assert inference is not None
    assert inference.runtime == "transformers"
    assert inference.is_deployment_claim is False
    assert len(inference.samples) == 1
    assert inference.samples[0].n_generated_tokens > 0


def test_run_is_reproducible(run_config: RunConfig):
    """Same config, same machine -> same numbers. Everything else depends on it."""
    first = run_evaluation(run_config, save=False)
    second = run_evaluation(run_config, save=False)

    for a, b in zip(first.tasks, second.tasks, strict=True):
        for metric_a, metric_b in zip(a.metrics, b.metrics, strict=True):
            assert metric_a.value == pytest.approx(metric_b.value, rel=1e-9)


def test_a_failing_task_does_not_abort_the_others(run_config: RunConfig, jsonl_corpus_file: Path):
    """The file exists, so pre-flight passes; the wrong column only surfaces at
    evaluation time. The other tasks must still produce numbers."""
    run_config.tasks.append(
        PerplexityTask(
            name="broken",
            dataset=DatasetSpec(
                source="jsonl", path=str(jsonl_corpus_file), text_column="no_such_column"
            ),
        )
    )
    record = run_evaluation(run_config, save=False)

    assert record.status == "failed"
    assert "broken" in record.error
    assert record.task("broken").metrics == []
    assert "KeyError" in record.task("broken").details["error"]
    # The healthy tasks still produced numbers.
    assert record.task("corpus").metric("perplexity") is not None
    assert record.task("evals").metric("acc") is not None


def test_preflight_blocks_a_run_before_the_model_loads(run_config: RunConfig):
    """Cheap checks first: a mistyped dataset should not cost a model download."""
    run_config.tasks.append(
        PerplexityTask(name="broken", dataset=DatasetSpec(source="text", path="does/not/exist.txt"))
    )
    with pytest.raises(ValueError, match="broken"):
        run_evaluation(run_config, save=False)


def test_preflight_reports_every_problem_at_once(run_config: RunConfig):
    run_config.tasks = [
        PerplexityTask(name="missing_a", dataset=DatasetSpec(source="text", path="no/a.txt")),
        PerplexityTask(name="missing_b", dataset=DatasetSpec(source="text", path="no/b.txt")),
        PerplexityTask(name="bare_hub", dataset=DatasetSpec(source="hub", path="wikitext")),
    ]
    problems = preflight(run_config)
    assert len(problems) == 3
    assert {p.split(":")[0] for p in problems} == {"missing_a", "missing_b", "bare_hub"}


def test_an_archived_record_stays_readable(run_config: RunConfig):
    """Records are historical data. A record written before a rule tightened
    must still load, or the experiment archive rots."""
    record = run_evaluation(run_config)
    store = RunStore(run_config.output_dir)
    path = store.run_dir(record.run_id) / "record.json"

    # Simulate a stored config holding a value later runs would reject.
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["config"]["tasks"][0]["dataset"] = {
        "source": "hub",
        "path": "wikitext",
        "name": "wikitext-2-raw-v1",
        "split": "test",
        "text_column": "text",
        "limit": 64,
    }
    path.write_text(json.dumps(raw), encoding="utf-8")

    reloaded = store.load(record.run_id)
    assert reloaded.config.tasks[0].dataset.path == "wikitext"


def test_run_is_persisted_and_reloadable(run_config: RunConfig):
    record = run_evaluation(run_config)
    store = RunStore(run_config.output_dir)

    reloaded = store.load(record.run_id)
    assert reloaded.run_id == record.run_id
    assert reloaded.model_dump(mode="json") == record.model_dump(mode="json")

    directory = store.run_dir(record.run_id)
    assert (directory / "record.json").exists()
    assert (directory / "config.yaml").exists()

    # The saved config alone must be enough to reproduce the run.
    assert RunConfig.from_yaml(directory / "config.yaml").fingerprint == record.config_fingerprint


def test_store_finds_runs_by_config_fingerprint(run_config: RunConfig):
    record = run_evaluation(run_config)
    found = RunStore(run_config.output_dir).find_by_fingerprint(run_config.fingerprint)
    assert found is not None and found.run_id == record.run_id


def test_store_resolve_accepts_ids_dirs_and_paths(run_config: RunConfig):
    record = run_evaluation(run_config)
    store = RunStore(run_config.output_dir)
    directory = store.run_dir(record.run_id)

    assert store.resolve(record.run_id).run_id == record.run_id
    assert store.resolve(str(directory)).run_id == record.run_id
    assert store.resolve(str(directory / "record.json")).run_id == record.run_id


def test_store_skips_unreadable_records(run_config: RunConfig, tmp_path: Path):
    run_evaluation(run_config)
    store = RunStore(run_config.output_dir)

    broken = store.run_dir("20200101T000000Z_broken_deadbeef")
    broken.mkdir(parents=True)
    (broken / "record.json").write_text("{ not json", encoding="utf-8")

    assert len(store.list_records()) == 1


def test_run_ids_are_unique_per_config(run_config: RunConfig):
    other = run_config.model_copy(update={"seed": 999})
    assert make_run_id(run_config) != make_run_id(other)


def test_a_run_compares_cleanly_against_itself(run_config: RunConfig):
    record = run_evaluation(run_config, save=False)
    report = compare_runs(record, record)

    assert report.passed
    assert report.comparisons
    assert all(c.retention == pytest.approx(1.0) for c in report.comparisons)


# --- CLI ----------------------------------------------------------------


def test_cli_evaluate_and_compare(tiny_model_dir: Path, text_corpus_file: Path, tmp_path: Path):
    runner = CliRunner()
    runs_dir = tmp_path / "cli-runs"

    def evaluate() -> str:
        result = runner.invoke(
            app,
            [
                "evaluate",
                "--model",
                str(tiny_model_dir),
                "--task",
                f"ppl:{text_corpus_file}",
                "--device",
                "cpu",
                "--dtype",
                "float32",
                "--max-length",
                "64",
                "--no-inference",
                "--output-dir",
                str(runs_dir),
            ],
        )
        assert result.exit_code == 0, result.output
        return result.output

    output = evaluate()
    assert "perplexity" in output

    records = RunStore(runs_dir).list_records()
    assert len(records) == 1

    # Comparing a run against itself must pass and exit 0.
    run_id = records[0].run_id
    result = runner.invoke(app, ["compare", run_id, run_id, "--runs-dir", str(runs_dir)])
    assert result.exit_code == 0, result.output
    assert "PASS" in result.output


def test_cli_compare_exits_nonzero_on_regression(tmp_path: Path, run_config: RunConfig):
    """CI needs a non-zero exit code to gate on."""
    record = run_evaluation(run_config)
    store = RunStore(run_config.output_dir)

    degraded = RunRecord.model_validate(record.model_dump())
    degraded.run_id = "degraded"
    for metric in degraded.task("corpus").metrics:
        if metric.name == "perplexity":
            metric.value *= 2.0
    store.save(degraded)

    result = CliRunner().invoke(
        app,
        [
            "compare",
            record.run_id,
            "degraded",
            "--runs-dir",
            str(run_config.output_dir),
            "--metric",
            "perplexity",
        ],
    )
    assert result.exit_code == 1
    assert "FAIL" in result.output


def test_cli_rejects_unknown_task(tiny_model_dir: Path):
    result = CliRunner().invoke(app, ["evaluate", "--model", str(tiny_model_dir), "--task", "nope"])
    assert result.exit_code != 0


def test_cli_requires_a_model():
    result = CliRunner().invoke(app, ["evaluate"])
    assert result.exit_code != 0


def test_cli_env_json_is_machine_readable():
    import json

    result = CliRunner().invoke(app, ["env", "--json"])
    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert "hardware_fingerprint" in payload
    assert "environment_fingerprint" in payload


def test_cli_saves_a_reproducible_config(
    tiny_model_dir: Path, text_corpus_file: Path, tmp_path: Path
):
    config_path = tmp_path / "run.yaml"
    result = CliRunner().invoke(
        app,
        [
            "evaluate",
            "--model",
            str(tiny_model_dir),
            "--task",
            f"ppl:{text_corpus_file}",
            "--device",
            "cpu",
            "--dtype",
            "float32",
            "--max-length",
            "64",
            "--no-inference",
            "--output-dir",
            str(tmp_path / "runs"),
            "--save-config",
            str(config_path),
        ],
    )
    assert result.exit_code == 0, result.output
    assert RunConfig.from_yaml(config_path).model.id == str(tiny_model_dir)
