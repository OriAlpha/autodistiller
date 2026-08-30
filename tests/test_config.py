"""The config contract: identical configs must hash identically, and cosmetic
fields must not affect the hash. Phase 6's cache correctness rests on this.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from autodistiller.config import (
    DatasetSpec,
    ModelSpec,
    MultipleChoiceTask,
    PerplexityTask,
    RunConfig,
)


def _config(**overrides) -> RunConfig:
    base = {
        "model": ModelSpec(id="tiny/model"),
        "tasks": [
            PerplexityTask(name="ppl", dataset=DatasetSpec(source="text", path="corpus.txt"))
        ],
    }
    return RunConfig(**{**base, **overrides})


def test_fingerprint_is_stable_across_instances():
    assert _config().fingerprint == _config().fingerprint


def test_fingerprint_ignores_label_and_output_dir():
    reference = _config().fingerprint
    assert _config(label="baseline run").fingerprint == reference
    assert _config(output_dir=Path("/somewhere/else")).fingerprint == reference


def test_fingerprint_changes_with_anything_that_affects_results():
    reference = _config().fingerprint
    assert _config(seed=99).fingerprint != reference
    assert _config(model=ModelSpec(id="tiny/model", dtype="float16")).fingerprint != reference

    altered = PerplexityTask(
        name="ppl", dataset=DatasetSpec(source="text", path="corpus.txt"), stride=128
    )
    assert _config(tasks=[altered]).fingerprint != reference


def test_duplicate_task_names_are_rejected():
    task = PerplexityTask(name="ppl", dataset=DatasetSpec(source="text", path="a.txt"))
    other = PerplexityTask(name="ppl", dataset=DatasetSpec(source="text", path="b.txt"))
    with pytest.raises(ValidationError, match="duplicate task name"):
        RunConfig(model=ModelSpec(id="m"), tasks=[task, other])


def test_unknown_keys_are_rejected():
    """A typo in a hand-written YAML config should fail loudly, not silently."""
    with pytest.raises(ValidationError):
        ModelSpec(id="m", dtpye="float16")


def test_yaml_roundtrip_preserves_fingerprint(tmp_path: Path):
    config = _config(
        tasks=[
            PerplexityTask(name="ppl", dataset=DatasetSpec(source="text", path="corpus.txt")),
            MultipleChoiceTask(
                name="mc",
                dataset=DatasetSpec(source="jsonl", path="evals.jsonl"),
                preprocessor=None,
            ),
        ],
        label="ignored",
    )
    path = config.save(tmp_path / "run.yaml")
    restored = RunConfig.from_yaml(path)

    assert restored.fingerprint == config.fingerprint
    assert [t.kind for t in restored.tasks] == ["perplexity", "multiple_choice"]


def test_task_union_discriminates_on_kind(tmp_path: Path):
    """Round-tripping must preserve the concrete task type, not just its fields."""
    config = _config(
        tasks=[
            MultipleChoiceTask(name="mc", dataset=DatasetSpec(source="jsonl", path="evals.jsonl"))
        ]
    )
    restored = RunConfig.from_yaml(config.save(tmp_path / "run.yaml"))
    assert isinstance(restored.tasks[0], MultipleChoiceTask)


def test_config_accepts_dataset_ids_it_cannot_load():
    """Validation must stay permissive.

    Stored run records embed the config that produced them, so a rule that
    rejects an old value would make historical records unreadable. Whether a
    dataset can actually load is a pre-flight concern, not a schema concern.
    """
    spec = DatasetSpec(source="hub", path="wikitext", name="wikitext-2-raw-v1")
    assert spec.path == "wikitext"


def test_evaluation_key_ignores_the_deployment_settings() -> None:
    """The two cache keys are only separable if one cannot invalidate the other.

    A perplexity number does not depend on the concurrency sweep, so changing
    it must not discard the measurement.
    """
    from autodistiller.config import DeploymentSpec

    def config(levels: list[int] | None) -> RunConfig:
        return RunConfig(
            model=ModelSpec(id="dummy/model"),
            tasks=[PerplexityTask(name="ppl", dataset=DatasetSpec(source="hub", path="wikitext"))],
            deployment=DeploymentSpec(concurrency_levels=levels) if levels else None,
        )

    sweep_a, sweep_b, no_deployment = config([1, 4]), config([1, 4, 16]), config(None)

    assert sweep_a.evaluation_fingerprint == sweep_b.evaluation_fingerprint
    assert sweep_a.fingerprint != sweep_b.fingerprint  # the config itself did change

    # Neutralized, not dropped: keys already on disk came from configs with no
    # deployment section, and must keep matching.
    assert no_deployment.evaluation_fingerprint == no_deployment.fingerprint
