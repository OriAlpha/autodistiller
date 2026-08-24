"""The persistent experiment cache.

What matters is not that the cache is fast but that it is *right*: it must
reuse a result only when reusing it is honest, and must not reuse one when
anything that could have moved the number has changed. Most of these tests are
about the second half.
"""

from __future__ import annotations

import json

import pytest

from autodistiller.cache import benchmark_key, experiment_key
from autodistiller.compression.backend import CompressionJob
from autodistiller.compression.methods import resolve_method
from autodistiller.compression.pipeline import (
    ARTIFACT_SIDECAR,
    artifact_dir,
    build_job,
    read_cached_artifact,
    run_compression,
    write_artifact_sidecar,
)
from autodistiller.config import (
    BaselineInferenceSpec,
    CompressionSpec,
    DatasetSpec,
    ModelSpec,
    PerplexityTask,
    RunConfig,
)
from autodistiller.metadata.environment import EnvironmentInfo
from autodistiller.metadata.hardware import GPUInfo, HardwareInfo
from autodistiller.results import (
    CompressionArtifact,
    ModelInfo,
    RunRecord,
    TaskResult,
)
from autodistiller.runner import run_evaluation
from autodistiller.store import RunStore


def _hardware(name: str = "RTX 5070", vram: int = 8 * 1024**3) -> HardwareInfo:
    return HardwareInfo(
        hostname="host",
        os="Windows",
        cpu="x86",
        accelerator="cuda",
        gpus=[GPUInfo(index=0, name=name, total_memory_bytes=vram, compute_capability="12.0")],
    )


def _environment(**overrides) -> EnvironmentInfo:
    packages = {"autodistiller": "0.3.0", "torch": "2.9.0", "transformers": "5.15.0"}
    packages.update(overrides.pop("packages", {}))
    return EnvironmentInfo(
        python_version=overrides.pop("python_version", "3.12.4"),
        platform=overrides.pop("platform", "Windows-11"),
        packages=packages,
        cuda_version=overrides.pop("cuda_version", "12.8"),
    )


# --- what the key covers ------------------------------------------------


def test_same_inputs_give_the_same_key():
    a = experiment_key("cfg", _hardware(), _environment())
    b = experiment_key("cfg", _hardware(), _environment())
    assert a == b


def test_a_different_gpu_is_a_different_experiment():
    """The roadmap requires hardware in the key. A perplexity number is portable
    but a VRAM or throughput number measured beside it is not."""
    assert experiment_key("cfg", _hardware("RTX 5070"), _environment()) != experiment_key(
        "cfg", _hardware("RTX 4090"), _environment()
    )


def test_a_torch_upgrade_invalidates_the_cache():
    assert experiment_key("cfg", _hardware(), _environment()) != experiment_key(
        "cfg", _hardware(), _environment(packages={"torch": "2.10.0"})
    )


def test_a_cuda_change_invalidates_the_cache():
    assert experiment_key("cfg", _hardware(), _environment()) != experiment_key(
        "cfg", _hardware(), _environment(cuda_version="13.0")
    )


def test_unrelated_packages_do_not_invalidate_the_cache():
    """Keying on every installed version is defensible and useless: a patch bump
    in a package that cannot move a metric would discard everything."""
    assert experiment_key("cfg", _hardware(), _environment()) == experiment_key(
        "cfg", _hardware(), _environment(packages={"safetensors": "9.9.9"})
    )


def test_the_os_build_does_not_invalidate_the_cache():
    assert experiment_key("cfg", _hardware(), _environment()) == experiment_key(
        "cfg", _hardware(), _environment(platform="Windows-11-10.0.26200")
    )


def test_a_python_patch_release_does_not_invalidate_the_cache():
    assert experiment_key("cfg", _hardware(), _environment()) == experiment_key(
        "cfg", _hardware(), _environment(python_version="3.12.9")
    )


def test_a_python_minor_release_does():
    assert experiment_key("cfg", _hardware(), _environment()) != experiment_key(
        "cfg", _hardware(), _environment(python_version="3.13.0")
    )


def test_benchmark_key_separates_request_shapes():
    common = dict(
        served_model="artifacts/m-fp8",
        backend="vllm",
        hardware=_hardware(),
        environment=_environment(),
    )
    assert benchmark_key(**common, settings={"max_tokens": 128}) != benchmark_key(
        **common, settings={"max_tokens": 512}
    )


def test_benchmark_key_separates_served_models():
    common = dict(
        backend="vllm", hardware=_hardware(), environment=_environment(), settings={"n": 1}
    )
    assert benchmark_key(served_model="a", **common) != benchmark_key(served_model="b", **common)


# --- the store index ----------------------------------------------------


def _record(store: RunStore, run_id: str, **overrides) -> RunRecord:
    config = RunConfig(model=ModelSpec(id="tiny/model"), output_dir=store.root)
    record = RunRecord(
        run_id=run_id,
        config=config,
        config_fingerprint=config.fingerprint,
        model=ModelInfo(id="tiny/model"),
        hardware=_hardware(),
        environment=_environment(),
        tasks=[TaskResult(name="wikitext2", kind="perplexity")],
        **overrides,
    )
    store.save(record)
    return record


def test_a_saved_record_is_findable_by_its_key(tmp_path):
    store = RunStore(tmp_path)
    _record(store, "run-a", experiment_key="key-1")
    assert store.find_experiment("key-1").run_id == "run-a"


def test_an_unknown_key_misses(tmp_path):
    store = RunStore(tmp_path)
    _record(store, "run-a", experiment_key="key-1")
    assert store.find_experiment("key-2") is None


def test_records_without_a_key_are_never_a_hit(tmp_path):
    """Runs predating the cache carry no key. Matching None against None would
    hand back an arbitrary old run."""
    store = RunStore(tmp_path)
    _record(store, "run-old")
    assert store.find_experiment(None) is None
    assert store.find_experiment("") is None


def test_a_failed_run_is_not_reused(tmp_path):
    store = RunStore(tmp_path)
    _record(store, "run-bad", experiment_key="key-1", status="failed", error="boom")
    assert store.find_experiment("key-1") is None


def test_the_newest_matching_run_wins(tmp_path):
    store = RunStore(tmp_path)
    _record(store, "run-a", experiment_key="key-1")
    _record(store, "run-b", experiment_key="key-1")
    assert store.find_experiment("key-1").run_id == "run-b"


def test_an_evaluation_key_does_not_answer_a_benchmark_lookup(tmp_path):
    """A record holding only quality metrics cannot serve a throughput
    question, however well its key matches."""
    store = RunStore(tmp_path)
    _record(store, "run-a", experiment_key="key-1", benchmark_key="key-1")
    assert store.find_benchmark("key-1") is None


def test_the_index_is_written_and_reused(tmp_path):
    store = RunStore(tmp_path)
    _record(store, "run-a", experiment_key="key-1")

    rows = [json.loads(line) for line in store.index_path.read_text().splitlines()]
    assert [row["run_id"] for row in rows] == ["run-a"]

    # A fresh store reads the index rather than every record.
    assert RunStore(tmp_path).find_experiment("key-1").run_id == "run-a"


def test_the_index_rebuilds_when_it_is_missing(tmp_path):
    store = RunStore(tmp_path)
    _record(store, "run-a", experiment_key="key-1")
    store.index_path.unlink()

    assert RunStore(tmp_path).find_experiment("key-1").run_id == "run-a"
    assert store.index_path.exists()


def test_resaving_a_record_replaces_its_index_row(tmp_path):
    """The optimizer attaches a benchmark to a record the evaluator already
    wrote. A second row would answer lookups with the older, keyless version."""
    store = RunStore(tmp_path)
    record = _record(store, "run-a", experiment_key="key-1")

    record.benchmark_key = "bench-1"
    store.save(record)

    rows = [json.loads(line) for line in store.index_path.read_text().splitlines()]
    assert len(rows) == 1
    assert rows[0]["benchmark_key"] == "bench-1"


def test_an_unreadable_record_does_not_force_a_rebuild_every_lookup(tmp_path):
    """A bad file on disk must not make the index look permanently short of the
    directory listing, or every lookup pays for a full rescan."""
    store = RunStore(tmp_path)
    _record(store, "run-a", experiment_key="key-1")
    (tmp_path / "run-broken").mkdir()
    (tmp_path / "run-broken" / "record.json").write_text("{not json", encoding="utf-8")

    rebuilt = RunStore(tmp_path)
    assert rebuilt.find_experiment("key-1").run_id == "run-a"

    rows = [json.loads(line) for line in store.index_path.read_text().splitlines()]
    assert len(rows) == 2
    assert {row["run_id"] for row in rows} == {"run-a", "run-broken"}

    # Now a store reading that index must trust it rather than rebuild.
    fresh = RunStore(tmp_path)
    fresh.rebuild_index = lambda: pytest.fail("rebuilt an index it should have trusted")
    assert fresh.find_experiment("key-1").run_id == "run-a"


def test_an_index_row_whose_directory_vanished_is_skipped(tmp_path):
    import shutil

    store = RunStore(tmp_path)
    _record(store, "run-a", experiment_key="key-1")
    _record(store, "run-b", experiment_key="key-2")
    shutil.rmtree(tmp_path / "run-b")

    fresh = RunStore(tmp_path)
    assert fresh.find_experiment("key-2") is None
    assert fresh.find_experiment("key-1").run_id == "run-a"


def test_summaries_are_newest_first(tmp_path):
    store = RunStore(tmp_path)
    _record(store, "run-a", experiment_key="key-1")
    _record(store, "run-b", experiment_key="key-2")
    assert [row["run_id"] for row in store.summaries()] == ["run-b", "run-a"]


# --- artifact identity --------------------------------------------------


def _job(tmp_path, *, method: str = "int4-gptq", texts: list[str] | None = None) -> CompressionJob:
    return CompressionJob(
        model_id="org/Model",
        method=resolve_method(method),
        output_dir=tmp_path / "out",
        calibration_texts=texts if texts is not None else ["alpha", "beta"],
    )


def test_calibration_data_changes_the_artifact_key(tmp_path):
    """Same model, same method, different calibration text: genuinely different
    weights, and so a different artifact."""
    assert (
        _job(tmp_path, texts=["alpha"]).artifact_key != _job(tmp_path, texts=["gamma"]).artifact_key
    )


def test_the_model_changes_the_artifact_key(tmp_path):
    a = _job(tmp_path)
    b = _job(tmp_path)
    b.model_id = "other/Model"
    assert a.artifact_key != b.artifact_key


def test_the_ignore_list_changes_the_artifact_key(tmp_path):
    a = _job(tmp_path)
    b = _job(tmp_path)
    b.ignore = ("lm_head", "model.layers.0")
    assert a.artifact_key != b.artifact_key


def test_artifact_directories_do_not_collide_across_calibration_sets(tmp_path):
    """The bug the key exists to prevent: two different artifacts written to one
    directory, the second silently replacing the first while records already
    written still point at the path."""
    model = ModelSpec(id="org/Model")

    def directory(path: str, text: str) -> str:
        corpus = tmp_path / path
        corpus.write_text(text, encoding="utf-8")
        spec = CompressionSpec(
            method="int4-gptq",
            calibration=DatasetSpec(source="text", path=str(corpus)),
        )
        return str(build_job(model, spec, output_root=tmp_path).output_dir)

    assert directory("a.txt", "one calibration corpus") != directory(
        "b.txt", "an entirely different corpus"
    )


def test_an_explicit_output_dir_is_honoured(tmp_path):
    spec = CompressionSpec(method="fp8", output_dir=tmp_path / "chosen")
    job = build_job(ModelSpec(id="org/Model"), spec, output_root=tmp_path)
    assert job.output_dir == tmp_path / "chosen"


def test_artifact_dir_without_a_key_keeps_the_readable_name(tmp_path):
    assert artifact_dir("org/Model", "fp8", tmp_path).name == "Model-fp8"
    assert artifact_dir("org/Model", "fp8", tmp_path, key="abcdef1234").name == "Model-fp8-abcdef12"


# --- artifact reuse -----------------------------------------------------


def _artifact(job: CompressionJob) -> CompressionArtifact:
    return CompressionArtifact(
        recipe=job.recipe(),
        backend="llmcompressor",
        source_model=job.model_id,
        output_dir=str(job.output_dir),
        artifact_bytes=1234,
    )


def _lay_down_weights(job: CompressionJob) -> None:
    job.output_dir.mkdir(parents=True, exist_ok=True)
    (job.output_dir / "config.json").write_text("{}", encoding="utf-8")
    (job.output_dir / "model.safetensors").write_bytes(b"weights")


def test_a_complete_artifact_is_reused(tmp_path):
    job = _job(tmp_path)
    _lay_down_weights(job)
    write_artifact_sidecar(job, _artifact(job))

    cached = read_cached_artifact(job)
    assert cached is not None
    assert cached.artifact_bytes == 1234


def test_a_sidecar_without_weights_is_not_reused(tmp_path):
    """A crashed run leaves the directory and the config behind. Reusing that
    fails much later and much less clearly."""
    job = _job(tmp_path)
    job.output_dir.mkdir(parents=True)
    (job.output_dir / "config.json").write_text("{}", encoding="utf-8")
    write_artifact_sidecar(job, _artifact(job))

    assert read_cached_artifact(job) is None


def test_an_empty_directory_is_not_reused(tmp_path):
    job = _job(tmp_path)
    job.output_dir.mkdir(parents=True)
    assert read_cached_artifact(job) is None


def test_a_foreign_artifact_in_a_chosen_directory_is_refused(tmp_path):
    """Only reachable through an explicit --output-dir, since derived paths
    carry the key. Recompressing would overwrite weights the user put there."""
    job = _job(tmp_path, texts=["alpha"])
    _lay_down_weights(job)
    write_artifact_sidecar(job, _artifact(job))

    other = _job(tmp_path, texts=["gamma"])
    other.output_dir = job.output_dir

    with pytest.raises(ValueError, match="different artifact"):
        read_cached_artifact(other)


def test_an_unreadable_sidecar_is_ignored_rather_than_fatal(tmp_path):
    job = _job(tmp_path)
    _lay_down_weights(job)
    (job.output_dir / ARTIFACT_SIDECAR).write_text("{ truncated", encoding="utf-8")
    assert read_cached_artifact(job) is None


def test_run_compression_reuses_instead_of_calling_the_backend(tmp_path, monkeypatch):
    model = ModelSpec(id="org/Model")
    spec = CompressionSpec(method="fp8")
    job = build_job(model, spec, output_root=tmp_path)
    _lay_down_weights(job)
    write_artifact_sidecar(job, _artifact(job))

    def explode(*args, **kwargs):
        pytest.fail("compressed a model that was already on disk")

    monkeypatch.setattr("autodistiller.compression.pipeline.resolve_compression_backend", explode)

    artifact = run_compression(model, spec, output_root=tmp_path)
    assert artifact.artifact_bytes == 1234


def test_refresh_bypasses_the_artifact_cache(tmp_path, monkeypatch):
    model = ModelSpec(id="org/Model")
    spec = CompressionSpec(method="fp8")
    job = build_job(model, spec, output_root=tmp_path)
    _lay_down_weights(job)
    write_artifact_sidecar(job, _artifact(job))

    called: list[str] = []

    class Backend:
        def available(self):
            return True, "fake"

        def compress(self, job, *, progress=None):
            called.append(job.method.name)
            return _artifact(job)

    monkeypatch.setattr(
        "autodistiller.compression.pipeline.resolve_compression_backend",
        lambda *a, **k: Backend(),
    )

    run_compression(model, spec, output_root=tmp_path, reuse=False)
    assert called == ["fp8"]


# --- end to end through the evaluator -----------------------------------


def _config(tiny_model_dir, text_corpus_file, tmp_path, **overrides) -> RunConfig:
    return RunConfig(
        model=ModelSpec(id=str(tiny_model_dir)),
        tasks=[
            PerplexityTask(
                name="corpus",
                dataset=DatasetSpec(source="text", path=str(text_corpus_file), limit=4),
                max_length=32,
            )
        ],
        baseline_inference=BaselineInferenceSpec(enabled=False),
        output_dir=tmp_path,
        **overrides,
    )


def test_an_identical_evaluation_is_not_run_twice(tiny_model_dir, text_corpus_file, tmp_path):
    config = _config(tiny_model_dir, text_corpus_file, tmp_path)
    store = RunStore(tmp_path)

    first = run_evaluation(config, store=store)
    second = run_evaluation(config, store=store)

    assert second.run_id == first.run_id
    assert len(list(tmp_path.glob("*/record.json"))) == 1


def test_refresh_measures_again(tiny_model_dir, text_corpus_file, tmp_path):
    config = _config(tiny_model_dir, text_corpus_file, tmp_path)
    store = RunStore(tmp_path)

    first = run_evaluation(config, store=store)
    second = run_evaluation(config, store=store, reuse=False)

    assert second.run_id != first.run_id


def test_a_changed_seed_is_a_different_experiment(tiny_model_dir, text_corpus_file, tmp_path):
    store = RunStore(tmp_path)
    first = run_evaluation(_config(tiny_model_dir, text_corpus_file, tmp_path), store=store)
    second = run_evaluation(
        _config(tiny_model_dir, text_corpus_file, tmp_path, seed=99), store=store
    )
    assert second.run_id != first.run_id


def test_a_cosmetic_label_change_still_hits(tiny_model_dir, text_corpus_file, tmp_path):
    """The label is excluded from the config hash, so it must not cost a rerun."""
    store = RunStore(tmp_path)
    first = run_evaluation(_config(tiny_model_dir, text_corpus_file, tmp_path), store=store)
    second = run_evaluation(
        _config(tiny_model_dir, text_corpus_file, tmp_path, label="renamed"), store=store
    )
    assert second.run_id == first.run_id


def test_a_cache_hit_needs_no_dataset(tiny_model_dir, text_corpus_file, tmp_path):
    """The lookup happens before preflight. A cached answer should not depend on
    the dataset still being reachable."""
    corpus = tmp_path / "corpus.txt"
    corpus.write_text(text_corpus_file.read_text(encoding="utf-8"), encoding="utf-8")

    config = _config(tiny_model_dir, corpus, tmp_path)
    store = RunStore(tmp_path)
    first = run_evaluation(config, store=store)

    corpus.unlink()
    assert run_evaluation(config, store=store).run_id == first.run_id


def test_the_stored_record_carries_its_key(tiny_model_dir, text_corpus_file, tmp_path):
    record = run_evaluation(_config(tiny_model_dir, text_corpus_file, tmp_path))
    assert record.experiment_key
    assert record.schema_version >= 2


def test_a_version_one_record_still_loads(tmp_path):
    """Records written before the cache existed must remain readable; they are
    history, just not reusable."""
    store = RunStore(tmp_path)
    record = _record(store, "run-a", experiment_key="key-1")

    payload = json.loads(record.to_json())
    payload["schema_version"] = 1
    for field in ("experiment_key", "benchmark_key", "candidate_id"):
        payload.pop(field)
    (tmp_path / "run-a" / "record.json").write_text(json.dumps(payload), encoding="utf-8")

    reloaded = RunStore(tmp_path).resolve("run-a")
    assert reloaded.experiment_key is None
