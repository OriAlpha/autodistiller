"""Export and reproducibility.

The milestone is that the exported result is deployable and rebuildable, so what
matters is that export *checks* rather than asserts: an artifact missing its
tokenizer, or quantized in a format the target runtime has no kernel for, must
come back as not deployable rather than as a confident manifest.
"""

from __future__ import annotations

import json

import pytest

from autodistiller.compression.pipeline import ARTIFACT_SIDECAR
from autodistiller.config import ModelSpec, RunConfig
from autodistiller.export import (
    CONFIG_FILENAME,
    MANIFEST_FILENAME,
    README_FILENAME,
    ExportManifest,
    build_manifest,
    export,
    gguf_note,
    inspect_artifact,
    read_artifact_sidecar,
    read_quant_method,
    render_readme,
)
from autodistiller.metadata.environment import EnvironmentInfo
from autodistiller.metadata.hardware import GPUInfo, HardwareInfo
from autodistiller.results import (
    CompressionArtifact,
    CompressionRecipe,
    ConcurrencyResult,
    DeploymentBenchmark,
    LatencyStats,
    MetricValue,
    ModelInfo,
    RunRecord,
    TaskResult,
)

GIB = 1024**3


def _hardware() -> HardwareInfo:
    return HardwareInfo(
        hostname="host",
        os="Windows",
        cpu="x86",
        accelerator="cuda",
        gpus=[GPUInfo(index=0, name="RTX 5070", total_memory_bytes=8 * GIB)],
    )


def _environment() -> EnvironmentInfo:
    return EnvironmentInfo(
        python_version="3.12.4",
        platform="Windows-11",
        packages={"transformers": "5.15.1"},
        torch_version="2.11.0+cu128",
    )


def _recipe(method: str = "fp8") -> CompressionRecipe:
    return CompressionRecipe(
        method=method,
        scheme="FP8_DYNAMIC",
        algorithm="rtn",
        weight_bits=8,
        activation_bits=8,
        ignore=["lm_head"],
    )


def _artifact(directory, *, source: str = "Qwen/Qwen3-0.6B") -> CompressionArtifact:
    return CompressionArtifact(
        recipe=_recipe(),
        backend="llmcompressor",
        source_model=source,
        output_dir=str(directory),
        artifact_bytes=700 * 1024**2,
    )


def _lay_down_artifact(directory, *, quant_method: str | None = "compressed-tensors", **files):
    """A directory shaped like something llmcompressor produced."""
    directory.mkdir(parents=True, exist_ok=True)

    config: dict = {"architectures": ["Qwen3ForCausalLM"], "model_type": "qwen3"}
    if quant_method is not None:
        config["quantization_config"] = {"quant_method": quant_method, "format": "float-quantized"}

    if files.get("config", True):
        (directory / "config.json").write_text(json.dumps(config), encoding="utf-8")
    if files.get("weights", True):
        (directory / "model.safetensors").write_bytes(b"weights")
    if files.get("tokenizer", True):
        (directory / "tokenizer.json").write_text("{}", encoding="utf-8")
    return directory


def _record(tmp_path, *, model_id: str, artifact: CompressionArtifact | None = None) -> RunRecord:
    config = RunConfig(model=ModelSpec(id=model_id), output_dir=tmp_path)
    return RunRecord(
        run_id="run-a",
        config=config,
        config_fingerprint=config.fingerprint,
        model=ModelInfo(id=model_id, context_length=2048),
        hardware=_hardware(),
        environment=_environment(),
        tasks=[
            TaskResult(
                name="wikitext2",
                kind="perplexity",
                metrics=[MetricValue(name="perplexity", value=22.7, higher_is_better=False)],
            )
        ],
        compression=artifact,
    )


# --- reading what is on disk --------------------------------------------


def test_quant_method_is_read_from_the_config(tmp_path):
    directory = _lay_down_artifact(tmp_path / "art")
    assert read_quant_method(directory) == "compressed-tensors"


def test_unquantized_weights_report_no_method(tmp_path):
    directory = _lay_down_artifact(tmp_path / "art", quant_method=None)
    assert read_quant_method(directory) is None


def test_a_broken_config_does_not_raise(tmp_path):
    directory = tmp_path / "art"
    directory.mkdir()
    (directory / "config.json").write_text("{ truncated", encoding="utf-8")
    assert read_quant_method(directory) is None


def test_the_sidecar_recovers_the_recipe(tmp_path):
    """Weights produced by `compress` have no run record; the recipe beside them
    is the only place the provenance lives."""
    directory = _lay_down_artifact(tmp_path / "art")
    payload = _artifact(directory).model_dump(mode="json") | {"artifact_key": "abc123"}
    (directory / ARTIFACT_SIDECAR).write_text(json.dumps(payload), encoding="utf-8")

    recovered = read_artifact_sidecar(directory)
    assert recovered is not None
    assert recovered.recipe.method == "fp8"
    assert recovered.source_model == "Qwen/Qwen3-0.6B"


def test_a_missing_sidecar_is_not_an_error(tmp_path):
    assert read_artifact_sidecar(_lay_down_artifact(tmp_path / "art")) is None


# --- deployability checks -----------------------------------------------


def test_a_complete_artifact_passes_every_check(tmp_path):
    checks = inspect_artifact(_lay_down_artifact(tmp_path / "art"))
    assert all(check.ok for check in checks)
    assert {check.name for check in checks} == {"config", "weights", "tokenizer", "format"}


def test_missing_weights_fail(tmp_path):
    checks = inspect_artifact(_lay_down_artifact(tmp_path / "art", weights=False))
    assert not next(c for c in checks if c.name == "weights").ok


def test_a_missing_tokenizer_fails(tmp_path):
    """The server cannot encode without one, and finds out at startup."""
    checks = inspect_artifact(_lay_down_artifact(tmp_path / "art", tokenizer=False))
    assert not next(c for c in checks if c.name == "tokenizer").ok


def test_an_unknown_quantization_format_fails(tmp_path):
    """Benchmarking beautifully and then not being servable is the failure this
    exists to catch."""
    directory = _lay_down_artifact(tmp_path / "art", quant_method="some-new-scheme")
    check = next(c for c in inspect_artifact(directory) if c.name == "format")
    assert not check.ok
    assert "some-new-scheme" in check.detail


def test_unquantized_weights_are_servable(tmp_path):
    directory = _lay_down_artifact(tmp_path / "art", quant_method=None)
    assert next(c for c in inspect_artifact(directory) if c.name == "format").ok


def test_a_directory_that_does_not_exist_fails_rather_than_raises(tmp_path):
    checks = inspect_artifact(tmp_path / "nope")
    assert checks and not checks[0].ok


# --- GGUF, where applicable ---------------------------------------------


def test_gguf_is_refused_for_quantized_weights():
    """llama.cpp converts from unquantized Hugging Face weights, so claiming a
    conversion path here would be wrong."""
    note = gguf_note("compressed-tensors")
    assert "Not applicable" in note
    assert "compressed-tensors" in note


def test_gguf_gives_a_command_for_unquantized_weights():
    note = gguf_note(None)
    assert "convert_hf_to_gguf.py" in note
    assert "llama-quantize" in note


# --- the manifest -------------------------------------------------------


def test_the_manifest_names_the_source_model_not_the_artifact_path(tmp_path):
    """A reader needs to know these weights came from Qwen3-0.6B, not from a
    directory whose name happens to contain a hash."""
    directory = _lay_down_artifact(tmp_path / "art")
    record = _record(tmp_path, model_id=str(directory), artifact=_artifact(directory))

    manifest = build_manifest(record)
    assert manifest.source_model == "Qwen/Qwen3-0.6B"
    assert manifest.artifact_dir == str(directory)


def test_the_manifest_recovers_the_recipe_from_the_sidecar(tmp_path):
    """The record carries no compression artifact, which is the normal case for
    an evaluation of a compressed candidate."""
    directory = _lay_down_artifact(tmp_path / "art")
    payload = _artifact(directory).model_dump(mode="json") | {"artifact_key": "abc123"}
    (directory / ARTIFACT_SIDECAR).write_text(json.dumps(payload), encoding="utf-8")

    manifest = build_manifest(_record(tmp_path, model_id=str(directory)))
    assert manifest.artifact is not None
    assert manifest.artifact.recipe.method == "fp8"


def test_the_serve_command_uses_the_context_length_not_the_output_length(tmp_path):
    """max_tokens is output tokens per request. Serving with --max-model-len 128
    would truncate every prompt."""
    directory = _lay_down_artifact(tmp_path / "art")
    record = _record(tmp_path, model_id=str(directory), artifact=_artifact(directory))
    record.deployment = DeploymentBenchmark(
        backend="vllm",
        endpoint="http://fake",
        served_model=str(directory),
        max_tokens=128,
        phases=[
            ConcurrencyResult(
                concurrency=1,
                n_requests=8,
                duration_s=1.0,
                ttft=LatencyStats(mean=0.05, p50=0.05, p90=0.05, p99=0.05, min=0.05, max=0.05),
                output_tokens_per_s=1000.0,
                peak_vram_bytes=5 * GIB,
            )
        ],
    )

    manifest = build_manifest(record)
    assert "--max-model-len 2048" in manifest.serve_command
    assert "--max-model-len 128" not in manifest.serve_command


def test_the_manifest_carries_the_commands_that_rebuild_it(tmp_path):
    directory = _lay_down_artifact(tmp_path / "art")
    record = _record(tmp_path, model_id=str(directory), artifact=_artifact(directory))

    commands = build_manifest(record).reproduce
    assert any("autodistiller compress" in c and "--method fp8" in c for c in commands)
    assert any("autodistiller evaluate" in c for c in commands)


def test_a_calibrated_recipe_reports_its_calibration_settings(tmp_path):
    """Calibration text changes the weights, so rebuilding without it rebuilds
    something else."""
    directory = _lay_down_artifact(tmp_path / "art")
    artifact = _artifact(directory)
    artifact.recipe = CompressionRecipe(
        method="int4-gptq",
        scheme="W4A16",
        algorithm="gptq",
        weight_bits=4,
        activation_bits=16,
        needs_calibration=True,
        n_calibration_samples=128,
        max_seq_length=2048,
        calibration_fingerprint="deadbeef",
    )
    record = _record(tmp_path, model_id=str(directory), artifact=artifact)

    compress = next(c for c in build_manifest(record).reproduce if "compress" in c)
    assert "--samples 128" in compress
    assert "--max-seq-length 2048" in compress


def test_a_baseline_export_describes_the_source_model(tmp_path):
    """ "Do not compress" is a deployable answer too."""
    manifest = build_manifest(_record(tmp_path, model_id="Qwen/Qwen3-0.6B"))
    assert manifest.source_model == "Qwen/Qwen3-0.6B"
    assert manifest.artifact_dir is None
    assert "Qwen/Qwen3-0.6B" in manifest.serve_command


def test_deployable_reflects_the_checks(tmp_path):
    good = _lay_down_artifact(tmp_path / "good")
    bad = _lay_down_artifact(tmp_path / "bad", tokenizer=False)

    assert build_manifest(_record(tmp_path, model_id=str(good))).deployable
    assert not build_manifest(_record(tmp_path, model_id=str(bad))).deployable


# --- writing the bundle -------------------------------------------------


def test_export_writes_beside_the_weights_by_default(tmp_path):
    """The directory you would serve is the one that explains itself."""
    directory = _lay_down_artifact(tmp_path / "art")
    record = _record(tmp_path, model_id=str(directory), artifact=_artifact(directory))

    manifest, destination = export(record)

    assert destination == directory
    for name in (MANIFEST_FILENAME, README_FILENAME, CONFIG_FILENAME):
        assert (directory / name).is_file()
    assert manifest.deployable


def test_the_written_manifest_round_trips(tmp_path):
    directory = _lay_down_artifact(tmp_path / "art")
    record = _record(tmp_path, model_id=str(directory), artifact=_artifact(directory))
    manifest, destination = export(record)

    reloaded = ExportManifest.load(destination / MANIFEST_FILENAME)
    assert reloaded.source_model == manifest.source_model
    assert reloaded.run_id == manifest.run_id
    assert reloaded.deployable


def test_the_saved_config_reproduces_the_run(tmp_path):
    """Reproducibility is the point: the config beside the weights must hash to
    the same experiment."""
    directory = _lay_down_artifact(tmp_path / "art")
    record = _record(tmp_path, model_id=str(directory), artifact=_artifact(directory))
    _, destination = export(record)

    saved = RunConfig.from_yaml(destination / CONFIG_FILENAME)
    assert saved.fingerprint == record.config_fingerprint


def test_a_bundle_can_be_assembled_elsewhere_without_the_weights(tmp_path):
    directory = _lay_down_artifact(tmp_path / "art")
    record = _record(tmp_path, model_id=str(directory), artifact=_artifact(directory))

    manifest, destination = export(record, output_dir=tmp_path / "bundle")

    assert (destination / MANIFEST_FILENAME).is_file()
    assert not (destination / "model.safetensors").exists()
    # It still points at the weights it did not copy.
    assert manifest.artifact_dir == str(directory)


def test_copying_weights_makes_the_bundle_movable(tmp_path):
    """A bundle that refers to weights it did not bring cannot be moved, so the
    manifest has to describe the copy rather than the original."""
    directory = _lay_down_artifact(tmp_path / "art")
    record = _record(tmp_path, model_id=str(directory), artifact=_artifact(directory))

    manifest, destination = export(record, output_dir=tmp_path / "bundle", copy_weights=True)

    assert (destination / "model.safetensors").is_file()
    assert manifest.artifact_dir == str(destination)
    assert str(destination) in manifest.serve_command


def test_exporting_a_baseline_in_place_is_refused(tmp_path):
    """There is no artifact directory to write into, and silently inventing one
    would put a manifest somewhere nobody looks."""
    with pytest.raises(ValueError, match="output directory"):
        export(_record(tmp_path, model_id="Qwen/Qwen3-0.6B"))


def test_a_baseline_exports_to_a_given_directory(tmp_path):
    manifest, destination = export(
        _record(tmp_path, model_id="Qwen/Qwen3-0.6B"), output_dir=tmp_path / "bundle"
    )
    assert (destination / MANIFEST_FILENAME).is_file()
    assert manifest.artifact is None


# --- the deployment note ------------------------------------------------


def test_the_readme_carries_what_a_deployer_needs(tmp_path):
    directory = _lay_down_artifact(tmp_path / "art")
    record = _record(tmp_path, model_id=str(directory), artifact=_artifact(directory))

    readme = render_readme(build_manifest(record))

    assert "Qwen/Qwen3-0.6B" in readme
    assert "vllm serve" in readme
    assert "wikitext2" in readme
    assert "autodistiller compress" in readme
    assert "RTX 5070" in readme  # numbers move with the stack; name it
    assert "PASS config" in readme


def test_the_readme_says_plainly_when_nothing_was_benchmarked(tmp_path):
    directory = _lay_down_artifact(tmp_path / "art")
    record = _record(tmp_path, model_id=str(directory), artifact=_artifact(directory))
    assert "No deployment benchmark" in render_readme(build_manifest(record))


def test_the_readme_reports_a_failed_check(tmp_path):
    directory = _lay_down_artifact(tmp_path / "art", tokenizer=False)
    readme = render_readme(build_manifest(_record(tmp_path, model_id=str(directory))))
    assert "FAIL tokenizer" in readme
