"""Compression backend integration.

The adapter is tested against a stub runner rather than a real llmcompressor
install: the contract that matters here is the one between AutoDistiller and a
backend subprocess -- job in, JSON artifact out, failures surfaced rather than
swallowed. Whether llmcompressor quantizes correctly is llmcompressor's test
suite, not ours.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from autodistiller.compression.backend import (
    CompressionError,
    CompressionJob,
    LLMCompressorBackend,
    resolve_compression_backend,
)
from autodistiller.compression.methods import (
    METHODS,
    available_methods,
    check_method,
    resolve_method,
)
from autodistiller.compression.pipeline import artifact_dir, build_job
from autodistiller.config import CompressionSpec, DatasetSpec, ModelSpec
from autodistiller.metadata.profiles import resolve_profile

TURING = resolve_profile("rtx-3090")  # sm_8.6, no fp8
ADA = resolve_profile("rtx-4090")  # sm_8.9, fp8 but no fp4
BLACKWELL = resolve_profile("rtx-5090")  # sm_12.0, everything


# --- method vocabulary --------------------------------------------------


def test_every_method_declares_a_scheme_and_capability():
    for method in METHODS.values():
        assert method.scheme
        assert method.required_capability
        assert method.algorithm in {"rtn", "gptq", "awq"}
        assert method.weight_bits <= method.activation_bits or method.activation_bits <= 8


def test_calibrated_algorithms_declare_they_need_calibration():
    """GPTQ and AWQ fit against activations; running them without data is a bug."""
    for method in METHODS.values():
        if method.algorithm in {"gptq", "awq"}:
            assert method.needs_calibration, method.name


def test_fp8_is_rejected_on_ampere():
    result = check_method(resolve_method("fp8"), profile=TURING)
    assert not result.available
    assert "fp8" in result.reasons[0]


def test_fp8_is_allowed_on_ada():
    assert check_method(resolve_method("fp8"), profile=ADA).available


def test_int4_is_allowed_everywhere_modern():
    for profile in (TURING, ADA, BLACKWELL):
        assert check_method(resolve_method("int4-awq"), profile=profile).available


def test_backend_support_is_separate_from_hardware():
    """A method the silicon can run may still be unservable by the runtime.

    Conflating the two produces recommendations that benchmark well and cannot
    be deployed.
    """
    method = resolve_method("int8")
    assert check_method(method, profile=BLACKWELL, backend="vllm").available

    result = check_method(method, profile=BLACKWELL, backend="llama.cpp")
    assert not result.available
    assert "llama.cpp" in result.reasons[0]


def test_unavailable_methods_are_still_listed_with_reasons():
    """Phase 4 must be able to explain what it filtered out."""
    entries = available_methods(profile=TURING, backend="vllm")
    assert len(entries) == len(METHODS)
    rejected = [e for e in entries if not e.available]
    assert rejected and all(e.reasons for e in rejected)


def test_unknown_method_lists_the_options():
    with pytest.raises(KeyError, match="int4-awq"):
        resolve_method("int3-magic")


# --- job construction ---------------------------------------------------


def test_uncalibrated_method_needs_no_dataset():
    job = build_job(ModelSpec(id="tiny/model"), CompressionSpec(method="fp8"))
    assert job.calibration_texts == []
    assert job.recipe().calibration_fingerprint is None
    assert job.recipe().n_calibration_samples == 0


def test_calibrated_method_without_data_fails_early():
    """Better than discovering it minutes into a quantization run."""
    with pytest.raises(ValueError, match="needs calibration data"):
        build_job(ModelSpec(id="tiny/model"), CompressionSpec(method="int4-gptq"))


def test_calibration_texts_are_loaded_and_capped(text_corpus_file: Path, jsonl_corpus_file: Path):
    spec = CompressionSpec(
        method="int4-gptq",
        calibration=DatasetSpec(source="jsonl", path=str(jsonl_corpus_file)),
        num_calibration_samples=3,
    )
    job = build_job(ModelSpec(id="tiny/model"), spec)
    assert len(job.calibration_texts) == 3
    assert job.recipe().n_calibration_samples == 3


def test_calibration_data_is_part_of_recipe_identity(jsonl_corpus_file: Path, tmp_path: Path):
    """Different calibration text produces different weights, so it must produce
    a different recipe fingerprint."""
    other = tmp_path / "other.jsonl"
    other.write_text(json.dumps({"text": "entirely different calibration text"}) + "\n", "utf-8")

    def recipe_for(path: Path):
        return build_job(
            ModelSpec(id="tiny/model"),
            CompressionSpec(
                method="int4-gptq", calibration=DatasetSpec(source="jsonl", path=str(path))
            ),
        ).recipe()

    assert recipe_for(jsonl_corpus_file).calibration_fingerprint is not None
    assert (
        recipe_for(jsonl_corpus_file).calibration_fingerprint
        != recipe_for(other).calibration_fingerprint
    )


def test_recipe_captures_what_was_asked_for():
    job = CompressionJob(
        model_id="tiny/model",
        method=resolve_method("int4-awq"),
        output_dir=Path("out"),
        calibration_texts=["a", "b"],
    )
    recipe = job.recipe()
    assert recipe.method == "int4-awq"
    assert recipe.scheme == "W4A16"
    assert recipe.algorithm == "awq"
    assert recipe.weight_bits == 4
    assert recipe.ignore == ["lm_head"]
    assert recipe.describe() == "W4A16"


def test_artifact_dir_is_derived_from_model_and_method():
    path = artifact_dir("Qwen/Qwen3-0.6B", "int4-awq", Path("artifacts"))
    assert path.name == "Qwen3-0.6B-int4-awq"


def test_explicit_output_dir_wins(tmp_path: Path):
    spec = CompressionSpec(method="fp8", output_dir=tmp_path / "chosen")
    assert build_job(ModelSpec(id="m"), spec).output_dir == tmp_path / "chosen"


# --- adapter contract, driven against a stub runner ---------------------


def _stub_runner(tmp_path: Path, body: str) -> str:
    """Write a script that stands in for the llmcompressor runner."""
    script = tmp_path / "stub_runner.py"
    script.write_text(body, encoding="utf-8")
    return str(script)


def _backend_with_runner(monkeypatch, script: str) -> LLMCompressorBackend:
    backend = LLMCompressorBackend(python_executable=sys.executable)
    monkeypatch.setattr("autodistiller.compression.backend.RUNNER", Path(script), raising=False)
    return backend


def _job(tmp_path: Path) -> CompressionJob:
    return CompressionJob(
        model_id="tiny/model",
        method=resolve_method("fp8"),
        output_dir=tmp_path / "artifact",
        calibration_texts=[],
    )


def test_adapter_parses_a_successful_result(tmp_path: Path, monkeypatch):
    script = _stub_runner(
        tmp_path,
        "import json,sys\n"
        "job = json.load(sys.stdin)\n"
        "json.dump({'ok': True, 'output_dir': job['output_dir'], 'artifact_bytes': 4242,\n"
        "           'duration_s': 1.5, 'versions': {'llmcompressor': '0.13.0'}}, sys.stdout)\n",
    )
    backend = _backend_with_runner(monkeypatch, script)
    artifact = backend.compress(_job(tmp_path))

    assert artifact.artifact_bytes == 4242
    assert artifact.backend == "llmcompressor"
    assert artifact.versions["llmcompressor"] == "0.13.0"
    assert artifact.recipe.method == "fp8"


def test_adapter_sends_the_job_the_backend_needs(tmp_path: Path, monkeypatch):
    """The subprocess boundary is where a wrong field goes unnoticed."""
    captured = tmp_path / "captured.json"
    script = _stub_runner(
        tmp_path,
        "import json,sys\n"
        "job = json.load(sys.stdin)\n"
        f"open({str(captured)!r}, 'w').write(json.dumps(job))\n"
        "json.dump({'ok': True, 'artifact_bytes': 1}, sys.stdout)\n",
    )
    backend = _backend_with_runner(monkeypatch, script)
    job = CompressionJob(
        model_id="tiny/model",
        method=resolve_method("int4-gptq"),
        output_dir=tmp_path / "artifact",
        calibration_texts=["calibrate me"],
        max_seq_length=512,
    )
    backend.compress(job)

    sent = json.loads(captured.read_text(encoding="utf-8"))
    assert sent["scheme"] == "W4A16"
    assert sent["algorithm"] == "gptq"
    assert sent["calibration_texts"] == ["calibrate me"]
    assert sent["max_seq_length"] == 512
    assert sent["ignore"] == ["lm_head"]


def test_adapter_raises_on_backend_failure(tmp_path: Path, monkeypatch):
    script = _stub_runner(
        tmp_path,
        "import json,sys\n"
        "json.dump({'ok': False, 'error': 'CUDA out of memory'}, sys.stdout)\n"
        "sys.exit(1)\n",
    )
    backend = _backend_with_runner(monkeypatch, script)
    with pytest.raises(CompressionError, match="CUDA out of memory"):
        backend.compress(_job(tmp_path))


def test_adapter_reports_a_crash_that_produced_no_json(tmp_path: Path, monkeypatch):
    """A backend that dies before writing a result must not look like success."""
    script = _stub_runner(tmp_path, "import sys\nsys.stderr.write('segfault\\n')\nsys.exit(3)\n")
    backend = _backend_with_runner(monkeypatch, script)
    with pytest.raises(CompressionError, match="produced no result"):
        backend.compress(_job(tmp_path))


def test_adapter_tolerates_noise_on_stdout(tmp_path: Path, monkeypatch):
    """Dependencies print. The result is still the last JSON object."""
    script = _stub_runner(
        tmp_path,
        "import json,sys\n"
        "print('loading checkpoint shards: 100%')\n"
        "json.dump({'ok': True, 'artifact_bytes': 7}, sys.stdout)\n",
    )
    backend = _backend_with_runner(monkeypatch, script)
    assert backend.compress(_job(tmp_path)).artifact_bytes == 7


def test_adapter_honours_a_timeout(tmp_path: Path, monkeypatch):
    script = _stub_runner(tmp_path, "import time\ntime.sleep(30)\n")
    backend = LLMCompressorBackend(python_executable=sys.executable, timeout_s=1)
    monkeypatch.setattr("autodistiller.compression.backend.RUNNER", Path(script), raising=False)
    with pytest.raises(CompressionError, match="timed out"):
        backend.compress(_job(tmp_path))


def test_backend_registry():
    assert isinstance(resolve_compression_backend("llmcompressor"), LLMCompressorBackend)
    with pytest.raises(KeyError, match="llmcompressor"):
        resolve_compression_backend("bitsandbytes")


def test_backend_reports_availability():
    usable, detail = LLMCompressorBackend(python_executable=sys.executable).available()
    assert usable
    assert sys.executable in detail


# --- runner script, in isolation ----------------------------------------


def test_runner_reports_bad_json_without_traceback():
    """The runner's contract is JSON out, always."""
    import subprocess

    from autodistiller.compression.backend import RUNNER

    completed = subprocess.run(
        [sys.executable, str(RUNNER)], input="not json", capture_output=True, text=True
    )
    payload = json.loads(completed.stdout)
    assert payload["ok"] is False
    assert "invalid job JSON" in payload["error"]


def test_runner_rejects_an_unknown_algorithm():
    import subprocess

    from autodistiller.compression.backend import RUNNER

    job = {"model_id": "m", "output_dir": "o", "scheme": "W4A16", "algorithm": "voodoo"}
    completed = subprocess.run(
        [sys.executable, str(RUNNER)], input=json.dumps(job), capture_output=True, text=True
    )
    payload = json.loads(completed.stdout)
    assert payload["ok"] is False
    assert "voodoo" in payload["error"]


def test_schemes_are_real_preset_names():
    """Scheme strings are passed straight to the backend, which rejects
    anything it does not recognize.

    An invented name (FP8_WEIGHT) shipped once and only failed at run time,
    minutes into a compression job. compressed-tensors is a dependency, so the
    valid set is checkable here instead.
    """
    preset = pytest.importorskip("compressed_tensors.quantization.quant_scheme")
    valid = set(preset.PRESET_SCHEMES)

    for method in METHODS.values():
        if method.is_gguf:
            continue
        assert method.scheme in valid, (
            f"{method.name} declares scheme {method.scheme!r}, "
            f"which is not a preset. Valid: {', '.join(sorted(valid))}"
        )


def test_gguf_schemes_are_real_llama_quantize_types():
    """The same guard for the other family.

    A GGUF scheme is passed straight to `llama-quantize` as its type argument,
    and an invented one fails after the convert step has already run -- which is
    the whole model converted for nothing. llama.cpp's own file-type enum ships
    in the `gguf` package, so the valid set is checkable here rather than at the
    end of a long job.
    """
    file_type = pytest.importorskip("gguf").LlamaFileType
    valid = {name.removeprefix("MOSTLY_") for name in (t.name for t in file_type)}

    for method in METHODS.values():
        if not method.is_gguf:
            continue
        assert method.scheme in valid, (
            f"{method.name} declares scheme {method.scheme!r}, which llama-quantize "
            f"does not accept. Valid: {', '.join(sorted(valid))}"
        )


def test_cuda_index_is_configured_for_the_isolated_environment():
    """`uv run` has no --torch-backend flag and ignores UV_TORCH_BACKEND, so
    without an explicit index the isolated environment silently resolves a
    CPU-only torch and AWQ cannot run at all."""
    backend = LLMCompressorBackend()
    command = backend._command()
    assert "--index" in command
    assert any("download.pytorch.org" in part for part in command)


def test_an_explicit_interpreter_skips_the_index():
    command = LLMCompressorBackend(python_executable=sys.executable)._command()
    assert "--index" not in command
    assert command[0] == sys.executable


# --- model kind ---------------------------------------------------------


def test_awq_is_refused_on_an_encoder():
    """Measured, not assumed.

    AWQ smooths activations through per-architecture mappings and
    llmcompressor registers none for encoders, so it falls back to Llama-shaped
    names, matches nothing, and divides by zero. Refusing here costs a config
    read; discovering it in the backend costs a calibration pass first.
    """
    from autodistiller.architecture import DECODER, ENCODER

    awq = resolve_method("int4-awq")

    assert awq.applies_to(DECODER)
    assert not awq.applies_to(ENCODER)

    refused = check_method(awq, model_kind=ENCODER)
    assert not refused.available
    assert any("encoder" in reason for reason in refused.reasons)


def test_the_methods_an_encoder_can_actually_produce():
    """The four that were run against a real BERT checkpoint and worked."""
    from autodistiller.architecture import ENCODER

    usable = {
        availability.method.name
        for availability in available_methods(model_kind=ENCODER)
        if availability.available
    }

    assert usable == {"int8", "int8-weight-only", "int4-gptq", "fp8", "fp8-static"}


def test_model_kind_defaults_to_decoder_when_unstated():
    """Skipping the check is not the same as failing it.

    An unreadable or hand-written config has no answer, and refusing every
    method on that basis would break every local checkpoint.
    """
    from autodistiller.architecture import DECODER, model_kind

    assert model_kind(None) == DECODER
    assert model_kind([]) == DECODER
    assert model_kind(["BertModel"]) == "encoder"
    assert model_kind(["Qwen3ForCausalLM"]) == DECODER

    assert check_method(resolve_method("int4-awq")).available


def test_an_encoder_recipe_does_not_claim_to_ignore_an_lm_head():
    """The recipe is what the artifact record carries.

    A BERT model has no ``lm_head``, so a recipe naming one describes a step
    that never happened -- and since the recipe is part of the artifact's
    identity, it would also key two different models to the same shape.
    """
    from unittest import mock

    spec = CompressionSpec(method="int8-weight-only")

    with mock.patch("autodistiller.compression.pipeline.model_kind_of", return_value="encoder"):
        encoder_job = build_job(ModelSpec(id="BAAI/bge-small-en-v1.5"), spec)
    with mock.patch("autodistiller.compression.pipeline.model_kind_of", return_value="decoder"):
        decoder_job = build_job(ModelSpec(id="Qwen/Qwen3-0.6B"), spec)

    assert encoder_job.ignore == ()
    assert encoder_job.recipe().ignore == []
    assert decoder_job.recipe().ignore == ["lm_head"]


def test_a_float32_checkpoint_is_compressed_at_sixteen_bits():
    """Quantization leaves embeddings and the output head alone.

    So a float32 source writes those tensors back out at 32 bits, and every
    serving runtime then downcasts them at load. Measured on bge-small-en-v1.5:
    70.7 MB written against the 45.1 MB anything would actually hold, an
    artifact 57% larger than the memory estimate for no benefit. Loading at
    bfloat16 brought it to 46.4 MB with the STS-B score unmoved.
    """
    import importlib.util
    from pathlib import Path

    import autodistiller.compression as compression

    # Located through the package rather than a path relative to the working
    # directory: the runner is not importable (it must not import AutoDistiller),
    # but where it lives is not a guess.
    runner_path = Path(compression.__file__).with_name("_runner.py")
    spec = importlib.util.spec_from_file_location("_ad_runner", runner_path)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)

    class Config:
        def __init__(self, dtype):
            self.dtype = dtype

    assert runner._resolve_dtype({"dtype": "auto"}, Config("torch.float32")) == "bfloat16"
    # A checkpoint already at 16 bits is followed, not re-specified.
    assert runner._resolve_dtype({"dtype": "auto"}, Config("torch.bfloat16")) == "auto"
    assert runner._resolve_dtype({"dtype": "auto"}, Config("torch.float16")) == "auto"
    # Declaring nothing is not "leave it alone": Transformers loads float32 when
    # nobody says otherwise, which is how bert-base-uncased -- and every
    # checkpoint of its era -- kept writing 32-bit tensors.
    assert runner._resolve_dtype({"dtype": "auto"}, Config(None)) == "bfloat16"
    # And asking for one is asking for it.
    assert runner._resolve_dtype({"dtype": "float32"}, Config("torch.float32")) == "float32"
