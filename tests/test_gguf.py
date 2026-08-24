"""The llama.cpp backend.

llama.cpp is not pip installable, so these tests exercise the parts that decide
*what* gets run rather than running it: tool discovery, command construction,
and the backend-specific semantics that keep a GGUF candidate from being
described in vLLM's vocabulary. The subprocess calls themselves are faked, the
same way the llmcompressor adapter is tested.
"""

from __future__ import annotations

import pytest

from autodistiller.candidates.generator import generate_candidates
from autodistiller.candidates.memory import weight_bytes
from autodistiller.compression.backend import CompressionError, CompressionJob
from autodistiller.compression.gguf import (
    ARTIFACT_NAME,
    CONVERT_SCRIPT,
    INTERMEDIATE_NAME,
    LLAMA_CPP_DIR_ENV,
    QUANTIZE_BINARY,
    LlamaCppBackend,
    find_convert_script,
    find_quantize_binary,
)
from autodistiller.compression.methods import METHODS, resolve_method
from autodistiller.compression.pipeline import build_job
from autodistiller.config import CompressionSpec, ModelSpec
from autodistiller.metadata.profiles import resolve_profile
from autodistiller.serving.backends import resolve_backend
from tests.test_candidates import qwen3_06b

BLACKWELL = resolve_profile("rtx-5090")


def _checkout(tmp_path, *, script: bool = True, binary: bool = True):
    """A directory shaped like a built llama.cpp checkout."""
    root = tmp_path / "llama.cpp"
    (root / "build" / "bin").mkdir(parents=True)
    if script:
        (root / CONVERT_SCRIPT).write_text("# converter", encoding="utf-8")
    if binary:
        (root / "build" / "bin" / QUANTIZE_BINARY).write_text("#!/bin/sh", encoding="utf-8")
    return root


def _job(tmp_path, method: str = "gguf-q4-k-m") -> CompressionJob:
    # A local directory, so resolve_local_model has nothing to download.
    source = tmp_path / "source-model"
    source.mkdir(parents=True, exist_ok=True)
    return CompressionJob(
        model_id=str(source),
        method=resolve_method(method),
        output_dir=tmp_path / "out",
        calibration_texts=[],
    )


# --- the method vocabulary ----------------------------------------------


def test_gguf_methods_are_served_only_by_llama_cpp():
    for method in METHODS.values():
        if method.is_gguf:
            assert method.backends == ("llama.cpp",)
            assert not method.servable_by("vllm")


def test_gguf_methods_are_built_by_llama_cpp():
    assert all(m.compression_backend == "llama.cpp" for m in METHODS.values() if m.is_gguf)
    assert all(m.produces == "file" for m in METHODS.values() if m.is_gguf)


def test_gguf_needs_no_tensor_core_format():
    """llama.cpp has its own kernels and runs on CPU as well, so a GGUF method
    must not be filtered out by a capability rule written for vLLM."""
    ampere = resolve_profile("rtx-3090")  # no fp8
    for method in METHODS.values():
        if method.is_gguf:
            assert method.runs_on(ampere)


def test_gguf_quantizes_embeddings_and_compressed_tensors_does_not():
    assert METHODS["gguf-q4-k-m"].quantizes_embeddings
    assert not METHODS["int4-gptq"].quantizes_embeddings


def test_a_k_quant_is_not_its_nominal_width():
    """Q4_K_M is a mix, not four bits everywhere. Sizing it from weight_bits
    would under-estimate every candidate."""
    method = METHODS["gguf-q4-k-m"]
    assert method.weight_bits == 4
    assert method.bits_per_weight > 4.5

    shape = qwen3_06b()
    naive = shape.n_parameters * 4 / 8
    assert weight_bytes(shape, method) > naive


# --- discovery ----------------------------------------------------------


def test_a_built_checkout_is_found(tmp_path):
    root = _checkout(tmp_path)
    assert find_convert_script(str(root)) == root / CONVERT_SCRIPT
    assert find_quantize_binary(str(root)) == root / "build" / "bin" / QUANTIZE_BINARY


def test_the_environment_variable_is_honoured(tmp_path, monkeypatch):
    root = _checkout(tmp_path)
    monkeypatch.setenv(LLAMA_CPP_DIR_ENV, str(root))
    assert find_convert_script() == root / CONVERT_SCRIPT


def test_nothing_found_is_not_an_error(tmp_path, monkeypatch):
    monkeypatch.delenv(LLAMA_CPP_DIR_ENV, raising=False)
    assert find_convert_script(str(tmp_path / "nope")) is None


def test_availability_names_what_is_missing(tmp_path, monkeypatch):
    """llama.cpp is not pip installable, so "unavailable" has to say which half
    is absent rather than failing inside a subprocess later."""
    monkeypatch.delenv(LLAMA_CPP_DIR_ENV, raising=False)
    monkeypatch.setattr("shutil.which", lambda name: None)

    ok, detail = LlamaCppBackend(llama_cpp_dir=str(tmp_path / "nope")).available()
    assert not ok
    assert CONVERT_SCRIPT in detail
    assert QUANTIZE_BINARY in detail
    assert LLAMA_CPP_DIR_ENV in detail


def test_a_complete_checkout_is_available(tmp_path):
    ok, detail = LlamaCppBackend(llama_cpp_dir=str(_checkout(tmp_path))).available()
    assert ok
    assert CONVERT_SCRIPT in detail


def test_a_half_built_checkout_is_not_available(tmp_path):
    """The script is in the repository but the binary has to be built."""
    root = _checkout(tmp_path, binary=False)
    ok, detail = LlamaCppBackend(llama_cpp_dir=str(root)).available()
    assert not ok
    assert QUANTIZE_BINARY in detail
    assert CONVERT_SCRIPT not in detail


# --- producing an artifact ----------------------------------------------


def test_compression_converts_then_quantizes(tmp_path, monkeypatch):
    root = _checkout(tmp_path)
    job = _job(tmp_path)
    calls: list[list[str]] = []

    def fake_run(command, **kwargs):
        calls.append(command)
        if str(job.output_dir / INTERMEDIATE_NAME) in command:
            (job.output_dir / INTERMEDIATE_NAME).write_bytes(b"f16")
        if str(job.output_dir / ARTIFACT_NAME) in command:
            (job.output_dir / ARTIFACT_NAME).write_bytes(b"quantized")
        return type("R", (), {"returncode": 0, "stdout": "", "stderr": ""})()

    monkeypatch.setattr("subprocess.run", fake_run)

    artifact = LlamaCppBackend(llama_cpp_dir=str(root), python_executable="python").compress(job)

    # A third call asks the binary for its version; the two that matter are the
    # conversion and the quantization, in that order.
    work = [c for c in calls if "--help" not in c]
    assert len(work) == 2

    convert, quantize = work
    assert CONVERT_SCRIPT in " ".join(convert)
    assert "--outtype" in convert and "f16" in convert
    assert QUANTIZE_BINARY in quantize[0]
    assert quantize[-1] == "Q4_K_M"  # the scheme is llama-quantize's type argument
    assert quantize[1].endswith(INTERMEDIATE_NAME)  # quantizes what convert wrote
    assert artifact.backend == "llama.cpp"
    assert artifact.recipe.method == "gguf-q4-k-m"


def test_the_f16_intermediate_is_deleted(tmp_path, monkeypatch):
    """It is the whole model again. Keeping it beside the quantized result costs
    more disk than the thing anyone serves."""
    root = _checkout(tmp_path)
    job = _job(tmp_path)

    def fake_run(command, **kwargs):
        if str(job.output_dir / INTERMEDIATE_NAME) in command:
            (job.output_dir / INTERMEDIATE_NAME).write_bytes(b"f16" * 1000)
        if str(job.output_dir / ARTIFACT_NAME) in command:
            (job.output_dir / ARTIFACT_NAME).write_bytes(b"quantized")
        return type("R", (), {"returncode": 0, "stdout": "", "stderr": ""})()

    monkeypatch.setattr("subprocess.run", fake_run)
    LlamaCppBackend(llama_cpp_dir=str(root), python_executable="python").compress(job)

    assert (job.output_dir / ARTIFACT_NAME).is_file()
    assert not (job.output_dir / INTERMEDIATE_NAME).exists()


def test_the_intermediate_is_deleted_even_when_quantizing_fails(tmp_path, monkeypatch):
    root = _checkout(tmp_path)
    job = _job(tmp_path)

    def fake_run(command, **kwargs):
        if str(job.output_dir / INTERMEDIATE_NAME) in command:
            (job.output_dir / INTERMEDIATE_NAME).write_bytes(b"f16")
            return type("R", (), {"returncode": 0, "stdout": "", "stderr": ""})()
        return type("R", (), {"returncode": 1, "stdout": "", "stderr": "bad type"})()

    monkeypatch.setattr("subprocess.run", fake_run)

    with pytest.raises(CompressionError, match="llama-quantize"):
        LlamaCppBackend(llama_cpp_dir=str(root), python_executable="python").compress(job)

    assert not (job.output_dir / INTERMEDIATE_NAME).exists()


def test_a_silent_failure_to_write_is_caught(tmp_path, monkeypatch):
    """Exit zero with no output file would otherwise be reported as success and
    fail much later, at serve time."""
    root = _checkout(tmp_path)
    job = _job(tmp_path)
    monkeypatch.setattr(
        "subprocess.run",
        lambda command, **kwargs: type("R", (), {"returncode": 0, "stdout": "", "stderr": ""})(),
    )

    with pytest.raises(CompressionError, match="wrote no"):
        LlamaCppBackend(llama_cpp_dir=str(root), python_executable="python").compress(job)


def test_compressing_without_the_tooling_says_so(tmp_path, monkeypatch):
    monkeypatch.delenv(LLAMA_CPP_DIR_ENV, raising=False)
    monkeypatch.setattr("shutil.which", lambda name: None)

    with pytest.raises(CompressionError, match="unavailable"):
        LlamaCppBackend(llama_cpp_dir=str(tmp_path / "nope")).compress(_job(tmp_path))


# --- routing ------------------------------------------------------------


def test_the_method_picks_the_toolchain(tmp_path):
    """Picking a method is picking a toolchain; the spec need not name both."""
    from autodistiller.compression.methods import resolve_method as rm

    assert rm("gguf-q4-k-m").compression_backend == "llama.cpp"
    assert rm("fp8").compression_backend == "llmcompressor"


def test_a_gguf_artifact_directory_is_keyed_like_any_other(tmp_path):
    job = build_job(
        ModelSpec(id="Qwen/Qwen3-0.6B"),
        CompressionSpec(method="gguf-q4-k-m"),
        output_root=tmp_path,
    )
    assert "gguf-q4-k-m" in job.output_dir.name
    assert job.artifact_key


# --- backend-specific serving semantics ---------------------------------


def test_llama_cpp_is_pointed_at_the_gguf_file_not_the_directory(tmp_path):
    """vLLM takes the directory; llama-server takes the file inside it. The same
    artifact, named differently by who is serving it."""
    directory = tmp_path / "art"
    directory.mkdir()
    (directory / ARTIFACT_NAME).write_bytes(b"gguf")

    assert resolve_backend("llama.cpp").model_path(str(directory)).endswith(ARTIFACT_NAME)
    assert resolve_backend("vllm").model_path(str(directory)) == str(directory)


def test_llama_cpp_launches_with_its_own_flags():
    command = resolve_backend("llama.cpp").launch_command("model.gguf", max_model_len=4096)
    assert "llama-server" in command
    assert "-c 4096" in command  # not --max-model-len
    assert "--port 8080" in command  # not 8000


def test_the_kv_cache_flag_is_not_the_same_word_in_both_runtimes():
    assert "kv-cache-dtype" in resolve_backend("vllm").kv_flag_template
    assert "cache-type-k" in resolve_backend("llama.cpp").kv_flag_template


def test_llama_cpp_is_not_offered_an_fp8_kv_cache():
    """It has its own quantized cache vocabulary and no fp8, so searching over
    fp8 would generate candidates it cannot serve."""
    assert resolve_backend("llama.cpp").kv_dtypes == ("auto",)
    assert "fp8" in resolve_backend("vllm").kv_dtypes


def test_llama_cpp_does_not_claim_ignore_eos():
    """vLLM pins every request to exactly max_tokens with it. llama-server has
    no such flag, and pretending otherwise would make the decode timings a lie."""
    assert not resolve_backend("llama.cpp").supports_ignore_eos


def test_a_llama_cpp_search_offers_only_gguf(tmp_path):
    result = generate_candidates(qwen3_06b(), profile=BLACKWELL, backend="llama.cpp")
    accepted = {c.method for c in result.accepted if not c.is_baseline}

    assert accepted
    assert all(METHODS[name].is_gguf for name in accepted)


def test_an_unspecified_serving_backend_does_not_block_the_method(tmp_path, monkeypatch):
    """Naming a GGUF method is naming llama.cpp. Defaulting the serving backend
    to vLLM and then refusing the method rejects a request nobody made."""
    from autodistiller.compression.pipeline import run_compression

    root = _checkout(tmp_path)
    monkeypatch.setattr(
        "subprocess.run",
        lambda command, **kwargs: type("R", (), {"returncode": 0, "stdout": "", "stderr": ""})(),
    )

    # Reaches the backend and fails there for a real reason, not at the gate.
    with pytest.raises(CompressionError, match="wrote no"):
        source = tmp_path / "source-model"
        source.mkdir()
        run_compression(
            ModelSpec(id=str(source)),
            CompressionSpec(
                method="gguf-q4-k-m", llama_cpp_dir=str(root), python_executable="python"
            ),
            output_root=tmp_path,
        )


def test_an_explicit_mismatch_is_still_refused(tmp_path):
    """Producing a GGUF for vLLM is minutes spent on a useless result."""
    from autodistiller.compression.pipeline import run_compression

    with pytest.raises(ValueError, match="cannot serve"):
        run_compression(
            ModelSpec(id="Qwen/Qwen3-0.6B"),
            CompressionSpec(method="gguf-q4-k-m"),
            output_root=tmp_path,
            serving_backend="vllm",
        )


def test_the_converter_never_adopts_the_surrounding_project():
    """uv run syncs whatever project it finds above the working directory. The
    converter is not part of this one, and across a WSL boundary uv reaches
    AutoDistiller's own Windows .venv through /mnt and starts deleting it."""
    from autodistiller.compression.gguf import NO_PROJECT

    for backend in (LlamaCppBackend(), LlamaCppBackend(wrapper='wsl -e bash -lc "{command}"')):
        command = backend._converter_command()
        assert NO_PROJECT in command, command


def test_a_wrapped_converter_resolves_uv_on_the_far_side():
    """Resolving uv here would find this machine's copy, which may be the wrong
    operating system entirely."""
    command = LlamaCppBackend(wrapper='wsl -e bash -lc "{command}"')._converter_command()
    assert command[0] == "uv"  # a bare name, for the far side's PATH to answer
