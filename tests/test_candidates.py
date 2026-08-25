"""Candidate generation and memory estimation.

The estimator is checked against numbers actually measured on an RTX 5070 in
earlier phases: artifact sizes from real llmcompressor runs, and the KV cache
figure vLLM reported at startup. An estimator that drifts from those is worse
than useless, because the whole point is to reject candidates before spending
GPU time on them.
"""

from __future__ import annotations

from typing import ClassVar

import pytest

from autodistiller.candidates.generator import (
    Candidate,
    generate_candidates,
)
from autodistiller.candidates.memory import (
    BYTES_PER_GIB,
    estimate_memory,
    kv_cache_bytes,
    max_context_for_budget,
    parse_size,
    weight_bytes,
)
from autodistiller.candidates.shape import ModelShape, shape_from_config
from autodistiller.compression.methods import METHODS, resolve_method
from autodistiller.metadata.profiles import resolve_profile

AMPERE = resolve_profile("rtx-3090")  # no fp8
BLACKWELL = resolve_profile("rtx-5090")  # 32 GiB, everything
SMALL = resolve_profile("rtx-5070")  # 8 GiB, everything


def qwen3_06b() -> ModelShape:
    """Qwen3-0.6B, the model measured throughout the earlier phases."""
    return ModelShape(
        model_id="Qwen/Qwen3-0.6B",
        n_layers=28,
        hidden_size=1024,
        intermediate_size=3072,
        n_attention_heads=16,
        n_kv_heads=8,
        head_dim=128,
        vocab_size=151936,
        max_position_embeddings=40960,
        tie_word_embeddings=True,
    )


# --- shape --------------------------------------------------------------


def test_shape_reads_grouped_query_attention():
    shape = qwen3_06b()
    assert shape.n_kv_heads == 8
    assert shape.n_attention_heads == 16


def test_missing_kv_head_count_falls_back_to_attention_heads():
    """Pre-GQA models report no num_key_value_heads and use one KV head each."""

    class Config:
        num_hidden_layers = 12
        hidden_size = 768
        num_attention_heads = 12
        vocab_size = 50257
        intermediate_size = 3072
        max_position_embeddings = 1024

    shape = shape_from_config("gpt2-ish", Config())
    assert shape.n_kv_heads == 12
    assert shape.head_dim == 64


def test_tied_embeddings_are_counted_once():
    tied = qwen3_06b()
    untied = ModelShape(**{**tied.__dict__, "tie_word_embeddings": False})
    assert untied.embedding_params == tied.embedding_params * 2


# --- weight estimates, against measured artifacts -----------------------


@pytest.mark.parametrize(
    ("method", "measured_gib"),
    [
        (None, 1.11),  # vLLM reported 1.11 GiB for bf16 weights
        ("fp8", 0.71),  # artifact on disk
        ("int8", 0.71),
        ("int4-gptq", 0.51),
        ("int4-awq", 0.51),
    ],
)
def test_weight_estimate_matches_measured_artifacts(method, measured_gib):
    """Within 5% of what these methods actually produced for Qwen3-0.6B."""
    estimated = weight_bytes(qwen3_06b(), resolve_method(method) if method else None)
    assert estimated / BYTES_PER_GIB == pytest.approx(measured_gib, rel=0.05)


def test_quantization_never_shrinks_embeddings():
    """Embeddings and the output head stay 16-bit, which is why a 4-bit model is
    never a quarter of its 16-bit size."""
    shape = qwen3_06b()
    int4 = weight_bytes(shape, resolve_method("int4-gptq"))
    assert int4 > shape.embedding_params * 2


def test_fewer_bits_means_fewer_bytes():
    """Within one format family. Across families the comparison is not about
    bits: GGUF quantizes the embeddings too, so its 4-bit is smaller than
    compressed-tensors' 4-bit for a reason the bit width does not show."""
    shape = qwen3_06b()
    sizes = {
        name: weight_bytes(shape, method)
        for name, method in METHODS.items()
        if method.weight_bits in (4, 8) and not method.is_gguf
    }
    assert min(sizes.values()) == sizes["int4-gptq"]
    assert weight_bytes(shape, None) > max(sizes.values())


def test_gguf_is_smaller_than_the_same_width_elsewhere():
    """compressed-tensors leaves embeddings at 16-bit and GGUF does not, which
    on a model carrying a quarter of its parameters in the embedding is the
    difference between the two formats at the same nominal width."""
    shape = qwen3_06b()
    assert weight_bytes(shape, METHODS["gguf-q4-k-m"]) < weight_bytes(shape, METHODS["int4-gptq"])


def test_gguf_size_matches_a_measured_artifact():
    """Against a GGUF this project actually produced, not a published figure.

    llama.cpp writes the embedding twice -- token_embd at the headline type and
    output.weight at Q6_K -- which a whole-file bits-per-weight average does not
    capture. Counting it once under-reported this artifact by 18.5%.
    """
    measured_mib = 461.8  # artifacts/Qwen3-0.6B-gguf-q4-k-m, llama.cpp b7584430
    estimate = weight_bytes(qwen3_06b(), METHODS["gguf-q4-k-m"]) / 1024**2

    assert estimate == pytest.approx(measured_mib, rel=0.05)
    assert estimate >= measured_mib, "a screen must not under-estimate what has to fit"


def test_gguf_estimates_rise_with_the_quantization_type():
    shape = qwen3_06b()
    sizes = [
        weight_bytes(shape, METHODS[n])
        for n in ("gguf-q3-k-m", "gguf-q4-k-m", "gguf-q5-k-m", "gguf-q6-k", "gguf-q8-0")
    ]
    assert sizes == sorted(sizes)
    assert sizes[-1] < weight_bytes(shape, None), "every quant beats bf16"


# --- KV cache, against what vLLM reported -------------------------------


def test_kv_cache_per_token_matches_vllm():
    """vLLM allocated 41,344 tokens in 4.42 GiB: 112 KiB per token."""
    per_token = qwen3_06b().kv_bytes_per_token
    assert per_token / 1024 == pytest.approx(112, rel=0.01)

    vllm_reported = 4.42 * BYTES_PER_GIB / 41344
    assert per_token == pytest.approx(vllm_reported, rel=0.02)


def test_kv_cache_scales_with_context_and_concurrency():
    shape = qwen3_06b()
    base = kv_cache_bytes(shape, max_model_len=1024, concurrency=1)
    assert kv_cache_bytes(shape, max_model_len=2048, concurrency=1) == base * 2
    assert kv_cache_bytes(shape, max_model_len=1024, concurrency=4) == base * 4


def test_fp8_kv_cache_halves_it():
    shape = qwen3_06b()
    full = kv_cache_bytes(shape, max_model_len=4096)
    assert kv_cache_bytes(shape, max_model_len=4096, kv_dtype="fp8") == full // 2


def test_kv_cache_overtakes_weights_on_long_context():
    """The reason context length is a candidate dimension at all."""
    shape = qwen3_06b()
    weights = weight_bytes(shape, None)
    assert kv_cache_bytes(shape, max_model_len=32768, concurrency=1) > weights


# --- fit ----------------------------------------------------------------


def test_estimate_reports_a_breakdown():
    estimate = estimate_memory(
        qwen3_06b(), None, max_model_len=4096, budget_bytes=8 * BYTES_PER_GIB
    )
    assert estimate.total_bytes == (
        estimate.weights_bytes + estimate.kv_cache_bytes + estimate.overhead_bytes
    )
    assert estimate.fits
    assert 0 < estimate.utilization < 1


def test_a_candidate_that_cannot_fit_is_reported_as_such():
    estimate = estimate_memory(
        qwen3_06b(),
        None,
        max_model_len=32768,
        concurrency=16,
        budget_bytes=8 * BYTES_PER_GIB,
    )
    assert not estimate.fits
    assert estimate.headroom_bytes < 0


def test_no_budget_means_no_verdict():
    estimate = estimate_memory(qwen3_06b(), None, max_model_len=2048)
    assert estimate.fits
    assert estimate.headroom_bytes is None


def test_max_context_for_budget_is_consistent_with_the_estimate():
    shape = qwen3_06b()
    budget = 8 * BYTES_PER_GIB
    longest = max_context_for_budget(shape, None, budget_bytes=budget)

    assert estimate_memory(shape, None, max_model_len=longest, budget_bytes=budget).fits
    assert not estimate_memory(
        shape, None, max_model_len=int(longest * 1.2), budget_bytes=budget
    ).fits


def test_compression_buys_context():
    shape = qwen3_06b()
    budget = 8 * BYTES_PER_GIB
    baseline = max_context_for_budget(shape, None, budget_bytes=budget)
    compressed = max_context_for_budget(shape, resolve_method("int4-awq"), budget_bytes=budget)
    assert compressed > baseline


def test_weights_alone_over_budget_leaves_no_context():
    tiny_budget = 100 * 1024**2
    assert max_context_for_budget(qwen3_06b(), None, budget_bytes=tiny_budget) == 0


# --- size parsing -------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("8GiB", 8 * 1024**3),
        ("8GB", 8 * 1000**3),
        ("8192MiB", 8 * 1024**3),
        ("16 gb", 16 * 1000**3),
        ("1024", 1024),
    ],
)
def test_parse_size(text, expected):
    assert parse_size(text) == expected


def test_gb_and_gib_are_not_the_same():
    assert parse_size("8GB") < parse_size("8GiB")


def test_unparseable_size_explains_the_format():
    with pytest.raises(ValueError, match="8GiB"):
        parse_size("a lot")


# --- generation ---------------------------------------------------------


def test_generates_a_small_explainable_space():
    """The roadmap asks for roughly 15-25 candidates, not a combinatorial blowup."""
    result = generate_candidates(qwen3_06b(), profile=SMALL, backend="vllm", concurrency=8)
    assert 10 <= len(result.accepted) <= 25
    assert result.rejected
    assert all(r.reasons for r in result.rejected)


def test_baseline_comes_first():
    result = generate_candidates(qwen3_06b(), profile=SMALL)
    assert result.accepted[0].is_baseline
    assert result.baseline is not None


def test_baseline_can_be_excluded():
    result = generate_candidates(qwen3_06b(), profile=SMALL, include_baseline=False)
    assert all(not c.is_baseline for c in result.accepted)


def test_hardware_filters_out_unsupported_formats():
    """Ampere has no FP8, for weights or for the KV cache."""
    result = generate_candidates(qwen3_06b(), profile=AMPERE, backend="vllm")
    assert all(c.method != "fp8" for c in result.accepted)
    assert all(c.kv_dtype != "fp8" for c in result.accepted)

    reasons = [r for r in result.rejected if any("fp8" in x for x in r.reasons)]
    assert reasons


def test_backend_filters_independently_of_hardware():
    """A Blackwell card can run every compressed-tensors format, and llama.cpp
    can serve none of them. Hardware support and backend support are separate
    questions and this is where that separation shows."""
    result = generate_candidates(qwen3_06b(), profile=BLACKWELL, backend="llama.cpp")

    served = {c.method for c in result.accepted if not c.is_baseline}
    assert served, "llama.cpp should accept the GGUF methods"
    assert all(METHODS[name].is_gguf for name in served)
    assert any("llama.cpp" in reason for r in result.rejected for reason in r.reasons)


def test_vllm_does_not_accept_gguf():
    result = generate_candidates(qwen3_06b(), profile=BLACKWELL, backend="vllm")
    served = {c.method for c in result.accepted if not c.is_baseline}
    assert served and not any(METHODS[name].is_gguf for name in served)


def test_memory_rejects_what_cannot_fit():
    result = generate_candidates(
        qwen3_06b(), profile=SMALL, budget_bytes=2 * BYTES_PER_GIB, concurrency=16
    )
    assert any("memory" in reason for r in result.rejected for reason in r.reasons)
    assert all(c.estimate.fits for c in result.accepted)


def test_a_bigger_card_accepts_more():
    small = generate_candidates(qwen3_06b(), profile=SMALL, concurrency=16, max_candidates=100)
    big = generate_candidates(qwen3_06b(), profile=BLACKWELL, concurrency=16, max_candidates=100)
    assert len(big.accepted) > len(small.accepted)


def test_context_longer_than_the_model_is_not_a_candidate():
    short = ModelShape(**{**qwen3_06b().__dict__, "max_position_embeddings": 2048})
    result = generate_candidates(short, profile=BLACKWELL)
    assert all(c.max_model_len <= 2048 for c in result.accepted)


def test_the_candidate_cap_keeps_every_method_represented():
    """Truncating the sorted list would drop the entire 4-bit family, which is
    exactly what a memory-constrained search is looking for."""
    result = generate_candidates(qwen3_06b(), profile=BLACKWELL, backend="vllm", max_candidates=14)
    assert len(result.accepted) == 14

    represented = {c.method for c in result.accepted}
    assert "int4-awq" in represented
    assert "int4-gptq" in represented
    assert None in represented  # the baseline


def test_methods_can_be_restricted():
    result = generate_candidates(
        qwen3_06b(), profile=BLACKWELL, methods=("int4-awq",), include_baseline=False
    )
    assert {c.method for c in result.accepted} == {"int4-awq"}


def test_unknown_method_is_rejected_loudly():
    with pytest.raises(KeyError, match="int4-awq"):
        generate_candidates(qwen3_06b(), methods=("int2-wishful",))


def test_candidate_ids_are_unique_and_descriptive():
    result = generate_candidates(qwen3_06b(), profile=SMALL, concurrency=4)
    ids = [c.id for c in result.accepted]
    assert len(ids) == len(set(ids))
    assert any("ctx" in i for i in ids)
    assert any("baseline" in i for i in ids)


def test_rejection_summary_groups_by_cause():
    result = generate_candidates(qwen3_06b(), profile=AMPERE, concurrency=16)
    summary = result.rejection_summary()
    assert summary
    assert sum(summary.values()) >= len(result.rejected)


def test_a_candidate_knows_its_own_estimate():
    candidate = Candidate(
        method="fp8",
        max_model_len=4096,
        kv_dtype="auto",
        estimate=estimate_memory(qwen3_06b(), resolve_method("fp8"), max_model_len=4096),
    )
    assert candidate.id == "fp8-ctx4096"
    assert "GiB" in candidate.describe()


# --- vision-language models ----------------------------------------------


def test_a_nested_text_config_is_found():
    """Gemma 3 and other vision-language models put the decoder's dimensions in
    `text_config` and leave the top level with neither tower's. Reading the top
    level fails outright, so the model cannot even be screened."""

    class Text:
        hidden_size = 2560
        num_hidden_layers = 34
        num_attention_heads = 8
        num_key_value_heads = 4
        head_dim = 256
        intermediate_size = 10240
        vocab_size = 262208
        max_position_embeddings = 131072

    class Vision:
        hidden_size = 1152
        num_hidden_layers = 27

    class Config:
        text_config = Text()
        vision_config = Vision()
        architectures: ClassVar = ["Gemma3ForConditionalGeneration"]

    shape = shape_from_config("google/gemma-3-4b-it", Config())

    assert shape.n_layers == 34, "read the vision tower's depth instead of the decoder's"
    assert shape.hidden_size == 2560
    assert shape.vocab_size == 262208
    # The architecture name lives at the top level, not in the nested config.
    assert shape.architecture == "Gemma3ForConditionalGeneration"


def test_a_flat_config_is_unaffected():
    """The overwhelming majority of models, which must not change."""
    shape = qwen3_06b()
    assert shape.hidden_size == 1024 and shape.n_layers == 28


def test_an_empty_text_config_falls_back_to_the_top_level():
    """Some configs declare the attribute and leave it hollow. Trusting it then
    would break a model that reads fine from the top level."""

    class Empty:
        pass

    class Config:
        text_config = Empty()
        hidden_size = 768
        num_hidden_layers = 12
        num_attention_heads = 12
        vocab_size = 50257
        intermediate_size = 3072
        max_position_embeddings = 1024

    shape = shape_from_config("flat/model", Config())
    assert shape.hidden_size == 768 and shape.n_layers == 12
