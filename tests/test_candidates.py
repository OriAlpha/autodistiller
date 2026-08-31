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


def test_an_encoder_is_described_by_its_own_arithmetic() -> None:
    """Same config field names, a different model underneath.

    A BERT block has a two-matrix MLP where a decoder has a gated three, and no
    KV cache at all. Running the decoder formulas over it returns a confident
    wrong estimate, which is worse than none: it is what candidates are screened
    against.
    """

    class Encoder:
        architectures: ClassVar[list[str]] = ["BertForMaskedLM"]
        num_hidden_layers = 12
        hidden_size = 768
        num_attention_heads = 12
        intermediate_size = 3072
        vocab_size = 30522
        max_position_embeddings = 512

    shape = shape_from_config("bert-base-uncased", Encoder())

    assert shape.is_encoder
    assert shape.kv_bytes_per_token == 0
    # bert-base is 109.5M parameters; the gap is layernorms, biases and pooler.
    assert shape.n_parameters == pytest.approx(109.5e6, rel=0.02)
    # The gated-MLP formula would have claimed half again as much feed-forward.
    assert shape.dense_mlp_params_per_layer == 2 * 768 * 3072


def test_rejects_models_the_arithmetic_does_not_describe() -> None:
    """Neither shape, so neither formula. Refused, not guessed into one."""

    class Unknown:
        architectures: ClassVar[list[str]] = ["SomeNovelModel"]
        num_hidden_layers = 12
        hidden_size = 768
        num_attention_heads = 12
        intermediate_size = 3072
        vocab_size = 30522
        max_position_embeddings = 512

    with pytest.raises(ValueError, match="not a decoder-only"):
        shape_from_config("someone/novel", Unknown())

    # Its name says decoder, but its encoder half has no KV cache, so the
    # decoder arithmetic does not describe it either.
    class EncoderDecoder(Unknown):
        architectures: ClassVar[list[str]] = ["WhisperForConditionalGeneration"]
        is_encoder_decoder = True

    with pytest.raises(ValueError, match="encoder-decoder"):
        shape_from_config("openai/whisper-small", EncoderDecoder())

    class Decoder(Unknown):
        architectures: ClassVar[list[str]] = ["Qwen3ForCausalLM"]
        num_key_value_heads = 4

    assert shape_from_config("Qwen/Qwen3-0.6B", Decoder()).n_kv_heads == 4


def test_moe_weights_count_every_expert() -> None:
    """A MoE layer is one gated MLP per expert, not one per layer.

    Counting it as dense reports Qwen3-30B-A3B as 3.34B and screens it as
    something that fits on an 8 GiB card. Every expert is resident whether or
    not a token routes to it, so the total is what a fit decision turns on --
    the active count the model's name advertises is about compute.
    """

    class Base:
        architectures: ClassVar[list[str]] = ["XForCausalLM"]
        tie_word_embeddings = False

    def params_b(**fields: object) -> float:
        config = type("C", (Base,), fields)
        return shape_from_config("m", config()).n_parameters / 1e9

    qwen3_moe = dict(
        num_hidden_layers=48,
        hidden_size=2048,
        num_attention_heads=32,
        num_key_value_heads=4,
        head_dim=128,
        vocab_size=151936,
        max_position_embeddings=40960,
        intermediate_size=6144,
        moe_intermediate_size=768,
        num_experts=128,
    )
    assert params_b(**qwen3_moe) == pytest.approx(30.5, rel=0.03)

    # num_local_experts, and experts that use intermediate_size directly.
    assert params_b(
        num_hidden_layers=32,
        hidden_size=4096,
        num_attention_heads=32,
        num_key_value_heads=8,
        vocab_size=32000,
        max_position_embeddings=32768,
        intermediate_size=14336,
        num_local_experts=8,
    ) == pytest.approx(46.7, rel=0.03)

    # An always-on shared expert on top of the routed ones.
    assert params_b(
        num_hidden_layers=28,
        hidden_size=3584,
        num_attention_heads=28,
        num_key_value_heads=4,
        vocab_size=151936,
        max_position_embeddings=32768,
        intermediate_size=18944,
        moe_intermediate_size=2560,
        num_experts=64,
        shared_expert_intermediate_size=20480,
    ) == pytest.approx(57.4, rel=0.03)

    # A dense model must be untouched by any of it.
    dense = dict(qwen3_moe)
    dense.pop("moe_intermediate_size")
    dense.pop("num_experts")
    shape = shape_from_config("dense", type("C", (Base,), dense)())
    assert not shape.is_moe
    assert shape.mlp_params_per_layer == 3 * 2048 * 6144


def _draft_config(block_size: int = 16, nested: bool = False) -> object:
    """A DFlash draft's config, shaped like the published checkpoints."""
    fields: dict[str, object] = {
        "architectures": ["DFlashDraftModel"],
        "num_hidden_layers": 5,
        "hidden_size": 5120,
        "intermediate_size": 17408,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "vocab_size": 248320,
        "max_position_embeddings": 262144,
        "tie_word_embeddings": False,
        "dtype": "bfloat16",
    }
    # DFlash puts it at the top level, DFlash2 inside dflash_config.
    fields["dflash_config"] = {"block_size": block_size} if nested else {}
    if not nested:
        fields["block_size"] = block_size
    return type("DraftConfig", (), fields)()


def test_draft_shape_is_read_where_the_causal_guard_would_refuse() -> None:
    """A draft is not a servable causal LM, but it is the same block arithmetic.

    `shape_from_config` rightly rejects `DFlashDraftModel`; the draft parser
    reuses ModelShape without that guard rather than duplicating the formulas.
    """
    from autodistiller.candidates.speculative import draft_shape_from_config

    config = _draft_config()
    with pytest.raises(ValueError, match="not a decoder-only"):
        shape_from_config("z-lab/X-DFlash", config)

    shape = draft_shape_from_config("z-lab/X-DFlash", config)
    assert shape.n_layers == 5
    assert shape.architecture == "DFlashDraftModel"
    # 5 blocks over hidden 5120, plus two untied 248320-row embedding tables.
    assert shape.n_parameters == pytest.approx(4.14e9, rel=0.02)


def test_speculative_candidates_carry_the_draft_and_its_memory() -> None:
    """A draft sits on the device beside the target for the whole run.

    Not counting it is the same class of mistake as not counting the KV cache:
    a candidate that is screened as fitting and then OOMs at serve time.
    """
    from autodistiller.candidates.generator import generate_candidates
    from autodistiller.candidates.speculative import SpeculativeSpec

    class Config:
        architectures: ClassVar[list[str]] = ["Qwen3ForCausalLM"]
        num_hidden_layers = 28
        hidden_size = 1024
        num_attention_heads = 16
        num_key_value_heads = 8
        head_dim = 128
        intermediate_size = 3072
        vocab_size = 151936
        max_position_embeddings = 40960

    shape = shape_from_config("Qwen/Qwen3-0.6B", Config())
    draft = SpeculativeSpec(
        method="dflash", model="z-lab/X", n_tokens=15, weights_bytes=2 * 1024**3
    )

    plain = generate_candidates(shape, methods=("fp8",), context_lengths=(2048,))
    with_draft = generate_candidates(
        shape, methods=("fp8",), context_lengths=(2048,), speculative=draft
    )

    # Every recipe is offered both ways, so the comparison is measurable.
    assert len(with_draft.accepted) == 2 * len(plain.accepted)

    speculative = [c for c in with_draft.accepted if c.speculative]
    assert all(c.id.endswith("-dflash15") for c in speculative)
    assert all(c.estimate.draft_bytes == 2 * 1024**3 for c in speculative)

    # The draft is real memory: the same recipe carries 2 GiB more of weights,
    # and the overhead fraction grows with the larger footprint rather than
    # being computed against the target alone.
    by_id = {c.id: c for c in with_draft.accepted}
    speculative_fp8 = by_id["fp8-ctx2048-dflash15"].estimate
    plain_fp8 = by_id["fp8-ctx2048"].estimate
    assert speculative_fp8.weights_bytes - plain_fp8.weights_bytes == 2 * 1024**3
    assert speculative_fp8.overhead_bytes > plain_fp8.overhead_bytes
    assert "draft 2.00" in speculative_fp8.describe()
    assert "draft" not in plain_fp8.describe()

    # A runtime that cannot serve a draft says so rather than shipping one.
    refused = generate_candidates(
        shape,
        methods=("fp8",),
        context_lengths=(2048,),
        speculative=draft,
        supports_speculative=False,
        backend="llama.cpp",
    )
    assert not [c for c in refused.accepted if c.speculative]
    assert any("cannot run dflash drafts" in r for j in refused.rejected for r in j.reasons)


def test_trim_keeps_both_halves_of_a_speculative_search() -> None:
    """The candidate cap must not silently delete the comparison.

    A speculative candidate shares its method with its plain twin and sorts
    after it, so rotating the trim on method alone fills every slot with plain
    candidates and drops the entire speculative half -- precisely when the user
    asked for the comparison by naming a draft.
    """
    from autodistiller.candidates.generator import generate_candidates
    from autodistiller.candidates.speculative import SpeculativeSpec

    class Config:
        architectures: ClassVar[list[str]] = ["Qwen3ForCausalLM"]
        num_hidden_layers = 36
        hidden_size = 2560
        num_attention_heads = 32
        num_key_value_heads = 8
        head_dim = 128
        intermediate_size = 9728
        vocab_size = 151936
        max_position_embeddings = 40960

    shape = shape_from_config("Qwen/Qwen3-4B", Config())
    draft = SpeculativeSpec(method="dflash", model="d", n_tokens=15, weights_bytes=1024**3)

    result = generate_candidates(
        shape, budget_bytes=8 * 1024**3, speculative=draft, max_candidates=12
    )
    speculative = [c for c in result.accepted if c.speculative]

    assert len(result.accepted) == 12
    assert speculative, "the cap dropped every speculative candidate"

    # Each method that survives should survive both ways, so there is something
    # to compare rather than a half-answered question.
    plain_methods = {c.method for c in result.accepted if not c.speculative}
    assert {c.method for c in speculative} == plain_methods


# --- encoders -----------------------------------------------------------


def bge_small() -> ModelShape:
    """BAAI/bge-small-en-v1.5, the encoder measured against here."""
    from autodistiller.architecture import ENCODER

    return ModelShape(
        model_id="BAAI/bge-small-en-v1.5",
        n_layers=12,
        hidden_size=384,
        intermediate_size=1536,
        n_attention_heads=12,
        n_kv_heads=12,
        head_dim=32,
        vocab_size=30522,
        max_position_embeddings=512,
        kind=ENCODER,
    )


def test_encoder_parameter_count_matches_the_real_checkpoint():
    """33.4M on the hub. The gap is layernorms, biases and the pooler."""
    assert bge_small().n_parameters == pytest.approx(33.4e6, rel=0.02)


def test_an_encoder_has_no_kv_cache_at_any_length():
    shape = bge_small()

    assert shape.kv_bytes_per_token == 0
    assert kv_cache_bytes(shape, max_model_len=512, concurrency=32) == 0


def test_encoder_activations_grow_quadratically_with_sequence_length():
    """The attention score matrix is what decides whether a batch fits.

    Doubling the sequence more than doubles the memory, which is why an
    encoder's search runs over sequence length at all rather than pinning it.
    """
    from autodistiller.candidates.memory import activation_bytes

    shape = bge_small()
    short = activation_bytes(shape, seq_len=128, batch=32)
    long = activation_bytes(shape, seq_len=512, batch=32)

    assert long > 4 * short
    # Linear in batch, at a fixed length.
    assert activation_bytes(shape, seq_len=128, batch=64) == pytest.approx(2 * short)


def test_encoder_estimate_uses_activations_and_says_so():
    """Reporting activations under "KV cache" names a thing the model lacks."""
    estimate = estimate_memory(bge_small(), None, max_model_len=512, concurrency=32)

    assert estimate.dynamic_label == "activations"
    assert estimate.kv_cache_bytes > 0
    assert "activations" in estimate.describe()

    decoder = estimate_memory(qwen3_06b(), None, max_model_len=2048)
    assert decoder.dynamic_label == "KV"


def test_encoder_search_drops_the_dimensions_it_does_not_have():
    """No cache means no KV dtype to vary, and no methods that need a decoder."""
    result = generate_candidates(bge_small(), profile=SMALL, backend="vllm")

    assert {c.kv_dtype for c in result.accepted} == {"auto"}
    assert {c.max_model_len for c in result.accepted} <= {128, 256, 512}
    assert "int4-awq" not in {c.method for c in result.accepted}
    assert not any(c.method and c.method.startswith("gguf") for c in result.accepted)
    # What is left is what a real compression run produced on this model.
    assert {c.method for c in result.accepted if c.method} == {
        "int8",
        "int8-weight-only",
        "int4-gptq",
        "fp8",
        "fp8-static",
    }


def test_encoder_overhead_does_not_grow_into_the_card():
    """Measured against vLLM's pooling server, not inherited from the LLM path.

    The decoder rule is a fraction of the budget, because CUDA graphs and the
    allocator size themselves against what is available. A pooling server has
    no cache to size and no context to reserve, so on an 8 GiB card that rule
    claimed 0.75 GiB for a model whose weights are 0.04 -- and every
    configuration then reported the same total, which is a screen that screens
    nothing.
    """
    from autodistiller.candidates.memory import ENCODER_OVERHEAD_FLOOR_BYTES

    shape = bge_small()
    small_card = estimate_memory(
        shape, None, max_model_len=128, concurrency=8, budget_bytes=8 * BYTES_PER_GIB
    )
    big_card = estimate_memory(
        shape, None, max_model_len=128, concurrency=8, budget_bytes=80 * BYTES_PER_GIB
    )

    assert small_card.overhead_bytes == big_card.overhead_bytes
    assert small_card.overhead_bytes == ENCODER_OVERHEAD_FLOOR_BYTES

    # A decoder still scales with the device, which is what was measured for it.
    decoder = estimate_memory(qwen3_06b(), None, max_model_len=2048, budget_bytes=8 * BYTES_PER_GIB)
    bigger = estimate_memory(qwen3_06b(), None, max_model_len=2048, budget_bytes=80 * BYTES_PER_GIB)
    assert bigger.overhead_bytes > decoder.overhead_bytes


def test_encoder_estimate_tracks_what_vllm_reported():
    """vLLM loaded the int8 artifact and reported 0.04 GiB of weights.

    The weights term is the one the screen can be held to exactly, so it is
    checked against the number the server printed rather than against itself.
    """
    shape = bge_small()
    estimate = estimate_memory(
        shape, resolve_method("int8-weight-only"), max_model_len=512, concurrency=32
    )

    assert estimate.weights_bytes / BYTES_PER_GIB == pytest.approx(0.04, abs=0.01)
    # And the total is now within a plausible distance of a real server, where
    # before the overhead rule alone put it above 1 GiB.
    assert estimate.total_gib < 0.6


def test_batch_size_is_searched_only_for_encoders():
    """It is what actually moves an embedding server's throughput.

    A generation request carries one prompt and is batched by the runtime's own
    scheduler, so offering the dimension there enumerates identical candidates.
    """
    encoder = generate_candidates(bge_small(), profile=SMALL, methods=("int8-weight-only",))
    decoder = generate_candidates(qwen3_06b(), profile=SMALL, methods=("int8-weight-only",))

    assert {c.batch_size for c in encoder.accepted} == {1, 8, 32}
    assert {c.batch_size for c in decoder.accepted} == {1}


def test_batching_scales_the_activation_estimate_until_the_scheduler_caps_it():
    """Memory grows with texts in flight, then stops.

    A server admits work up to a token budget and runs the rest after, so past
    that point asking for a bigger batch does not cost more memory -- it queues.
    Without the cap the estimate is a function of the client's concurrency,
    which the server never honoured: 256 texts came out at 2.44 GiB against a
    server that peaked at 0.77 including weights and CUDA graphs.
    """
    from autodistiller.candidates.memory import ACTIVATION_RESIDENT_TOKEN_BUDGET

    result = generate_candidates(
        bge_small(),
        profile=SMALL,
        methods=(),
        context_lengths=(512,),
        concurrency=8,
    )
    by_batch = {c.batch_size: c for c in result.accepted if c.method is None}

    assert by_batch[8].estimate.kv_cache_bytes > by_batch[1].estimate.kv_cache_bytes
    # 8 x 8 texts of 512 tokens is exactly the budget, so 32 queues rather than
    # costing more.
    assert ACTIVATION_RESIDENT_TOKEN_BUDGET == 8 * 8 * 512
    assert by_batch[32].estimate.kv_cache_bytes == by_batch[8].estimate.kv_cache_bytes

    # And the id says which is which, or three rows read identically.
    assert by_batch[32].id.endswith("-b32")
    assert "-b" not in by_batch[1].id
