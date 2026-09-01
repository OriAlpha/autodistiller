"""Greedy generation smoke test, for models that generate.

This proves the loaded model actually produces text, captures its output so
drift is human-inspectable, and records rough timings.

Those timings come from Transformers. They are **not** deployment performance.
The roadmap is explicit that vLLM numbers must be measured inside vLLM, so the
result is tagged ``runtime="transformers"`` and ``is_deployment_claim=False``
and Phase 2 will produce the real serving measurements.
"""

from __future__ import annotations

import time

import torch

from ..config import BaselineInferenceSpec
from ..models.loader import LoadedModel
from ..results import GenerationSample, InferenceResult


@torch.inference_mode()
def run_baseline_inference(handle: LoadedModel, spec: BaselineInferenceSpec) -> InferenceResult:
    if not spec.enabled or not spec.prompts:
        return InferenceResult()

    # An encoder or a vision tower has nothing to generate: no head that
    # predicts a next token, and for a vision tower no tokenizer to build a
    # prompt with either. Transformers answers this itself, so the check is its
    # answer rather than a second list of which kinds can. Skipped rather than
    # attempted-and-failed: a smoke test that does not apply is not a failure of
    # the model, and recording it as one would mark every such run failed.
    if not handle.model.can_generate():
        return InferenceResult()

    tokenizer = handle.tokenizer
    is_cuda = handle.device.type == "cuda"
    if is_cuda:
        torch.cuda.reset_peak_memory_stats(handle.device)

    generate_kwargs = {
        "max_new_tokens": spec.max_new_tokens,
        "do_sample": False,  # greedy: identical config must give identical text
        "pad_token_id": tokenizer.pad_token_id,
    }

    # Warm up outside the timed region: the first CUDA generate pays for kernel
    # autotuning and allocator growth that no real workload repeats.
    for _ in range(spec.warmup_runs):
        warm = tokenizer(spec.prompts[0], return_tensors="pt").to(handle.device)
        handle.model.generate(**warm, **generate_kwargs)

    samples: list[GenerationSample] = []
    started = time.perf_counter()

    for prompt in spec.prompts:
        encoded = tokenizer(prompt, return_tensors="pt").to(handle.device)
        n_prompt_tokens = int(encoded["input_ids"].shape[-1])

        best_latency = float("inf")
        # `repeats` is validated as >= 1, so the loop always assigns output_ids.
        output_ids: torch.Tensor = torch.empty(0)
        for _ in range(spec.repeats):
            if is_cuda:
                torch.cuda.synchronize(handle.device)
            run_started = time.perf_counter()
            output_ids = handle.model.generate(**encoded, **generate_kwargs)
            if is_cuda:
                torch.cuda.synchronize(handle.device)
            # Min over repeats: the fastest run is the least contaminated by
            # background load on the machine.
            best_latency = min(best_latency, time.perf_counter() - run_started)

        generated = output_ids[0][n_prompt_tokens:]
        samples.append(
            GenerationSample(
                prompt=prompt,
                output=tokenizer.decode(generated, skip_special_tokens=True)
                if spec.capture_output
                else None,
                n_prompt_tokens=n_prompt_tokens,
                n_generated_tokens=int(generated.numel()),
                latency_s=best_latency,
            )
        )

    total_duration = time.perf_counter() - started
    rates = [s.tokens_per_second for s in samples if s.latency_s > 0]

    return InferenceResult(
        runtime="transformers",
        is_deployment_claim=False,
        samples=samples,
        mean_tokens_per_second=sum(rates) / len(rates) if rates else 0.0,
        total_duration_s=total_duration,
        peak_vram_bytes=int(torch.cuda.max_memory_allocated(handle.device)) if is_cuda else None,
    )


__all__ = ["run_baseline_inference"]
