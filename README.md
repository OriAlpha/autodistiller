# AutoDistiller

**Find the best way to deploy your LLM — automatically.**

[![CI](https://github.com/OriAlpha/Autodistiller/actions/workflows/ci.yml/badge.svg)](https://github.com/OriAlpha/Autodistiller/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/autodistiller)](https://pypi.org/project/autodistiller/)
[![Python](https://img.shields.io/pypi/pyversions/autodistiller)](https://pypi.org/project/autodistiller/)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue)](LICENSE)

---

## The problem

You want to run a model on your GPU. Should you quantize it? To INT8, INT4, FP8? Which one keeps
enough quality? Will it even fit in VRAM? How much faster is it *actually*?

Answering that by hand means compressing several versions, evaluating each, starting a server,
benchmarking, and tracking what you measured. Hours of work, repeated whenever anything changes.

## What AutoDistiller does

One command:

```bash
uv run autodistiller optimize --model Qwen/Qwen3-0.6B --max-vram 8GB --min-quality 95
```

It compresses the realistic options, checks quality against your baseline, benchmarks the survivors
in a real vLLM or llama.cpp server, and tells you which to deploy — and what you give up by
choosing it.

```
 Option                 Candidate   Wins on                      Gives up
 fastest (throughput)   int4-awq    peak throughput 641 tok/s    hellaswag/acc_norm 0.5625 against a
                                                                 best of 0.5781 (within noise -- not
                                                                 a measurable difference)
 best quality           int8-w-o    hellaswag/acc_norm 0.5781    peak throughput 463 against 641 tok/s
```

It reports error bars and says when a difference is too small to be real, because a
recommendation built on noise is worse than no recommendation.

It does not implement quantization itself. It drives the tools that do — LLM Compressor,
llama.cpp — measures the results honestly, and picks.

---

## Install

```bash
pip install autodistiller
```

Check what it found on your machine:

```bash
uv run autodistiller env
```

CPU-only install: `UV_TORCH_BACKEND=cpu uv sync`

---

## Try it

**See the options before spending anything on them.** Free — it is arithmetic, no GPU needed.

```bash
uv run autodistiller candidates --model Qwen/Qwen3-0.6B --max-vram 8GB
```

**Measure your model as it is**, so later numbers mean something.

```bash
uv run autodistiller evaluate --model Qwen/Qwen3-0.6B --task wikitext2
```

**Compress one version** and check what it cost.

```bash
uv run autodistiller compress --model Qwen/Qwen3-0.6B --method fp8
```

**Or let it decide**, and export the winner ready to serve.

```bash
uv run autodistiller optimize --model Qwen/Qwen3-0.6B --max-vram 8GB \
  --launch-preset wsl-vllm --export ./my-model
```

Nothing is measured twice — repeat any of these and it reuses the earlier result.

---

## What it measured

Real output from the tool, on an RTX 5070 Laptop (8 GiB), vLLM 0.27, plugged in.

### A complete run: Qwen3-4B on an 8 GiB card

bf16 Qwen3-4B is 7.49 GiB of weights, so it does not fit. Quantizing is not an
optimisation here, it is the only way to run the model at all.

```bash
uv run autodistiller optimize --model Qwen/Qwen3-4B --max-vram 8GiB   --task hellaswag --task arc_easy   --method fp8 --method int4-awq --method int8-weight-only   --calibration wikitext2 --launch-preset wsl-vllm   --no-stop-early --objective throughput
```

It compresses each method, evaluates accuracy on both tasks, starts a real vLLM
server for each survivor and benchmarks it, then ranks:

```
| Candidate                | Stage       | Quality               |     Size | Throughput | TTFT |
|--------------------------|-------------|-----------------------|----------|------------|------|
| int4-awq-ctx2048         | benchmarked | hellaswag 0.5625 ±.044| 2.48 GiB |  641 tok/s | 60ms |
|                          |             | arc_easy  0.7422 ±.039|          |            |      |
| int8-weight-only-ctx2048 | benchmarked | hellaswag 0.5781 ±.044| 4.17 GiB |  463 tok/s | 68ms |
|                          |             | arc_easy  0.7812 ±.037|          |            |      |
| fp8-ctx2048              | benchmarked | hellaswag 0.5703 ±.044| 4.12 GiB |  444 tok/s | 67ms |
|                          |             | arc_easy  0.7734 ±.037|          |            |      |

Search took 20.4 min -- 13.0 compressing, 25.1 evaluating
```

Then the trade-offs, and what each option costs:

```
Pareto frontier - hellaswag/acc_norm vs Peak VRAM vs TTFT p50 vs Peak throughput
| Candidate                | hellaswag/acc_norm | Peak VRAM | TTFT p50 | Peak throughput | Verdict        |
|--------------------------|--------------------|-----------|----------|-----------------|----------------|
| int4-awq-ctx2048         |             0.5625 |  7.75 GiB |     60ms |       641 tok/s | Pareto-optimal |
| fp8-ctx2048              |             0.5703 |  7.82 GiB |     67ms |       444 tok/s | Pareto-optimal |
| int8-weight-only-ctx2048 |             0.5781 |  7.82 GiB |     68ms |       463 tok/s | Pareto-optimal |

Recommendations
| Option               | Candidate        | Wins on                   | Gives up                          |
|----------------------|------------------|---------------------------|-----------------------------------|
| fastest (throughput) | int4-awq-ctx2048 | peak throughput 641 tok/s | hellaswag/acc_norm 0.5625 against |
|                      |                  |                           | a best of 0.5781 (within noise -- |
|                      |                  |                           | not a measurable difference)      |
| smallest             | int4-awq-ctx2048 | artifact size 2.48 GiB    | (same)                            |
```

### What that tells you

| Method | Size | vs bf16 | hellaswag | arc_easy | Throughput |
|---|---|---|---|---|---|
| bf16 (does not fit) | 7.49 GiB | — | — | — | — |
| **`int4-awq`** | **2.48 GiB** | **67% smaller** | 0.5625 ±.044 | 0.7422 ±.039 | **641 tok/s** |
| `fp8` | 4.12 GiB | 45% smaller | 0.5703 ±.044 | 0.7734 ±.037 | 444 tok/s |
| `int8-weight-only` | 4.17 GiB | 44% smaller | 0.5781 ±.044 | 0.7812 ±.037 | 463 tok/s |

**Read the error bars.** The accuracy spread across all three is 0.016 while the
standard error on each is 0.044 — nearly three times larger. These models are
**not measurably different in accuracy**, so int4-awq is 67% smaller and 44%
faster for no cost anyone can demonstrate. The tool says "within noise" rather
than letting you read 0.5625 vs 0.5781 as a real loss.

That 20.4 minutes is fast only because the cache reused four compressions and
four benchmarks from an earlier search. A cold run of the same command is closer
to an hour, most of it evaluation.

### Qwen3-0.6B — where it is optional

Small enough that bf16 fits, so here you get quality *retention* against a real
baseline:

| Method | Size | Quality kept | Throughput @c32 |
|---|---|---|---|
| baseline (bf16) | 1.11 GiB | 100% | 3036 tok/s |
| `int8-weight-only` | 0.72 GiB | 100.2% | — |
| `fp8` | 0.71 GiB | 99.0% | **3623 tok/s** |
| `int4-awq` | 0.51 GiB | 86.4% | — |

Note the reversal: **fp8 wins at 0.6B, int4-awq wins at 4B.** At 4B on an 8 GiB
card the model is memory-bandwidth-bound, so halving the weights beats cheaper
arithmetic. You would not guess that; you have to measure it, which is the point.

> **Your numbers will differ.** The same model on the same GPU measured **3x
> apart** during development: once because the laptop was on battery (GPU capped
> to 34 W), once because another process held 5.9 GiB of the card. That is why
> every run records the hardware and software it ran on, and why a cached result
> is never reused across a change in either.

---

## Which models work

Any Hugging Face causal LM. There is no supported-model list in the code and no
per-architecture handling — dimensions come from `config.json` and loading goes
through `AutoModelForCausalLM`. Every one of these was checked on an 8 GiB card:

| Model | Family | Params |
|---|---|---|
| `openai-community/gpt2` | GPT-2 | 0.15B |
| `TinyLlama/TinyLlama-1.1B-Chat-v1.0` | Llama | 1.10B |
| `meta-llama/Llama-3.2-1B-Instruct` | Llama 3 | 1.24B |
| `deepseek-ai/deepseek-coder-1.3b-instruct` | DeepSeek | 1.35B |
| `stabilityai/stablelm-2-1_6b` | StableLM | 1.64B |
| `HuggingFaceTB/SmolLM2-1.7B-Instruct` | Llama | 1.71B |
| `EleutherAI/pythia-1.4b` | GPT-NeoX | 1.82B |
| `ibm-granite/granite-3.1-2b-instruct` | Granite | 2.53B |
| `google/gemma-2-2b-it` | Gemma 2 | 2.61B |
| `meta-llama/Llama-3.2-3B-Instruct` | Llama 3 | 3.21B |
| `microsoft/Phi-3-mini-4k-instruct` | Phi-3 | 3.82B |
| `google/gemma-3-4b-it` | Gemma 3 (VLM) | 3.88B |
| `Qwen/Qwen3-4B` | Qwen 3 | 4.02B |
| `mistralai/Mistral-7B-Instruct-v0.3` | Mistral | 7.25B |
| `allenai/OLMo-2-1124-7B` | OLMo 2 | 7.30B |
| `meta-llama/Llama-3.1-8B-Instruct` | Llama 3 | 8.03B |
| `tiiuae/falcon-7b-instruct` | Falcon | 10.87B |

Vision-language models are read by their language tower — Gemma 3 keeps the
decoder in a nested `text_config`, and that is the half that gets quantized and
whose KV cache dominates memory. The vision encoder is not counted.

**Gated models** (Llama, Gemma) need Hugging Face access: run `hf auth login`,
then request access on the model page and wait for approval.

**Not supported:** encoder-decoder models (T5, BART), because loading goes
through `AutoModelForCausalLM`. Non-NVIDIA accelerators are also out of scope —
the capability rules are keyed on CUDA compute capability.

### Compression methods

| Method | Bits | Calibration | Served by | Needs |
|---|---|---|---|---|
| `int8-weight-only` | W8A16 | no | vLLM | INT8 (sm_75+) |
| `int8` | W8A8 | **yes** | vLLM | INT8 (sm_75+) |
| `int4-gptq` | W4A16 | **yes** | vLLM | INT4 (sm_75+) |
| `int4-awq` | W4A16 | **yes** | vLLM | INT4 (sm_75+) |
| `fp8` | W8A8 | no | vLLM | FP8 (sm_89+) |
| `fp8-static` | W8A8 | **yes** | vLLM | FP8 (sm_89+) |
| `gguf-q8-0` … `gguf-q3-k-m` | 8 → 3 bit | no | llama.cpp | any, CPU included |

Methods marked **yes** need `--calibration wikitext2` (or your own corpus) and
fail immediately with a clear message otherwise. `autodistiller methods` prints
this for your actual GPU, marking what it can and cannot run.

---

## Commands

| | |
|---|---|
| `env` | what hardware and software will be recorded |
| `candidates` | the options worth measuring, and why the rest were dropped |
| `evaluate` | measure a model's quality |
| `compare` | did quality hold? |
| `compress` | build one compressed version |
| `prune` | drop the transformer blocks that do the least |
| `benchmark` | measure a running server |
| `optimize` | do all of it, and recommend |
| `export` | make a result deployable and reproducible |
| `runs` · `show` · `history` | what you have measured before |

`--help` on any of them. `autodistiller methods` shows what your GPU and backend support.

---

## Status

Phases 1–9 are done, which is the whole v1.0 scope.

| Phase | | Phase | |
|---|---|---|---|
| 1 Evaluation engine | done | 6 Experiment cache | done |
| 2 Deployment profiling | done | 7 Pareto analysis | done |
| 3 Compression backends | done | 8 Export | done |
| 4 Candidate generation | done | 9 llama.cpp | done |
| 5 Constrained optimization | done | 10 Post-v1 research | pruning done |

Both backends are verified end to end against real binaries: vLLM on GPU, and llama.cpp built from
source and actually run — CPU-only so far, so GGUF throughput has not been measured on a GPU.

**On Windows**, llama.cpp's binaries are Linux executables, so build it in WSL and add
`--llama-cpp-wsl`:

```bash
uv run autodistiller compress --model Qwen/Qwen3-0.6B --method gguf-q4-k-m \
  --llama-cpp ~/llama.cpp --llama-cpp-wsl
```

v1.0 targets Hugging Face models on NVIDIA GPUs, vLLM first and llama.cpp next, with
INT4/INT8/AWQ/GPTQ and FP8 through existing backends.

Depth pruning landed from Phase 10: `--prune 2,4` searches block count as a dimension alongside
quantization, and the two compose — prune first, then quantize what is left. It is measured, not
assumed, and on small models it usually loses: dropping 4 of Qwen3-0.6B's 28 blocks took wikitext-2
perplexity from 17.0 to 29.0, which the search reports as dominated rather than recommending it.
2:4 sparsity is not offered, because vLLM 0.27 removed sparsity support outright and llama.cpp never
had it. Still post-v1: knowledge distillation, student-model search, and Bayesian optimization — the
last only if the discrete search space proves limiting.

---

## More

- **[Design notes](docs/design.md)** — why it works this way, and the failures that shaped it
- **[vLLM on WSL](docs/vllm-on-wsl.md)** — four undocumented blockers, and their fixes
- **[Contributing](CONTRIBUTING.md)** · **[Changelog](CHANGELOG.md)**

AutoTrainer is a separate project: it trains and fine-tunes, AutoDistiller takes a trained model
and makes it deployable. They stay interoperable rather than merged.

---

## License

Apache 2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).
