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

### Qwen3-4B — where quantizing is not optional

bf16 Qwen3-4B needs ~8 GB of weights, so it does not fit. Every option below is
one you would actually have to choose between. Accuracy is `acc_norm` on 128
questions each from hellaswag and arc_easy:

| Method | Size | hellaswag | arc_easy | Throughput | TTFT |
|---|---|---|---|---|---|
| **`int4-awq`** | **2.48 GiB** | 0.5625 ±0.044 | 0.7422 ±0.039 | **641 tok/s** | **60 ms** |
| `int8-weight-only` | 4.17 GiB | 0.5781 ±0.044 | 0.7812 ±0.037 | 463 tok/s | 68 ms |
| `fp8` | 4.12 GiB | 0.5703 ±0.044 | 0.7734 ±0.037 | 444 tok/s | 67 ms |

Read the error bars. The accuracy spread is 0.016; the standard error is 0.044 —
nearly three times larger. **These models are not measurably different in
accuracy**, and int4-awq is 44% faster and 40% smaller. The tool says so rather
than leaving you to notice:

```
 Option                 Candidate   Wins on                      Gives up
 fastest (throughput)   int4-awq    peak throughput 641 tok/s    hellaswag/acc_norm 0.5625 against a
                                                                 best of 0.5781 (within noise -- not
                                                                 a measurable difference)
```

That run took **20.4 min** — most of it evaluation, and only that fast because
the cache reused four compressions and four benchmarks from an earlier search.

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

## Commands

| | |
|---|---|
| `env` | what hardware and software will be recorded |
| `candidates` | the options worth measuring, and why the rest were dropped |
| `evaluate` | measure a model's quality |
| `compare` | did quality hold? |
| `compress` | build one compressed version |
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
| 5 Constrained optimization | done | 10 Post-v1 research | later |

Both backends are verified end to end against real binaries: vLLM on GPU, and llama.cpp built from
source and actually run — CPU-only so far, so GGUF throughput has not been measured on a GPU.

**On Windows**, llama.cpp's binaries are Linux executables, so build it in WSL and add
`--llama-cpp-wsl`:

```bash
uv run autodistiller compress --model Qwen/Qwen3-0.6B --method gguf-q4-k-m \
  --llama-cpp ~/llama.cpp --llama-cpp-wsl
```

v1.0 targets Hugging Face models on NVIDIA GPUs, vLLM first and llama.cpp next, with
INT4/INT8/AWQ/GPTQ and FP8 through existing backends. Phase 10 is deliberately post-v1: knowledge
distillation, pruning, student-model search, and Bayesian optimization — the last only if the
discrete search space proves limiting.

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
