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
 best quality           baseline    quality retention 100.00%    780 tok/s against a best of 1320
 fastest (throughput)   int4-awq    peak throughput 1320 tok/s   quality 94.10% against 100.00%
 smallest               int4-awq    artifact size 0.51 GiB       quality 94.10% against 100.00%
```

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

Qwen3-0.6B on an RTX 5070 Laptop (8 GiB), vLLM 0.27, plugged in. Produced by the tool itself.

**Compression** — wikitext-2 perplexity, same baseline for every row:

| Method | Size | Quality kept |
|---|---|---|
| baseline (bf16) | 1.11 GiB | 100% |
| `int8-weight-only` | 0.72 GiB | 100.2% |
| `fp8` | 0.71 GiB | 99.0% |
| `int4-awq` | 0.51 GiB | 86.4% |

**Speed** — measured inside vLLM, zero failed requests:

| Concurrency | bf16 | fp8 | fp8 gain |
|---|---|---|---|
| 1 | 195 tok/s | 253 tok/s | **1.30x** |
| 4 | 649 tok/s | 824 tok/s | 1.27x |
| 16 | 2144 tok/s | 2626 tok/s | 1.22x |
| 32 | 3036 tok/s | 3623 tok/s | 1.19x |

So on this model FP8 is the pick: ~25% faster for 1% of quality. INT4 saves another 0.2 GiB but
costs 13.6% — the kind of trade-off you want to see before choosing, not after.

> **Your numbers will differ.** The same model on the same GPU measured **3x apart** during
> development: once because the laptop was on battery (GPU capped to 34 W), once because another
> process held 5.9 GiB of the card. That is why every run records the hardware and software it ran
> on, and why a cached result is never reused across a change in either.

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

**Known gap:** the llama.cpp path is implemented and unit-tested but has not been run against real
llama.cpp binaries. Treat GGUF support as experimental until it has.

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
