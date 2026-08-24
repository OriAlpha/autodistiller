# AutoDistiller

**Automatically find the best LLM deployment configuration for your hardware and quality constraints.**

[![CI](https://github.com/OriAlpha/Autodistiller/actions/workflows/ci.yml/badge.svg)](https://github.com/OriAlpha/Autodistiller/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/autodistiller)](https://pypi.org/project/autodistiller/)
[![Python](https://img.shields.io/pypi/pyversions/autodistiller)](https://pypi.org/project/autodistiller/)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue)](LICENSE)

AutoDistiller is the automation layer *above* established compression and serving backends. You
provide a model, a deployment backend, hardware and constraints; AutoDistiller evaluates realistic
candidates, benchmarks them in the target runtime, and recommends the best qualifying
configuration.

It does not implement quantization kernels and does not reimplement AWQ, GPTQ or any other mature
algorithm. It composes them, measures them under real deployment conditions, and chooses.

---

## Status: Phases 1–5

Phase 1 is complete and usable on its own. Its milestone is deliberately unglamorous:

> **Establish a trustworthy baseline before any compression is attempted.**

Everything later — candidate generation, constrained optimization, Pareto analysis — is only as
good as the baseline it is measured against. So Phase 1 ships:

| Capability | Where |
|---|---|
| Hugging Face model loading with resolved provenance | [`models/loader.py`](src/autodistiller/models/loader.py) |
| Reproducible, hashable run configuration | [`config.py`](src/autodistiller/config.py) |
| Perplexity as a low-cost screening metric | [`evaluation/perplexity.py`](src/autodistiller/evaluation/perplexity.py) |
| Task + custom evaluation datasets | [`evaluation/datasets.py`](src/autodistiller/evaluation/datasets.py) |
| Baseline inference smoke test | [`evaluation/baseline_inference.py`](src/autodistiller/evaluation/baseline_inference.py) |
| Quality regression reporting | [`regression.py`](src/autodistiller/regression.py) |
| Model / dataset / library / hardware metadata | [`metadata/`](src/autodistiller/metadata) |
| Deployment benchmarking in the real serving runtime | [`serving/`](src/autodistiller/serving) |
| NVIDIA hardware profiles and format capabilities | [`metadata/profiles.py`](src/autodistiller/metadata/profiles.py) |
| Compression through existing backends | [`compression/`](src/autodistiller/compression) |
| Candidate generation and memory screening | [`candidates/`](src/autodistiller/candidates) |
| Constrained optimization and ranking | [`optimize/`](src/autodistiller/optimize) |

Phases 6–10 (experiment cache, Pareto analysis, export, llama.cpp) are on the
[roadmap](#roadmap) below.

### On isolation

AutoDistiller imports neither vLLM nor llmcompressor. Serving runtimes and compression backends
have heavy, mutually incompatible pins — llmcompressor caps `transformers<=5.14.1` while
AutoDistiller runs 5.15.x — and quietly downgrading a library would change the stack every recorded
baseline was measured against. So both run in their own environments: vLLM over HTTP, compression
as a subprocess. A useful side effect is that llama.cpp needs no new benchmark client in Phase 9,
because it speaks the same OpenAI API.

---

## Setup

AutoDistiller uses [uv](https://docs.astral.sh/uv/) as its standard project and dependency manager.

```bash
pip install autodistiller
```

Or, for development on the project itself:

```bash
uv sync
```

That creates the environment, installs everything from the committed `uv.lock`, and installs
AutoDistiller in editable mode. Then:

```bash
uv run autodistiller --help
```

### GPU builds

`pyproject.toml` points `torch` at the CUDA 12.8 index on Linux and Windows, which covers NVIDIA
GPUs through Blackwell (`sm_120`). For a CPU-only install:

```bash
UV_TORCH_BACKEND=cpu uv sync
```

Check what AutoDistiller detected:

```bash
uv run autodistiller env
```

---

## Quick start

### 1. Establish a baseline

```bash
uv run autodistiller evaluate --model Qwen/Qwen3-0.6B --task wikitext2 --limit 256
```

This loads the model, runs a greedy generation smoke test, scores perplexity, and writes a
complete run record to `runs/<run_id>/`.

### 2. Evaluate on your own data

Public benchmarks tell you about public benchmarks. Your eval set tells you whether a compressed
model is deployable for *your* use case.

```bash
uv run autodistiller evaluate \
  --model Qwen/Qwen3-0.6B \
  --task wikitext2 \
  --task mc:examples/datasets/deployment_qa.jsonl \
  --task ppl:path/to/your/domain_corpus.txt
```

### 3. Benchmark it in the runtime you will deploy

```bash
uv run autodistiller benchmark --endpoint http://localhost:8000 --backend vllm --concurrency 1,4,16
```

Reports TTFT, per-token decode latency, throughput and VRAM at each concurrency level. These are
deployment claims, measured inside vLLM. Running on Windows? See
[docs/vllm-on-wsl.md](docs/vllm-on-wsl.md).

### 4. Compress it

```bash
uv run autodistiller methods
```

```bash
uv run autodistiller compress --model Qwen/Qwen3-0.6B --method int4-awq --calibration wikitext2
```

`methods` lists what this GPU and serving backend can actually use — hardware support (does the
silicon have the tensor cores?) and backend support (does the runtime have a kernel?) are checked
separately, because a method can pass one and fail the other.

### 5. See the search space before spending on it

```bash
uv run autodistiller candidates --model Qwen/Qwen3-0.6B --max-vram 8GiB --concurrency 16
```

Enumerates compression method × context length × KV cache dtype, then filters by hardware support,
backend support, and estimated memory. Rejected configurations are listed *with their reasons* —
a shorter list with no explanation is not an explainable search space.

Memory is estimated from the model's config alone, so a whole search space costs a few kilobytes
rather than a download per candidate. On Qwen3-0.6B the estimates land within 2% of the artifacts
that were actually produced, and the KV-cache figure matches what vLLM reports at startup.

### 6. Let it decide for you

```bash
uv run autodistiller optimize   --model Qwen/Qwen3-0.6B   --backend vllm   --max-vram 8GiB   --min-quality 95   --objective throughput   --calibration wikitext2   --launch-preset wsl-vllm
```

Generates candidates, screens them on estimated memory, compresses the survivors, scores quality
against the baseline, benchmarks whatever still qualifies in a real server, and ranks the rest.
Each stage is more expensive than the last, so a candidate that fails cheaply never costs anything
more.

The objective sets the search order, which is what makes `--stop-early` (the default) honest: under
`throughput` the most compressed candidate is tried first, so the first one that holds quality is
also the fastest one that holds quality. Under `quality` the order reverses.

Without `--launch-preset`, the deployment stage is skipped and ranking falls back to what can be
measured without a server (quality, size). Latency and throughput constraints require it, and the
command says so rather than silently ignoring them.

### 7. Check a candidate against the baseline

```bash
uv run autodistiller compare <baseline_run_id> <candidate_run_id> --min-retention 0.95
```

Exits non-zero when quality did not hold, so it drops straight into CI.

### 8. Browse what you have measured

```bash
uv run autodistiller runs
```

```bash
uv run autodistiller show <run_id> --verbose
```

---

## Tasks

Run `uv run autodistiller tasks` for the live list.

**Presets** — `wikitext2`, `wikitext103`, `arc_easy`, `arc_challenge`, `hellaswag`, `piqa`

**Your own data**

| Syntax | Meaning |
|---|---|
| `ppl:corpus.txt` | perplexity over a local text file |
| `ppl:corpus.jsonl` | perplexity over a local JSONL corpus (`text` field) |
| `mc:evals.jsonl` | multiple choice over a local JSONL file |

The multiple-choice schema is one JSON object per line:

```json
{"id": "q1", "context": "Question: What is 2+2?\nAnswer:", "choices": [" 3", " 4"], "answer_index": 1}
```

Choices keep their own leading space: they are appended to the context verbatim so tokenization
matches what a real prompt would produce.

For full control, use a config file — see
[`examples/configs/baseline.yaml`](examples/configs/baseline.yaml):

```bash
uv run autodistiller evaluate --config examples/configs/baseline.yaml
```

---

## Metrics

**Perplexity** (`perplexity`, `nll_per_token`, `bits_per_byte`) — strided windows, so every token is
scored exactly once and with as much left context as the window allows. Naive chunking scores the
first token of every chunk with no context at all, which inflates the number. `bits_per_byte` is
tokenizer-independent and stays meaningful when a candidate ships a different tokenizer.

**Multiple choice** (`acc`, `acc_norm`) — each candidate answer is scored by log-probability and the
highest-scoring one wins. No sampling, so results are exactly reproducible. `acc_norm` normalizes by
answer length so longer answers are not penalized for having more tokens.

Both report a standard error, which `compare` uses to distinguish a real regression from noise.

---

## Why every run records so much

A run record carries the config, the resolved model commit, an architecture fingerprint, dataset
content fingerprints, library versions, and the hardware it ran on. That is not bookkeeping for its
own sake:

- **Comparability is checkable.** `compare` refuses to score a comparison where the two runs used
  different data, and warns when the hardware or software stack moved.
- **Phase 6's experiment cache needs it.** Reusing a measurement is only safe if you can prove the
  inputs were identical. The config hash and fingerprints are that proof.
- **It is the long-term differentiator.** The defensible asset is measured knowledge: which
  configurations work on which models, GPUs, backends and software stacks.

### On performance numbers

The baseline inference step reports tokens/sec. It is tagged `runtime: "transformers"` and
`is_deployment_claim: false`, and the CLI says so every time it prints them. Transformers timings
are a smoke test, not serving performance. Deployment numbers get measured inside the deployment
backend — that is Phase 2.

---

## Reproducibility

Runs are seeded (Python, NumPy, torch, CUDA), cuDNN autotuning is pinned off, and the resolved
config is written next to every result:

```bash
uv run autodistiller evaluate --model Qwen/Qwen3-0.6B --save-config my-baseline.yaml
uv run autodistiller evaluate --config my-baseline.yaml   # same numbers
```

The config hash covers everything that can move a metric and excludes what cannot (`label`,
`output_dir`).

---

## Development

```bash
uv sync
```

```bash
uv run pytest
```

```bash
uv run ruff check . && uv run ruff format --check .
```

The suite runs on CPU in a few seconds against a tiny model built in-process, so the full
load → evaluate → record → compare path is covered without downloading anything.

See [CONTRIBUTING.md](CONTRIBUTING.md) for the full workflow, and
[CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) for community expectations. Security issues go through
[SECURITY.md](SECURITY.md) rather than the public tracker.

---

## Releasing

Releases publish to PyPI automatically via
[Trusted Publishing](https://docs.pypi.org/trusted-publishers/) — there is no API token to store
or rotate.

1. Bump `version` in `pyproject.toml` (`__version__` reads it from package metadata, so there is
   nothing else to keep in sync).
2. Commit, then tag and push: `git tag v0.2.0 && git push --tags`
3. Publish a GitHub release for that tag.

[`release.yml`](.github/workflows/release.yml) then re-runs the full test suite, checks the tag
matches the packaged version, builds an sdist and wheel with `uv build`, and uploads. PyPI version
numbers can never be reused, so both gates run before anything is uploaded.

## Roadmap

| Phase | Scope | Status |
|---|---|---|
| 1 | Evaluation engine | **done** |
| 2 | Hardware & deployment profiling (vLLM) | **done** |
| 3 | Compression backend integration (LLM Compressor adapters) | **done** |
| 4 | Candidate generator | **done** |
| 5 | Constrained optimization | **done** |
| 6 | Persistent experiment cache | next |
| 7 | Pareto analysis | planned |
| 8 | Export & reproducibility | planned |
| 9 | Multi-backend expansion (llama.cpp) | planned |
| 10 | Post-v1 research (distillation, pruning, Bayesian search) | post-v1 |

### v1.0 target

Hugging Face models, NVIDIA GPUs, evaluation-first workflow, vLLM as the first deployment backend,
INT4/INT8/AWQ/GPTQ and selected FP8 paths through existing backends, constrained enumeration rather
than advanced AutoML, a persistent experiment cache, Pareto analysis, and reproducible export.

The `optimize` command from the roadmap arrives once Phases 2–5 land:

```bash
uv run autodistiller optimize \
  --model Qwen/Qwen3-4B \
  --backend vllm \
  --max-vram 8GB \
  --min-quality 95 \
  --objective throughput
```

It will call this same evaluation engine underneath.

---

## Relationship to AutoTrainer

AutoTrainer and AutoDistiller are separate projects. AutoTrainer covers training and fine-tuning;
AutoDistiller covers deployment optimization. They share interfaces where useful (model metadata,
evaluation, experiment tracking, hardware detection) and stay interoperable, but the repositories
are not merged.

## License

Apache-2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).
