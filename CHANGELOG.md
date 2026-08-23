# Changelog

All notable changes to this project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.0] - 2026-08-23

Phases 2 and 3 of the roadmap: deployment profiling and compression.

### Added

- **Deployment benchmarking** (`autodistiller benchmark`). Drives a running
  OpenAI-compatible server and reports TTFT, per-output-token decode latency,
  throughput and VRAM across a concurrency sweep. Results are tagged
  `is_deployment_claim=True`, in contrast to Phase 1's Transformers smoke test.
  AutoDistiller imports no serving runtime; it measures a server it did not
  start, so llama.cpp will need no new client.
- **Hardware profiles** (`autodistiller profiles`). Numeric-format support is
  derived from compute capability rather than a list of card names, so the rule
  stays correct for GPUs nobody has added yet.
- **Compression** (`autodistiller compress`, `autodistiller methods`) through
  llmcompressor: `int8`, `int8-weight-only`, `int4-gptq`, `int4-awq`, `fp8`,
  `fp8-static`. Hardware support and backend support are reported separately,
  because a method the silicon can run may have no kernel in the serving
  runtime. Rejected methods come back with reasons.
- Compression runs in an isolated environment. llmcompressor caps
  `transformers<=5.14.1` while AutoDistiller runs 5.15.x, and downgrading it in
  the main environment would change the stack every recorded baseline was
  measured against.
- Calibration text is supplied by AutoDistiller rather than named for the
  backend to fetch, so it is fingerprinted like evaluation data and forms part
  of the recipe identity.
- `compressed-tensors` dependency, so the Phase 1 evaluator can score compressed
  artifacts and the existing regression report works on them unchanged.
- `docs/vllm-on-wsl.md`: the six undocumented blockers to running vLLM on WSL2.

### Fixed

- VRAM is sampled through NVML rather than `torch.cuda.mem_get_info`, which
  describes only the calling process's CUDA context. Benchmarking a server in
  another process it reported a flat 1.11 GiB against nvidia-smi's 6.8 GiB.

### Verified

Qwen3-0.6B on an RTX 5070: FP8 in 26s (1.11 -> 0.71 GiB), served by vLLM at
1.19-1.30x throughput, wikitext-2 perplexity 22.69 -> 23.05 (98.45% retention).
All six compression methods produce loadable artifacts.

## [0.1.1] - 2026-08-23

### Fixed

- Ship the `py.typed` marker in the wheel. 0.1.0 advertised the
  `Typing :: Typed` classifier but omitted the marker, so type checkers
  silently ignored the package's annotations for anyone installing from PyPI.

## [0.1.0] - 2026-08-23

First release. Phase 1 of the roadmap: the evaluation engine.

### Added

- Hugging Face model loading, recording the resolved commit, parameter count,
  context length and an architecture fingerprint.
- `RunConfig`: a hashable, YAML-serializable description of a run. The hash
  covers everything that can move a metric and excludes what cannot.
- Strided perplexity with `nll_per_token` and `bits_per_byte`. Every token is
  scored once, with context; cross-entropy is chunked to bound peak memory.
- Log-likelihood multiple choice reporting `acc` and `acc_norm`, over built-in
  hub presets (`wikitext2`, `wikitext103`, `arc_easy`, `arc_challenge`,
  `hellaswag`, `piqa`) or your own JSONL eval sets.
- Baseline inference smoke test, tagged as the Transformers runtime and
  explicitly not a deployment performance claim.
- Quality regression reporting: direction-aware retention, noise significance
  from standard errors, and comparability checks that refuse to score a
  comparison made on different data.
- Run store keyed by config fingerprint.
- Metadata capture for hardware, CUDA, and library versions.
- CLI: `env`, `tasks`, `evaluate`, `compare`, `runs`, `show`.

[Unreleased]: https://github.com/OriAlpha/Autodistiller/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/OriAlpha/Autodistiller/releases/tag/v0.2.0
[0.1.1]: https://github.com/OriAlpha/Autodistiller/releases/tag/v0.1.1
[0.1.0]: https://github.com/OriAlpha/Autodistiller/releases/tag/v0.1.0
