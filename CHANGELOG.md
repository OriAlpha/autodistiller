# Changelog

All notable changes to this project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

Phase 8 of the roadmap: export and reproducibility.

### Added

- **`autodistiller export <run>`**: writes a manifest, a `DEPLOY.md` note and
  the exact config beside the weights, so the directory you would serve is the
  one that explains itself. Nothing is converted -- llmcompressor already writes
  a Hugging Face directory, so the artifact *is* the export. What was missing
  was the provenance tying it to the measurements and a verified claim that a
  server can load it.
- **Deployability is checked, not asserted.** Weights present, tokenizer
  present, and the quantization format one the target runtime actually has a
  kernel for (read from `quantization_config.quant_method`, which is what the
  runtime dispatches on). `export` exits non-zero when the artifact would not
  load. An artifact that benchmarks beautifully and then cannot be served is the
  failure this exists to catch.
- The manifest carries the recipe, the calibration fingerprint, the metrics, the
  deployment benchmark, the hardware and the software stack, plus the commands
  that rebuild it. The saved config re-hashes to the same experiment.
- `optimize --export DIR` exports the winning configuration, so finding its run
  id by hand is not a step. `--copy-weights` assembles a bundle that can be
  moved; without it the bundle refers to the weights where they already are.
- Export recovers the recipe from the artifact's own sidecar when the run record
  has none -- the normal case for weights produced by `compress` directly, and
  for an evaluation of a compressed candidate, which records the artifact
  directory as its model rather than the recipe that built it.
- GGUF is reported as not applicable to compressed-tensors artifacts, with the
  reason: llama.cpp converts from unquantized Hugging Face weights and carries
  its own quantization schemes. For an uncompressed model the manifest gives the
  `convert_hf_to_gguf.py` and `llama-quantize` commands. Making llama.cpp a
  measured backend is Phase 9.

### Fixed

- The optimizer recorded a candidate's compression artifact only when it also
  benchmarked it, so an evaluation of a compressed candidate named the artifact
  directory as its model and said nothing about the recipe that produced it.
  The roadmap asks for the exact recipe to be saved with each result; it now is.

### Verified

Qwen3-0.6B on an RTX 5070: exporting an FP8 artifact passes all four checks and
emits a `vllm serve` command against the directory it wrote itself into, and
`optimize --export` produces the same bundle for the winning candidate.

## [0.3.0] - 2026-08-24

Phases 4 to 7 of the roadmap: AutoDistiller now makes the compression decision
itself, does not measure the same thing twice, and shows the trade-off rather
than only the winner.

### Added

#### Candidate generation (Phase 4)

- **`autodistiller candidates`**: the search space, before spending anything on
  it. Compression method x context length x KV cache dtype, filtered by hardware
  support, backend support and estimated memory.
- Model dimensions are read from the Hugging Face config alone, so a whole
  search space is screened for a few kilobytes rather than a download per
  candidate.
- Memory estimates account for embeddings and `lm_head` staying 16-bit, which is
  why a 4-bit model is never a quarter of its 16-bit size.
- Rejected candidates are kept with their reasons. "Your GPU has no FP8" is an
  answer; a silently shorter list is not.

#### Constrained optimization (Phase 5)

- **`autodistiller optimize`**: the roadmap's headline command. Candidates run
  through progressively more expensive stages -- memory estimate, compression,
  quality screen, deployment benchmark -- and are dropped as soon as they fail,
  so the costly work only lands on configurations still worth it.
- Constraints decide what may be recommended; the objective decides which of
  those wins, and also sets the search order. That ordering is what makes
  stopping early honest: under throughput the most compressed candidate is
  proven first, so the first qualifying candidate is the fastest qualifying one.
- Server lifecycle management, so a dozen candidates can be benchmarked in
  sequence without starting each one by hand.

#### Persistent experiment cache (Phase 6)

- **Nothing is measured twice.** `evaluate`, `compress` and `optimize` reuse an
  identical earlier result instead of repeating it. Each takes `--refresh` to
  measure again anyway.
- **Experiment keys** (`cache.py`) covering what the roadmap asks for: model,
  hardware, backend, compression method, calibration data, software versions and
  benchmark configuration. Two keys rather than one, because an evaluation and a
  deployment benchmark are independently expensive and independently
  invalidated: changing the concurrency sweep should not discard a perplexity
  measurement.
- The stack half of the key is narrow on purpose -- `autodistiller`, `torch`,
  `transformers`, `tokenizers`, `datasets`, CUDA, Python minor. Keying on every
  installed version is defensible and useless: a `safetensors` patch bump would
  discard every cached result without changing any of them.
- **Compressed artifacts are reused**, which is the largest saving: the
  optimizer repeats compression more than anything else, and it is the only
  stage measured in minutes. An `autodistiller-artifact.json` sidecar records
  what a directory holds; weights are verified present before anything is
  reused, so an interrupted run is not mistaken for a finished one.
- **Deployment benchmarks are persisted** and reused. `RunRecord` already had
  `deployment` and `compression` fields; the optimizer now fills them, so a
  repeated search pays only for the stages whose inputs actually changed.
- **`autodistiller history`**: what has been measured and what can be reused,
  with `--rebuild` and `--json`. Reads `runs/index.jsonl` -- one row per record,
  keys and summary, no metrics -- so a lookup does not parse every run ever
  done. Derived state: delete it and it rebuilds. Also the shape a shared
  benchmark database would want, being flat rows carrying a complete key rather
  than a local file layout.
- `autodistiller runs --model` to filter history by model.
- `RunRecord.schema_version` is 2, adding `experiment_key`, `benchmark_key` and
  `candidate_id`. Version 1 records still load; they are history rather than
  reusable results, and never match a lookup.

#### Pareto analysis (Phase 7)

- **Trade-offs, not a single score.** `optimize` reports the configurations
  where you cannot improve one axis without losing another, across quality,
  VRAM, latency and throughput, and names the best-quality, fastest, smallest
  and balanced options. Each says what choosing it costs.
- A candidate is never ranked on a number nobody measured. Treating an
  unmeasured throughput as best or worst would put it on the frontier for a
  reason that is not a measurement; those candidates are reported separately.
- An axis never mixes measured and estimated values. When nothing was
  benchmarked the VRAM axis falls back to estimates and is labelled as such.
- The frontier is drawn over the axes that have data. Keeping latency and
  throughput in the set when nothing was benchmarked would make every candidate
  incomparable and the frontier empty -- true, and useless.
- Ties on an objective break towards the frontier: two candidates can score
  identically while one is beaten outright on every other axis.
- `--no-pareto` prints only the winner. `--no-stop-early` is what makes the
  frontier worth looking at, since early stopping measures exactly one
  qualifying candidate.

### Fixed

- **Compressed artifacts could silently overwrite each other.** The output
  directory was named `<model>-<method>`, which ignores the calibration data,
  the `ignore` list and the sequence length -- all of which change the produced
  weights. Compressing one model and method with two different calibration sets
  wrote both to the same directory, the second replacing the first, while every
  record already written still pointed at the path and described weights that
  were no longer there. Directories are now content-addressed by the recipe.
- **Two runs in the same second shared a directory.** Run ids are timestamped to
  the second, so a repeated fast evaluation wrote the second record straight
  over the first. The store now disambiguates ids within its own namespace;
  `--refresh` on a cached evaluation reached this reliably.
- The optimizer attached each candidate's benchmark to that candidate's
  evaluation record. Context length is not a compression parameter, so
  candidates differing only in it share one artifact and one evaluation while
  having genuinely different benchmarks -- and all but the last were lost.
  Benchmarks are now recorded separately.

### Verified

Qwen3-0.6B on an RTX 5070: a repeated `evaluate` returns the cached record
instead of re-running, and a repeated `compress --method fp8` reuses the
artifact rather than spending 17.7s producing it again. A repeated `optimize`
reports its reused stages and reaches the same recommendation.

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

[Unreleased]: https://github.com/OriAlpha/Autodistiller/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/OriAlpha/Autodistiller/releases/tag/v0.3.0
[0.2.0]: https://github.com/OriAlpha/Autodistiller/releases/tag/v0.2.0
[0.1.1]: https://github.com/OriAlpha/Autodistiller/releases/tag/v0.1.1
[0.1.0]: https://github.com/OriAlpha/Autodistiller/releases/tag/v0.1.0
