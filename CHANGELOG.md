# Changelog

All notable changes to this project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

[Unreleased]: https://github.com/OriAlpha/Autodistiller/compare/v0.1.1...HEAD
[0.1.1]: https://github.com/OriAlpha/Autodistiller/releases/tag/v0.1.1
[0.1.0]: https://github.com/OriAlpha/Autodistiller/releases/tag/v0.1.0
