# Contributing to AutoDistiller

Thanks for taking a look. This is an early project, so the most useful
contributions right now are bug reports with a reproducible run record, and
support for evaluation tasks or hardware we have not tested against.

By participating you agree to abide by the [Code of Conduct](CODE_OF_CONDUCT.md).

## Getting set up

AutoDistiller uses [uv](https://docs.astral.sh/uv/):

```bash
uv sync
```

No GPU? Install the CPU build of torch instead:

```bash
UV_TORCH_BACKEND=cpu uv sync
```

Confirm what got detected:

```bash
uv run autodistiller env
```

## Before you open a pull request

```bash
uv run pytest
```

```bash
uv run ruff check . && uv run ruff format . && uv run mypy
```

CI runs exactly these on Python 3.10 and 3.12. The suite takes a few seconds and
needs no network: it builds a tiny model in-process, so the whole
load, evaluate, record, compare path is covered without downloading weights.

Mark anything that does need a download or a GPU:

```python
@pytest.mark.slow   # needs a model or dataset download
@pytest.mark.gpu    # needs a CUDA device
```

## What this project does and does not do

AutoDistiller is an orchestration layer. It does not implement quantization
kernels and does not reimplement AWQ, GPTQ or any other mature algorithm. It
composes existing backends, measures them under real deployment conditions, and
picks a configuration. Pull requests that add a compression algorithm from
scratch will likely be redirected toward an adapter around an existing library.

Two rules matter more than style:

**A measurement carries its provenance.** Anything that can move a number
belongs in the run record: config, resolved model commit, dataset fingerprint,
library versions, hardware. If a comparison cannot be shown to be valid, it
should refuse to report rather than report something misleading.

**Performance numbers name their runtime.** Timings from Transformers are a
smoke test. Serving performance gets measured inside the serving backend. Do not
present one as the other.

## Adding an evaluation task

Most new tasks need no new code. A local dataset works out of the box:

```bash
uv run autodistiller evaluate --model <model> --task mc:my_evals.jsonl
```

To add a built-in preset, add a factory to `PRESETS` in
[`evaluation/registry.py`](src/autodistiller/evaluation/registry.py). If the hub
dataset has its own schema, add a row transform to
[`evaluation/preprocessors.py`](src/autodistiller/evaluation/preprocessors.py)
and reference it by name, so configs stay serializable and hashable.

## Commits and pull requests

Write commit subjects in the imperative, under ~72 characters, and explain *why*
in the body when it is not obvious. Keep one logical change per pull request.

New behavior needs a test. Bug fixes need a test that fails before the fix.

## Reporting a bug

Include the output of `uv run autodistiller env`, and the run record
(`runs/<run_id>/record.json`) if a run is involved. Between them they pin down
the hardware, library versions and exact config, which is usually most of the
diagnosis.

Security issues go through [SECURITY.md](SECURITY.md) instead, not the public
issue tracker.

## Releases

Maintainers only. Bump `version` in `pyproject.toml`, tag `vX.Y.Z`, and publish a
GitHub release; CI verifies the tag matches the packaged version, runs the tests
and uploads to PyPI.
