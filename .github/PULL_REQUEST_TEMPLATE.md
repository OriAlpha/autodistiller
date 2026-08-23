## What this changes

<!-- And why. Link the issue if there is one. -->

## Checklist

- [ ] `uv run pytest` passes
- [ ] `uv run ruff check . && uv run ruff format --check . && uv run mypy` pass
- [ ] Tests added for new behavior, or a failing test for a bug fix
- [ ] `CHANGELOG.md` updated under Unreleased, if user-visible

## If this touches measurement

- [ ] Anything that can move a metric is recorded in the run record
- [ ] Performance numbers say which runtime produced them
