"""Experiment identity.

A cached result is only reusable if *everything that could have changed the
number* is the same. The roadmap lists what that means: the model, the hardware,
the backend, the compression method, the calibration data, the software
versions, and the benchmark configuration. This module turns that list into two
keys.

Two, not one, because a run record can carry two independently expensive
results. An evaluation depends on the model, the tasks and the datasets; a
deployment benchmark depends on the served weights, the launch settings and the
request shape. Changing the concurrency sweep should not discard a perplexity
measurement, and re-screening on a new dataset should not discard a throughput
curve. One key each keeps them separable.

Both keys are plain hex strings over sorted JSON, so they mean the same thing on
another machine, in another process, or in a shared results database later.
"""

from __future__ import annotations

from .metadata.environment import EnvironmentInfo
from .metadata.hardware import HardwareInfo
from .metadata.hashing import hash_obj

KEY_VERSION = 1
"""Bumped when the key's *composition* changes.

Included in the hash so that a change to what the key covers invalidates old
entries rather than silently matching them under different rules.
"""


def experiment_key(
    config_fingerprint: str,
    hardware: HardwareInfo,
    environment: EnvironmentInfo,
) -> str:
    """Identity of an evaluation.

    ``config_fingerprint`` already covers the model, the tasks, the datasets,
    the seed and any compression spec. What it cannot know is the machine and
    the stack, so those come from the other two.
    """
    return hash_obj(
        {
            "v": KEY_VERSION,
            "config": config_fingerprint,
            "hardware": hardware.fingerprint,
            "environment": environment.cache_fingerprint,
        }
    )


def benchmark_key(
    *,
    served_model: str,
    backend: str,
    hardware: HardwareInfo,
    environment: EnvironmentInfo,
    settings: dict,
) -> str:
    """Identity of a deployment benchmark.

    ``served_model`` is the path or id of the weights actually served, which for
    a compressed candidate is the artifact directory -- and artifact directories
    are content-addressed by recipe, so two different recipes can never share
    one. ``settings`` carries the request shape and any serving flags: prompt
    length, output length, the concurrency sweep, context length, KV dtype.
    """
    return hash_obj(
        {
            "v": KEY_VERSION,
            "served_model": served_model,
            "backend": backend,
            "hardware": hardware.fingerprint,
            "environment": environment.cache_fingerprint,
            "settings": settings,
        }
    )


__all__ = ["KEY_VERSION", "benchmark_key", "experiment_key"]
