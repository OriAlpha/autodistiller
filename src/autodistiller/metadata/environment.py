"""Software environment capture.

A perplexity number is meaningless without the stack that produced it: a
transformers upgrade can shift results, and a different CUDA build can shift
them again. Phase 6's experiment cache keys on this, so it lives here from the
start rather than being retrofitted.
"""

from __future__ import annotations

import platform
import sys
from importlib import metadata as importlib_metadata

from pydantic import BaseModel, Field

from .hashing import hash_obj

TRACKED_PACKAGES = (
    "autodistiller",
    "torch",
    "transformers",
    "tokenizers",
    "datasets",
    "accelerate",
    "safetensors",
    "huggingface-hub",
    "numpy",
)

CACHE_RELEVANT_PACKAGES = (
    "autodistiller",
    "torch",
    "transformers",
    "tokenizers",
    "datasets",
)
"""Packages whose version can move a measured number.

The cache key covers these rather than the whole environment. Keying on
everything is defensible in theory and useless in practice: a `safetensors`
patch bump would discard every cached result without changing any of them.
These five can. `autodistiller` is on the list because it *is* the measurement
code, and `datasets` because it decides which text gets scored.
"""


class EnvironmentInfo(BaseModel):
    """Versions of everything that can move a metric."""

    python_version: str
    platform: str
    packages: dict[str, str] = Field(default_factory=dict)
    torch_version: str | None = None
    cuda_version: str | None = None
    cudnn_version: str | None = None

    @property
    def fingerprint(self) -> str:
        return hash_obj(self.model_dump())

    @property
    def cache_fingerprint(self) -> str:
        """Identity of the stack, for deciding whether a cached result still holds.

        Narrower than :attr:`fingerprint`: the OS build string and packages that
        cannot move a metric are left out, so an unrelated dependency bump does
        not throw the cache away. CUDA is in, because kernel changes do move
        numbers.
        """
        return hash_obj(
            {
                "python": ".".join(self.python_version.split(".")[:2]),
                "packages": {
                    name: version
                    for name, version in self.packages.items()
                    if name in CACHE_RELEVANT_PACKAGES
                },
                "cuda": self.cuda_version,
            }
        )


def _package_version(name: str) -> str | None:
    try:
        return importlib_metadata.version(name)
    except importlib_metadata.PackageNotFoundError:
        return None


def collect_environment() -> EnvironmentInfo:
    packages = {name: v for name in TRACKED_PACKAGES if (v := _package_version(name))}

    torch_version = cuda_version = cudnn_version = None
    try:
        import torch

        torch_version = torch.__version__
        cuda_version = torch.version.cuda
        if torch.backends.cudnn.is_available():
            raw = torch.backends.cudnn.version()
            cudnn_version = str(raw) if raw is not None else None
    except Exception:  # torch missing or broken: record what we can, don't crash
        pass

    return EnvironmentInfo(
        python_version=sys.version.split()[0],
        platform=platform.platform(),
        packages=packages,
        torch_version=torch_version,
        cuda_version=cuda_version,
        cudnn_version=cudnn_version,
    )
