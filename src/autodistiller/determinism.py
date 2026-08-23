"""Seeding and determinism controls.

Phase 1's milestone is a *trustworthy* baseline. A baseline that moves by 0.3
perplexity between identical runs cannot be used to judge whether a quantized
candidate regressed, so runs are seeded and cuDNN autotuning is pinned.
"""

from __future__ import annotations

import os
import random


def seed_everything(seed: int, *, deterministic_algorithms: bool = False) -> int:
    """Seed Python, NumPy and torch. Returns the seed for logging."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        # Fixed algorithm choice beats fastest algorithm choice when the point
        # of the run is comparability.
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        if deterministic_algorithms:
            # Opt-in: some attention kernels have no deterministic
            # implementation and will raise rather than silently vary.
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
            torch.use_deterministic_algorithms(True, warn_only=True)
    except ImportError:
        pass

    return seed
