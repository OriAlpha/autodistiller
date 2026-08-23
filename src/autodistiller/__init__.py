"""AutoDistiller: automated LLM deployment optimization.

Phase 1 provides the evaluation engine: the trustworthy baseline every later
phase measures against.
"""

from importlib.metadata import version

from .config import (
    BaselineInferenceSpec,
    DatasetSpec,
    ModelSpec,
    MultipleChoiceTask,
    PerplexityTask,
    RunConfig,
)
from .regression import RegressionReport, compare_runs
from .results import MetricValue, ModelInfo, RunRecord, TaskResult
from .runner import run_evaluation
from .store import RunStore

# Single source of truth is pyproject.toml; nothing to keep in sync at release.
__version__ = version("autodistiller")

__all__ = [
    "BaselineInferenceSpec",
    "DatasetSpec",
    "MetricValue",
    "ModelInfo",
    "ModelSpec",
    "MultipleChoiceTask",
    "PerplexityTask",
    "RegressionReport",
    "RunConfig",
    "RunRecord",
    "RunStore",
    "TaskResult",
    "__version__",
    "compare_runs",
    "run_evaluation",
]
