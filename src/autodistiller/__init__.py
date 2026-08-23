"""AutoDistiller: automated LLM deployment optimization.

Phase 1 provides the evaluation engine -- the trustworthy baseline every later
phase measures against.
"""

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

__version__ = "0.1.0"

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
