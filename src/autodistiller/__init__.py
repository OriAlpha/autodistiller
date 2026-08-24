"""AutoDistiller: automated LLM deployment optimization.

Everything re-exported here is the supported API and follows semantic
versioning: it will not change incompatibly before 2.0. Anything reached by
importing a submodule directly is internal and may move between minor releases.

The surface mirrors what the CLI does, in the order it does it::

    generate_candidates   what is worth measuring
    run_compression       build one compressed artifact
    run_evaluation        measure quality
    run_deployment_benchmark   measure a running server
    Optimizer             all of it, under constraints
    ParetoReport          the trade-offs behind a recommendation
    export                make a result deployable and reproducible

Provenance types (``RunRecord``, ``RunStore``, ``ExportManifest``) are public
because a stored result outlives the version that produced it, and reading one
back should not require reaching into internals.
"""

from importlib.metadata import version

from .cache import benchmark_key, experiment_key
from .candidates.generator import Candidate, CandidateSet, generate_candidates
from .compression.methods import METHODS, CompressionMethod, resolve_method
from .compression.pipeline import run_compression
from .config import (
    BaselineInferenceSpec,
    CompressionSpec,
    DatasetSpec,
    DeploymentSpec,
    ModelSpec,
    MultipleChoiceTask,
    PerplexityTask,
    RunConfig,
)
from .export import ExportManifest, export
from .metadata.environment import EnvironmentInfo, collect_environment
from .metadata.hardware import HardwareInfo, detect_hardware
from .optimize.constraints import Constraints, Objective
from .optimize.pareto import ParetoReport
from .optimize.pipeline import (
    CandidateOutcome,
    OptimizationResult,
    Optimizer,
    QualityComparison,
    quality_retention,
)
from .regression import RegressionReport, compare_runs
from .results import (
    CompressionArtifact,
    DeploymentBenchmark,
    MetricValue,
    ModelInfo,
    RunRecord,
    TaskResult,
)
from .runner import run_evaluation
from .serving.backends import Backend, resolve_backend
from .serving.benchmark import run_deployment_benchmark
from .store import RunStore

# Single source of truth is pyproject.toml; nothing to keep in sync at release.
__version__ = version("autodistiller")

__all__ = [
    "METHODS",
    "Backend",
    "BaselineInferenceSpec",
    "Candidate",
    "CandidateOutcome",
    "CandidateSet",
    "CompressionArtifact",
    "CompressionMethod",
    "CompressionSpec",
    "Constraints",
    "DatasetSpec",
    "DeploymentBenchmark",
    "DeploymentSpec",
    "EnvironmentInfo",
    "ExportManifest",
    "HardwareInfo",
    "MetricValue",
    "ModelInfo",
    "ModelSpec",
    "MultipleChoiceTask",
    "Objective",
    "OptimizationResult",
    "Optimizer",
    "ParetoReport",
    "PerplexityTask",
    "QualityComparison",
    "RegressionReport",
    "RunConfig",
    "RunRecord",
    "RunStore",
    "TaskResult",
    "__version__",
    "benchmark_key",
    "collect_environment",
    "compare_runs",
    "detect_hardware",
    "experiment_key",
    "export",
    "generate_candidates",
    "quality_retention",
    "resolve_backend",
    "resolve_method",
    "run_compression",
    "run_deployment_benchmark",
    "run_evaluation",
]
