"""Result schema.

Everything AutoDistiller measures ends up in a ``RunRecord``: a single JSON
document containing the config that produced it, the metrics, and enough
provenance (model, dataset, library, hardware) to decide whether two records are
comparable at all. Later phases add candidate/compression fields; the
``schema_version`` is here so those additions stay readable.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from .config import RunConfig
from .metadata.environment import EnvironmentInfo
from .metadata.hardware import HardwareInfo

SCHEMA_VERSION = 1


class MetricValue(BaseModel):
    """One number, plus everything needed to compare it to another one."""

    name: str
    value: float
    higher_is_better: bool
    stderr: float | None = None
    unit: str | None = None

    def format(self) -> str:
        text = f"{self.value:.4f}" if abs(self.value) < 1e4 else f"{self.value:.4g}"
        if self.stderr is not None:
            text += f" ± {self.stderr:.4f}"
        if self.unit:
            text += f" {self.unit}"
        return text


class TaskResult(BaseModel):
    """Outcome of a single evaluation task."""

    name: str
    kind: str
    metrics: list[MetricValue] = Field(default_factory=list)
    n_samples: int = 0
    n_tokens: int = 0
    duration_s: float = 0.0
    dataset_fingerprint: str | None = None
    details: dict[str, Any] = Field(default_factory=dict)

    def metric(self, name: str) -> MetricValue | None:
        return next((m for m in self.metrics if m.name == name), None)

    @property
    def primary_metric(self) -> MetricValue | None:
        """The metric a headline comparison should use."""
        preferred = ("acc_norm", "acc", "word_perplexity", "perplexity")
        for name in preferred:
            if (found := self.metric(name)) is not None:
                return found
        return self.metrics[0] if self.metrics else None


class GenerationSample(BaseModel):
    prompt: str
    output: str | None = None
    n_prompt_tokens: int = 0
    n_generated_tokens: int = 0
    latency_s: float = 0.0

    @property
    def tokens_per_second(self) -> float:
        return self.n_generated_tokens / self.latency_s if self.latency_s > 0 else 0.0


class InferenceResult(BaseModel):
    """Transformers-level generation smoke test.

    ``runtime`` and ``is_deployment_claim`` are recorded explicitly: the roadmap
    forbids presenting Transformers timings as serving-backend performance, and
    the schema should make that impossible to do by accident.
    """

    runtime: str = "transformers"
    is_deployment_claim: bool = False
    samples: list[GenerationSample] = Field(default_factory=list)
    mean_tokens_per_second: float = 0.0
    total_duration_s: float = 0.0
    peak_vram_bytes: int | None = None


class LatencyStats(BaseModel):
    """Distribution of a latency measurement, in seconds."""

    mean: float
    p50: float
    p90: float
    p99: float
    min: float
    max: float

    def format_ms(self) -> str:
        return f"p50 {self.p50 * 1000:.0f}ms / p90 {self.p90 * 1000:.0f}ms"


class ConcurrencyResult(BaseModel):
    """One rung of the concurrency sweep."""

    concurrency: int
    n_requests: int
    n_failed: int = 0
    duration_s: float
    ttft: LatencyStats | None = None
    tpot: LatencyStats | None = Field(
        default=None, description="Time per output token, excluding prefill"
    )
    request_latency: LatencyStats | None = None
    total_output_tokens: int = 0
    output_tokens_per_s: float = 0.0
    requests_per_s: float = 0.0
    mean_prompt_tokens: float = 0.0
    peak_vram_bytes: int | None = None
    errors: list[str] = Field(default_factory=list)


class DeploymentBenchmark(BaseModel):
    """Performance measured inside a real serving runtime.

    Unlike :class:`InferenceResult`, these numbers *are* deployment claims:
    they came from the runtime the user would actually deploy.
    """

    backend: str
    runtime_version: str | None = None
    endpoint: str
    served_model: str
    is_deployment_claim: bool = True
    prompt_tokens_requested: int = 0
    max_tokens: int = 0
    phases: list[ConcurrencyResult] = Field(default_factory=list)
    device_total_vram_bytes: int | None = None

    @property
    def peak_vram_bytes(self) -> int | None:
        peaks = [p.peak_vram_bytes for p in self.phases if p.peak_vram_bytes]
        return max(peaks) if peaks else None

    @property
    def best_throughput(self) -> ConcurrencyResult | None:
        return max(self.phases, key=lambda p: p.output_tokens_per_s, default=None)

    @property
    def single_stream(self) -> ConcurrencyResult | None:
        """The concurrency-1 rung: what one user experiences."""
        return next((p for p in self.phases if p.concurrency == 1), None)


class ModelInfo(BaseModel):
    """Resolved identity of the weights that were actually loaded."""

    model_config = ConfigDict(protected_namespaces=())

    id: str
    revision: str | None = None
    resolved_commit: str | None = None
    architecture: str | None = None
    dtype: str | None = None
    device: str | None = None
    n_parameters: int | None = None
    context_length: int | None = None
    vocab_size: int | None = None
    weights_size_bytes: int | None = None
    architecture_fingerprint: str | None = None
    is_local: bool = False

    @property
    def n_parameters_b(self) -> float | None:
        return self.n_parameters / 1e9 if self.n_parameters else None


class RunRecord(BaseModel):
    """A complete, self-describing evaluation result."""

    schema_version: int = SCHEMA_VERSION
    run_id: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    status: Literal["ok", "failed"] = "ok"
    error: str | None = None

    config: RunConfig
    config_fingerprint: str
    model: ModelInfo
    hardware: HardwareInfo
    environment: EnvironmentInfo

    tasks: list[TaskResult] = Field(default_factory=list)
    baseline_inference: InferenceResult | None = None
    deployment: DeploymentBenchmark | None = None
    total_duration_s: float = 0.0

    def task(self, name: str) -> TaskResult | None:
        return next((t for t in self.tasks if t.name == name), None)

    def to_json(self) -> str:
        return json.dumps(self.model_dump(mode="json"), indent=2, ensure_ascii=False)

    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json(), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: Path | str) -> RunRecord:
        return cls.model_validate_json(Path(path).read_text(encoding="utf-8"))
