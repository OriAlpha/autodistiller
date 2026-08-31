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

SCHEMA_VERSION = 2
"""2 adds the Phase 6 cache keys. Version 1 records still load: the new
fields are optional, and a record without them simply never matches a
cache lookup, so it is history rather than a reusable result."""


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


class CompressionRecipe(BaseModel):
    """What was asked for.

    Complete enough to reproduce the artifact: algorithm, scheme, what was left
    alone, and a fingerprint of the calibration data. Calibration text changes
    the produced weights, so two artifacts with the same method but different
    calibration are not the same artifact.
    """

    method: str
    scheme: str
    algorithm: str
    weight_bits: int
    activation_bits: int
    ignore: list[str] = Field(default_factory=list)
    needs_calibration: bool = False
    n_calibration_samples: int = 0
    max_seq_length: int = 0
    calibration_fingerprint: str | None = None

    @property
    def label(self) -> str:
        return f"{self.method} ({self.scheme})"

    def describe(self) -> str:
        return f"W{self.weight_bits}A{self.activation_bits}"


class CompressionArtifact(BaseModel):
    """What was produced, and by which stack."""

    recipe: CompressionRecipe
    backend: str
    source_model: str
    output_dir: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    artifact_bytes: int | None = None
    duration_s: float = 0.0
    versions: dict[str, str] = Field(default_factory=dict)

    @property
    def artifact_gib(self) -> float | None:
        return self.artifact_bytes / 1024**3 if self.artifact_bytes else None

    def compression_ratio(self, baseline_bytes: int | None) -> float | None:
        """How much smaller than the uncompressed weights, if known."""
        if not baseline_bytes or not self.artifact_bytes:
            return None
        return baseline_bytes / self.artifact_bytes


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
    throughput_efficiency: float | None = Field(
        default=None,
        description=(
            "Measured throughput over what the per-token timings imply. Near 1 is "
            "healthy; far below means wall-clock time went somewhere other than serving."
        ),
    )
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)

    @property
    def throughput(self) -> float:
        """The rate this phase measured, whichever one it was.

        An endpoint that answers in a single response emits no output tokens, so
        tokens per second is zero on every rung and separates nothing. What it
        does produce is answers, and requests per second is then the rate that
        tells two configurations apart.
        """
        return self.output_tokens_per_s or self.requests_per_s

    @property
    def throughput_unit(self) -> str:
        return "tok/s" if self.output_tokens_per_s else "req/s"

    @property
    def latency_p50(self) -> float | None:
        """What one user waits for an answer.

        Time to first token when there is a stream, and end-to-end latency when
        there is not -- the same question, asked of a protocol that answers all
        at once.
        """
        if self.ttft is not None:
            return self.ttft.p50
        return self.request_latency.p50 if self.request_latency is not None else None


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
    prompt_fingerprint: str | None = None
    """Set when a real prompt was used instead of the generated filler."""
    max_tokens: int = 0
    phases: list[ConcurrencyResult] = Field(default_factory=list)
    device_total_vram_bytes: int | None = None

    @property
    def peak_vram_bytes(self) -> int | None:
        peaks = [p.peak_vram_bytes for p in self.phases if p.peak_vram_bytes]
        return max(peaks) if peaks else None

    @property
    def best_throughput(self) -> ConcurrencyResult | None:
        """The rung that served the most, by whichever rate was measured.

        Tokens per second first, because that is what a generation benchmark is
        ranked on. An endpoint that answers in a single response produces no
        output tokens at all, so every rung ties at zero and the tuple falls
        through to requests per second -- without which this returns the first
        rung, which is the *slowest*, and a throughput floor gets checked
        against the wrong measurement.
        """
        return max(
            self.phases,
            key=lambda p: (p.output_tokens_per_s, p.requests_per_s),
            default=None,
        )

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

    # Phase 6. Two keys because a record can hold two independently expensive
    # results; see :mod:`autodistiller.cache`.
    experiment_key: str | None = None
    benchmark_key: str | None = None
    candidate_id: str | None = Field(
        default=None, description="Which optimizer candidate produced this record"
    )

    model: ModelInfo
    hardware: HardwareInfo
    environment: EnvironmentInfo

    tasks: list[TaskResult] = Field(default_factory=list)
    baseline_inference: InferenceResult | None = None
    deployment: DeploymentBenchmark | None = None
    compression: CompressionArtifact | None = None
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
