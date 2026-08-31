"""Reproducible run configuration.

A ``RunConfig`` is the complete, serializable description of an evaluation:
which model, which tasks, which decoding settings, which seed. Two identical
configs on identical hardware must produce identical numbers. Phase 6's
experiment cache will be built on that, so the config hash deliberately excludes
cosmetic fields (label, output directory) that do not affect results.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

from .metadata.hashing import hash_obj

DType = Literal["auto", "float32", "float16", "bfloat16"]


class StrictModel(BaseModel):
    """Reject unknown keys so a typo in a YAML config fails loudly."""

    model_config = ConfigDict(extra="forbid")


class ModelSpec(StrictModel):
    """Which weights to load and how."""

    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    id: str = Field(description="Hugging Face repo id or local path")
    revision: str | None = Field(default=None, description="Branch, tag or commit sha")
    dtype: DType = "auto"
    device: str = Field(default="auto", description="'auto', 'cpu', 'cuda', or 'cuda:1'")
    trust_remote_code: bool = False
    attn_implementation: str | None = Field(
        default=None, description="e.g. 'sdpa', 'eager', 'flash_attention_2'"
    )
    max_position_embeddings: int | None = Field(
        default=None, description="Override the context length reported by the model config"
    )


class DatasetSpec(StrictModel):
    """Where evaluation text comes from.

    ``source`` distinguishes a Hugging Face hub dataset from local files, which
    is how "task/custom evaluation datasets" stay a single code path.
    """

    source: Literal["hub", "jsonl", "text"] = "hub"
    path: str = Field(description="Hub dataset id, or a local file path for jsonl/text")
    name: str | None = Field(default=None, description="Hub dataset config name")
    split: str = "test"
    text_column: str = "text"
    limit: int | None = Field(default=None, ge=1, description="Cap on documents/examples")

    @field_validator("limit")
    @classmethod
    def _reject_zero(cls, value: int | None) -> int | None:
        if value is not None and value < 1:
            raise ValueError("limit must be >= 1")
        return value


class PerplexityTask(StrictModel):
    """Strided perplexity: the cheap screening metric of the roadmap."""

    kind: Literal["perplexity"] = "perplexity"
    name: str
    dataset: DatasetSpec
    max_length: int | None = Field(
        default=None, description="Window size; defaults to the model context length (capped)"
    )
    stride: int | None = Field(
        default=None,
        description="Window step. Defaults to max_length (disjoint windows, fastest).",
    )
    batch_size: int = Field(default=1, ge=1)
    doc_separator: str = "\n\n"


class MultipleChoiceTask(StrictModel):
    """Log-likelihood multiple choice: accuracy on a real downstream task."""

    kind: Literal["multiple_choice"] = "multiple_choice"
    name: str
    dataset: DatasetSpec
    batch_size: int = Field(default=1, ge=1)
    context_column: str = "context"
    choices_column: str = "choices"
    answer_column: str = "answer_index"
    preprocessor: str | None = Field(
        default=None,
        description=(
            "Named row transform applied before column mapping, for hub datasets whose "
            "schema is not already (context, choices, answer_index). Referenced by name "
            "rather than by function so configs stay serializable and hashable."
        ),
    )


class EmbeddingTask(StrictModel):
    """Sentence-pair similarity: the screening metric for an embedding model.

    Perplexity and multiple choice both need a model that predicts tokens. An
    encoder produces a vector and nothing else, so quality is measured on what
    the vector is for: whether two sentences a human called similar land close
    together.

    Absolute per model, like every other task here -- a score, not a comparison.
    That is what lets two runs be compared afterwards through the same retention
    machinery rather than needing both models loaded at once.
    """

    kind: Literal["embedding"] = "embedding"
    name: str
    dataset: DatasetSpec
    batch_size: int = Field(default=16, ge=1)
    max_length: int | None = Field(
        default=None, description="Encoder window; defaults to the model's own"
    )
    pooling: Literal["mean", "cls"] | None = Field(
        default=None,
        description=(
            "How token vectors become one sentence vector. Left unset, it is "
            "read from the model's own sentence-transformers config, because a "
            "screen that pools differently from the server is not measuring the "
            "thing that gets deployed. Mean averages unmasked positions; cls "
            "takes the first token, which is what BGE and vLLM both use."
        ),
    )
    text_a_column: str = "sentence1"
    text_b_column: str = "sentence2"
    score_column: str = "label"


class RetrievalTask(StrictModel):
    """nDCG@k over a corpus, which is what an embedding model is usually for.

    Sentence similarity is the cheap screen and a weak proxy: a model can hold
    its STS correlation and still return different documents, and returning
    documents is the job. This measures the job.

    Three datasets rather than one, because that is the shape of the task: a
    corpus to search, queries to search it with, and human judgements of which
    documents answer which query.
    """

    kind: Literal["retrieval"] = "retrieval"
    name: str
    dataset: DatasetSpec
    """The corpus. Named ``dataset`` like every other task so the pre-flight
    dataset check and the fingerprinting work on it unchanged."""

    queries: DatasetSpec
    qrels: DatasetSpec

    top_k: int = Field(default=10, ge=1, description="Cut-off for nDCG@k")
    batch_size: int = Field(default=32, ge=1)
    max_length: int | None = None
    pooling: Literal["mean", "cls"] | None = None

    doc_id_column: str = "_id"
    doc_text_column: str = "text"
    doc_title_column: str | None = "title"
    query_id_column: str = "_id"
    query_text_column: str = "text"
    qrel_query_column: str = "query-id"
    qrel_doc_column: str = "corpus-id"
    qrel_score_column: str = "score"


TaskSpec = Annotated[
    PerplexityTask | MultipleChoiceTask | EmbeddingTask | RetrievalTask,
    Field(discriminator="kind"),
]


class BaselineInferenceSpec(StrictModel):
    """A greedy generation smoke test.

    Timings recorded here come from Transformers, not a serving runtime. They
    exist to prove the model actually runs and to catch gross regressions, never
    to make deployment performance claims. Phase 2 measures those inside vLLM
    instead.
    """

    enabled: bool = True
    prompts: list[str] = Field(
        default_factory=lambda: [
            "The capital of France is",
            "Explain in one sentence why the sky appears blue:",
            "def fibonacci(n):",
        ]
    )
    max_new_tokens: int = Field(default=32, ge=1)
    warmup_runs: int = Field(default=1, ge=0)
    repeats: int = Field(default=1, ge=1)
    capture_output: bool = Field(
        default=True, description="Store generated text so drift is human-inspectable"
    )


class DeploymentSpec(StrictModel):
    """How to benchmark a running serving endpoint.

    AutoDistiller measures a server it did not start, so the endpoint is part of
    the configuration rather than something the tool provisions. See
    :mod:`autodistiller.serving.backends` for why.
    """

    backend: str = "vllm"
    endpoint: str = "http://localhost:8000"
    served_model: str | None = Field(
        default=None, description="Defaults to whatever the endpoint reports serving"
    )
    prompt_tokens: int = Field(default=256, ge=1)
    prompt_file: Path | None = Field(
        default=None,
        description=(
            "Benchmark with this text instead of generated filler. Matters for "
            "speculative decoding, whose speedup depends on the prompt's content."
        ),
    )
    max_tokens: int = Field(default=128, ge=1)
    concurrency_levels: list[int] = Field(default_factory=lambda: [1, 4, 16])
    requests_per_level: int | None = Field(default=None, ge=1)
    warmup_requests: int = Field(default=2, ge=0)
    use_chat: bool = Field(
        default=False,
        description="Use /v1/chat/completions. Defaults to /v1/completions, which "
        "avoids chat-template differences between backends.",
    )
    device_index: int = 0

    @field_validator("concurrency_levels")
    @classmethod
    def _positive_levels(cls, levels: list[int]) -> list[int]:
        if not levels:
            raise ValueError("need at least one concurrency level")
        if any(level < 1 for level in levels):
            raise ValueError("concurrency levels must be >= 1")
        return levels


class CompressionSpec(StrictModel):
    """Which compression to apply, and with what calibration data.

    Runs in an isolated environment: llmcompressor caps transformers below the
    version AutoDistiller uses, and downgrading it in the main environment
    would change the stack every recorded baseline was measured against.
    """

    method: str = Field(description="See `autodistiller methods`")
    backend: str | None = Field(
        default=None,
        description=(
            "Compression toolchain. Defaults to whichever one produces the chosen "
            "method: llmcompressor for compressed-tensors, llama.cpp for GGUF."
        ),
    )
    calibration: DatasetSpec | None = Field(
        default=None, description="Required for calibrated methods (GPTQ, AWQ, INT8 activations)"
    )
    num_calibration_samples: int = Field(default=128, ge=1)
    max_seq_length: int = Field(default=2048, ge=1)
    ignore: list[str] = Field(
        default_factory=lambda: ["lm_head"],
        description="Modules left uncompressed. lm_head is quantization-sensitive.",
    )
    output_dir: Path | None = Field(
        default=None, description="Defaults to a directory named after the method"
    )
    python_executable: str | None = Field(
        default=None, description="Reuse a prepared interpreter instead of `uv run --with`"
    )
    llama_cpp_dir: str | None = Field(
        default=None,
        description="llama.cpp checkout, for GGUF methods. Also read from LLAMA_CPP_DIR.",
    )
    llama_cpp_wrapper: str | None = Field(
        default=None,
        description=(
            "Command template to run the llama.cpp toolchain somewhere else, e.g. "
            "inside WSL. Formatted with {command}. Needed on Windows, where the "
            "binaries are Linux executables."
        ),
    )


class RunConfig(StrictModel):
    """The full, hashable description of an evaluation run."""

    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    model: ModelSpec
    tasks: list[TaskSpec] = Field(default_factory=list)
    baseline_inference: BaselineInferenceSpec = Field(default_factory=BaselineInferenceSpec)
    deployment: DeploymentSpec | None = None
    compression: CompressionSpec | None = None
    seed: int = 1234
    label: str | None = Field(default=None, description="Human label; excluded from the hash")
    output_dir: Path = Field(default=Path("runs"), description="Excluded from the hash")

    # Fields that describe bookkeeping rather than computation. Changing these
    # must not invalidate a cached result.
    _HASH_EXCLUDED = ("label", "output_dir")

    @field_validator("tasks")
    @classmethod
    def _unique_task_names(cls, tasks: list[TaskSpec]) -> list[TaskSpec]:
        seen: set[str] = set()
        for task in tasks:
            if task.name in seen:
                raise ValueError(f"duplicate task name: {task.name!r}")
            seen.add(task.name)
        return tasks

    @property
    def fingerprint(self) -> str:
        payload = self.model_dump(mode="json", exclude=set(self._HASH_EXCLUDED))
        return hash_obj(payload)

    @property
    def evaluation_fingerprint(self) -> str:
        """Identity of what an evaluation actually measures.

        The two cache keys are only separable if changing one cannot invalidate
        the other, and a perplexity number does not depend on the concurrency
        sweep, the endpoint or the request shape. Keying an evaluation on the
        full config makes a deployment tweak discard measurements it did not
        touch, which is the thing :mod:`autodistiller.cache` exists to prevent.

        The section is neutralized rather than dropped, so the digest still
        matches every key already on disk: those were all written from configs
        that carry no deployment section, and dropping the field outright would
        change their hash for no reason.
        """
        payload = self.model_dump(mode="json", exclude=set(self._HASH_EXCLUDED))
        payload["deployment"] = None
        return hash_obj(payload)

    # Serialization
    def to_yaml(self) -> str:
        return yaml.safe_dump(self.model_dump(mode="json"), sort_keys=False, allow_unicode=True)

    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_yaml(), encoding="utf-8")
        return path

    @classmethod
    def from_yaml(cls, path: Path | str) -> RunConfig:
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError(f"{path}: expected a YAML mapping at the top level")
        return cls.model_validate(raw)
