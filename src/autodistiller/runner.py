"""Evaluation orchestration.

One entry point, :func:`run_evaluation`, takes a :class:`RunConfig` and returns a
:class:`RunRecord`. Everything expensive happens behind it and everything
measured is captured with the provenance needed to compare it later.

A failing task does not abort the run: a candidate that breaks on one dataset
but holds on three others is exactly the kind of result the optimizer needs to
see, so failures are recorded per-task and the run continues.
"""

from __future__ import annotations

import logging
import time
import traceback
from collections.abc import Callable

from .cache import experiment_key
from .config import (
    EmbeddingTask,
    ImageClassificationTask,
    MultipleChoiceTask,
    PerplexityTask,
    RetrievalTask,
    RunConfig,
    TaskSpec,
)
from .determinism import seed_everything
from .evaluation.baseline_inference import run_baseline_inference
from .evaluation.datasets import (
    check_dataset_available,
    load_image_classification,
    load_multiple_choice,
    load_sentence_pairs,
    load_text_corpus,
)
from .evaluation.embedding import evaluate_embedding, evaluate_retrieval
from .evaluation.image_classification import evaluate_image_classification
from .evaluation.multiple_choice import evaluate_multiple_choice
from .evaluation.perplexity import evaluate_perplexity
from .evaluation.preprocessors import get_preprocessor
from .metadata.environment import collect_environment
from .metadata.hardware import detect_hardware
from .models.loader import loaded_model
from .results import RunRecord, TaskResult
from .store import RunStore

logger = logging.getLogger(__name__)

StatusFn = Callable[[str], None]
ProgressFn = Callable[[int, int], None]


class RunObserver:
    """Hooks the CLI uses to render progress. The default does nothing."""

    def stage(self, message: str) -> None:  # pragma: no cover - UI glue
        logger.info("%s", message)

    def task_started(self, task: TaskSpec) -> None:  # pragma: no cover - UI glue
        logger.info("task %s (%s)", task.name, task.kind)

    def task_progress(self, done: int, total: int) -> None:  # pragma: no cover - UI glue
        pass

    def task_finished(self, result: TaskResult) -> None:  # pragma: no cover - UI glue
        logger.info("task %s finished in %.1fs", result.name, result.duration_s)


def _failed_task(task: TaskSpec, exc: Exception, duration: float) -> TaskResult:
    logger.warning("task %s failed: %s", task.name, exc)
    return TaskResult(
        name=task.name,
        kind=task.kind,
        metrics=[],
        duration_s=duration,
        details={
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(limit=5),
        },
    )


def _run_task(handle, task: TaskSpec, observer: RunObserver) -> TaskResult:
    if isinstance(task, PerplexityTask):
        corpus = load_text_corpus(task.dataset)
        return evaluate_perplexity(handle, task, corpus=corpus, progress=observer.task_progress)

    if isinstance(task, MultipleChoiceTask):
        dataset = load_multiple_choice(
            task.dataset,
            context_column=task.context_column,
            choices_column=task.choices_column,
            answer_column=task.answer_column,
            transform=get_preprocessor(task.preprocessor),
        )
        return evaluate_multiple_choice(
            handle, task, dataset=dataset, progress=observer.task_progress
        )

    if isinstance(task, EmbeddingTask):
        return evaluate_embedding(
            handle,
            task,
            dataset=load_sentence_pairs(
                task.dataset,
                text_a_column=task.text_a_column,
                text_b_column=task.text_b_column,
                score_column=task.score_column,
            ),
            progress=observer.task_progress,
        )

    if isinstance(task, RetrievalTask):
        return evaluate_retrieval(handle, task, progress=observer.task_progress)

    if isinstance(task, ImageClassificationTask):
        return evaluate_image_classification(
            handle,
            task,
            dataset=load_image_classification(
                task.dataset,
                image_column=task.image_column,
                label_column=task.label_column,
            ),
            progress=observer.task_progress,
        )

    raise TypeError(f"unsupported task kind: {type(task).__name__}")


def preflight(config: RunConfig) -> list[str]:
    """Report every task whose dataset cannot possibly load.

    Loading a model is the expensive part of a run. Checking the cheap things
    first means a mistyped dataset costs a second instead of a download, and
    reporting *all* of them at once beats fixing typos one run at a time.
    """
    problems: list[str] = []
    for task in config.tasks:
        try:
            check_dataset_available(task.dataset)
        except (ValueError, FileNotFoundError) as exc:
            problems.append(f"{task.name}: {exc}")
    return problems


def run_evaluation(
    config: RunConfig,
    *,
    observer: RunObserver | None = None,
    store: RunStore | None = None,
    save: bool = True,
    reuse: bool = True,
) -> RunRecord:
    """Execute a full evaluation and return (and optionally persist) the record.

    With ``reuse``, an experiment already measured on this hardware and this
    software stack is returned instead of being run again. The check lives here
    rather than in the callers so that everything routed through this function
    gets it: the ``evaluate`` command, and the optimizer's per-candidate quality
    screening.
    """
    observer = observer or RunObserver()
    store = store or RunStore(config.output_dir)
    started = time.perf_counter()

    # Before the seed and before preflight: both are cheap, but neither is free,
    # and a cache hit should not need a dataset to be reachable at all.
    hardware = detect_hardware()
    environment = collect_environment()
    key = experiment_key(config.evaluation_fingerprint, hardware, environment)

    if reuse and (cached := store.find_experiment(key)) is not None:
        observer.stage(f"Reusing {cached.run_id}: identical config, hardware and stack")
        return cached

    if problems := preflight(config):
        listed = "\n  - ".join(problems)
        raise ValueError(f"cannot start the run; fix these task datasets first:\n  - {listed}")

    seed_everything(config.seed)
    observer.stage(f"Seeded run with {config.seed}")
    observer.stage(f"Hardware: {hardware.describe()}")

    run_id = store.new_run_id(config)
    tasks: list[TaskResult] = []
    baseline = None
    status: str = "ok"
    error: str | None = None

    observer.stage(f"Loading {config.model.id}")
    with loaded_model(config.model) as handle:
        observer.stage(
            f"Loaded {handle.info.architecture} "
            f"({(handle.info.n_parameters or 0) / 1e9:.2f}B params, "
            f"{handle.info.dtype}, ctx={handle.info.context_length})"
        )

        if config.baseline_inference.enabled:
            observer.stage("Baseline inference (transformers runtime, not a deployment claim)")
            try:
                baseline = run_baseline_inference(handle, config.baseline_inference)
            except Exception as exc:
                logger.warning("baseline inference failed: %s", exc)
                status, error = "failed", f"baseline_inference: {type(exc).__name__}: {exc}"

        for task in config.tasks:
            observer.task_started(task)
            task_started = time.perf_counter()
            try:
                result = _run_task(handle, task, observer)
            except Exception as exc:
                result = _failed_task(task, exc, time.perf_counter() - task_started)
                status = "failed"
                error = error or f"{task.name}: {type(exc).__name__}: {exc}"
            tasks.append(result)
            observer.task_finished(result)

        model_info = handle.info

    record = RunRecord(
        run_id=run_id,
        status="ok" if status == "ok" else "failed",
        error=error,
        config=config,
        config_fingerprint=config.fingerprint,
        experiment_key=key,
        candidate_id=config.label,
        model=model_info,
        hardware=hardware,
        environment=environment,
        tasks=tasks,
        baseline_inference=baseline,
        total_duration_s=time.perf_counter() - started,
    )

    if save:
        directory = store.save(record)
        observer.stage(f"Saved run to {directory}")

    return record


__all__ = ["RunObserver", "preflight", "run_evaluation"]
