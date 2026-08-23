"""Standalone llmcompressor driver.

Runs in its own environment, not AutoDistiller's, and therefore imports nothing
from AutoDistiller. It reads a JSON job on stdin and writes a JSON result to
stdout; everything else the backend prints goes to stderr so the result stays
machine-readable.

The isolation is not fussiness. llmcompressor caps ``transformers<=5.14.1``
while AutoDistiller runs 5.15.x, and silently downgrading transformers in the
main environment would change the stack that every recorded baseline was
measured against.

Invoked as::

    uv run --with llmcompressor python _runner.py < job.json
"""

from __future__ import annotations

import json
import sys
import time
import traceback
from pathlib import Path

ALGORITHMS = ("rtn", "gptq", "awq")


def _emit(payload: dict) -> None:
    """Write the result to stdout and nothing else ever does."""
    json.dump(payload, sys.stdout)
    sys.stdout.flush()


def _versions() -> dict[str, str]:
    from importlib import metadata

    found = {}
    for package in ("llmcompressor", "compressed-tensors", "transformers", "torch"):
        try:
            found[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            continue
    return found


def _build_modifier(job: dict):
    """Translate an AutoDistiller method into an llmcompressor modifier."""
    algorithm = job["algorithm"]
    scheme = job["scheme"]
    ignore = job.get("ignore") or ["lm_head"]

    if algorithm == "gptq":
        from llmcompressor.modifiers.quantization import GPTQModifier

        return GPTQModifier(targets="Linear", scheme=scheme, ignore=ignore)

    if algorithm == "awq":
        from llmcompressor.modifiers.awq import AWQModifier

        return AWQModifier(targets="Linear", scheme=scheme, ignore=ignore)

    from llmcompressor.modifiers.quantization import QuantizationModifier

    return QuantizationModifier(targets="Linear", scheme=scheme, ignore=ignore)


def _build_calibration_dataset(job: dict, tokenizer):
    """Tokenize the caller's calibration texts.

    AutoDistiller supplies the text rather than naming a dataset for
    llmcompressor to fetch, so the calibration data is fingerprinted upstream
    and the same bytes are reproducible later.
    """
    texts = job.get("calibration_texts") or []
    if not texts:
        return None

    from datasets import Dataset

    max_length = job.get("max_seq_length", 2048)
    encoded = tokenizer(
        texts,
        truncation=True,
        max_length=max_length,
        padding=False,
        add_special_tokens=True,
    )
    return Dataset.from_dict(
        {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
        }
    )


def _directory_size(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def run(job: dict) -> dict:
    started = time.perf_counter()

    if job["algorithm"] not in ALGORITHMS:
        raise ValueError(f"unknown algorithm {job['algorithm']!r}; expected one of {ALGORITHMS}")

    from llmcompressor import oneshot
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_id = job["model_id"]
    output_dir = job["output_dir"]
    common = {"trust_remote_code": job.get("trust_remote_code", False)}

    tokenizer = AutoTokenizer.from_pretrained(model_id, **common)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=job.get("dtype", "auto"), device_map=job.get("device_map", "auto"), **common
    )

    modifier = _build_modifier(job)
    dataset = _build_calibration_dataset(job, tokenizer)

    oneshot_kwargs: dict = {
        "model": model,
        "recipe": modifier,
        "output_dir": output_dir,
    }
    if dataset is not None:
        oneshot_kwargs.update(
            {
                "dataset": dataset,
                "num_calibration_samples": min(
                    job.get("num_calibration_samples", 128), len(dataset)
                ),
                "max_seq_length": job.get("max_seq_length", 2048),
            }
        )

    oneshot(**oneshot_kwargs)
    tokenizer.save_pretrained(output_dir)

    return {
        "ok": True,
        "output_dir": output_dir,
        "duration_s": time.perf_counter() - started,
        "artifact_bytes": _directory_size(Path(output_dir)),
        "versions": _versions(),
        "n_calibration_samples": len(dataset) if dataset is not None else 0,
    }


def main() -> int:
    try:
        job = json.load(sys.stdin)
    except json.JSONDecodeError as exc:
        _emit({"ok": False, "error": f"invalid job JSON: {exc}"})
        return 2

    try:
        _emit(run(job))
        return 0
    except Exception as exc:
        _emit(
            {
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(limit=8),
                "versions": _versions(),
            }
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())
