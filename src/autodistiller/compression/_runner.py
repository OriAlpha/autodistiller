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

DECODER_SUFFIXES = ("ForCausalLM", "ForConditionalGeneration")
"""Deliberately a copy of ``autodistiller.architecture``.

This file runs in its own environment and imports nothing from AutoDistiller, so
the six lines below are duplicated rather than shared. Keeping them in step is
cheaper than the alternative, which is installing AutoDistiller into an
environment that exists to hold a conflicting transformers pin.
"""


def _emit(payload: dict) -> None:
    """Write the result to stdout and nothing else ever does."""
    json.dump(payload, sys.stdout)
    sys.stdout.flush()


def _dtype_kwarg() -> str:
    """Transformers renamed `torch_dtype` to `dtype` in v5.

    The isolated environment resolves its own transformers, which may be either
    major version, so the runner cannot assume the one AutoDistiller uses.
    """
    import transformers

    return "dtype" if int(transformers.__version__.split(".")[0]) >= 5 else "torch_dtype"


def _versions() -> dict[str, str]:
    from importlib import metadata

    found = {}
    for package in ("llmcompressor", "compressed-tensors", "transformers", "torch"):
        try:
            found[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            continue
    return found


def _build_modifier(job: dict, ignore: list[str]):
    """Translate an AutoDistiller method into an llmcompressor modifier."""
    algorithm = job["algorithm"]
    scheme = job["scheme"]

    if algorithm == "gptq":
        from llmcompressor.modifiers.quantization import GPTQModifier

        return GPTQModifier(targets="Linear", scheme=scheme, ignore=ignore)

    if algorithm == "awq":
        from llmcompressor.modifiers.awq import AWQModifier

        return AWQModifier(targets="Linear", scheme=scheme, ignore=ignore)

    from llmcompressor.modifiers.quantization import QuantizationModifier

    return QuantizationModifier(targets="Linear", scheme=scheme, ignore=ignore)


def _resolve_dtype(job: dict, config) -> str:
    """What to load the weights as before quantizing them.

    ``auto`` means 16-bit unless the checkpoint declares something narrower.
    Quantization leaves embeddings and the output head alone, so a 32-bit source
    writes those tensors out at 32 bits -- and every serving runtime then
    downcasts them at load. Measured on bge-small-en-v1.5: 70.7 MB written
    against the 45.1 MB anything would actually hold, an artifact 57% larger
    than the memory estimate for no benefit.

    A config that declares *nothing* counts as 32-bit, because that is what
    Transformers loads when nobody says otherwise. Reading the silence as "leave
    it alone" is what let bert-base-uncased -- which states no dtype at all,
    like every checkpoint of its era -- keep writing float32 after this function
    already existed.

    An explicit dtype is honoured, since asking for one is asking for it.
    """
    requested = job.get("dtype", "auto")
    if requested != "auto":
        return requested
    source = str(getattr(config, "dtype", None) or getattr(config, "torch_dtype", None) or "")
    return "auto" if source.endswith(("float16", "bfloat16")) else "bfloat16"


def _load_model(job: dict):
    """Load the weights with the Auto class this architecture needs.

    An encoder has no causal LM head, and asking for one either refuses or
    attaches a randomly initialised head that then gets calibrated and
    quantized. Returns the model and what to leave alone: ``lm_head`` is a
    decoder's output layer, and naming a module the model does not have is not
    something the backend has to tolerate.
    """
    from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

    model_id = job["model_id"]
    common = {"trust_remote_code": job.get("trust_remote_code", False)}

    config = AutoConfig.from_pretrained(model_id, **common)
    names = [str(n) for n in (getattr(config, "architectures", None) or ())]
    is_decoder = not names or any(name.endswith(DECODER_SUFFIXES) for name in names)

    auto_class = AutoModelForCausalLM if is_decoder else AutoModel
    dtype = _resolve_dtype(job, config)
    model = auto_class.from_pretrained(
        model_id,
        config=config,
        device_map=job.get("device_map", "auto"),
        **{_dtype_kwarg(): dtype},
        **common,
    )

    # The config still names the checkpoint's dtype, and the save follows the
    # config rather than the loaded tensors -- so without this the weights are
    # written back out at the width they were downcast from.
    if dtype != "auto":
        import torch

        resolved = getattr(torch, dtype)
        for field in ("dtype", "torch_dtype"):
            if getattr(model.config, field, None) is not None:
                setattr(model.config, field, resolved)
    ignore = list(job.get("ignore") or []) if is_decoder else []
    return model, ignore


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
    from transformers import AutoTokenizer

    model_id = job["model_id"]
    output_dir = job["output_dir"]
    common = {"trust_remote_code": job.get("trust_remote_code", False)}

    tokenizer = AutoTokenizer.from_pretrained(model_id, **common)
    model, ignore = _load_model(job)

    modifier = _build_modifier(job, ignore)
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
