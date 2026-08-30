"""Depth pruning.

The checks that matter are the two that fail silently: picking the wrong blocks,
and writing a checkpoint that will not reload. Both are exercised against a real
(tiny) Llama rather than a mock, because both are properties of what
Transformers actually saves.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from autodistiller.compression.prune import (
    PruneJob,
    block_influence,
    choose_layers,
    layer_list,
    run_prune,
    slice_per_layer_config,
)
from autodistiller.config import ModelSpec

CALIBRATION = [
    "The quick brown fox jumps over the lazy dog and keeps running.",
    "Perplexity measures how well a language model predicts a sample of text.",
    "Quantization reduces the memory a neural network needs to run.",
]


@pytest.fixture(scope="module")
def deep_model_dir(tmp_path_factory: pytest.TempPathFactory, tiny_model_dir: Path) -> Path:
    """The session tiny model, but deep enough for dropping to be interesting."""
    from transformers import AutoConfig, LlamaForCausalLM

    directory = tmp_path_factory.mktemp("deep-model")
    for name in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"):
        source = tiny_model_dir / name
        if source.is_file():
            shutil.copy(source, directory / name)

    config = AutoConfig.from_pretrained(tiny_model_dir)
    config.num_hidden_layers = 6
    LlamaForCausalLM(config).save_pretrained(directory)
    return directory


def test_choose_layers_drops_the_least_influential():
    influence = [0.9, 0.1, 0.8, 0.2, 0.7]

    dropped, keep = choose_layers(influence, 2)

    assert dropped == [1, 3]
    assert keep == [0, 2, 4]


def test_choose_layers_never_drops_the_last_block():
    """Its output is what the final norm and the head were fitted against.

    A low score there is not an invitation: removing it is a much bigger change
    than the number suggests, so it is not a candidate at any drop count.
    """
    influence = [0.9, 0.8, 0.7, 0.001]

    dropped, keep = choose_layers(influence, 3)

    assert 3 not in dropped
    assert keep == [3]

    with pytest.raises(ValueError, match="last block is never dropped"):
        choose_layers(influence, 4)


def test_block_influence_scores_every_layer(deep_model_dir: Path):
    from autodistiller.models.loader import loaded_model

    with loaded_model(ModelSpec(id=str(deep_model_dir), device="cpu")) as handle:
        scores = block_influence(
            handle.model, handle.tokenizer, CALIBRATION, device=handle.device, max_length=64
        )

    assert len(scores) == 6
    # 1 - cosine similarity, so a block cannot move the stream by a negative
    # amount and cannot move it further than antiparallel.
    assert all(0.0 <= score <= 2.0 for score in scores)


def test_layer_list_finds_the_blocks(deep_model_dir: Path):
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(deep_model_dir)

    name, blocks = layer_list(model)

    assert name.endswith("layers")
    assert len(blocks) == 6


def test_slice_per_layer_config_cuts_lists_to_the_survivors():
    """Qwen3 and Gemma describe attention per layer.

    A list left at the original depth reloads as a config that disagrees with
    the weights, and the failure surfaces long after pruning reported success.
    """

    class Config:
        def __init__(self):
            self.layer_types = ["full", "sliding", "full", "sliding"]
            self.num_hidden_layers = 4
            self.hidden_size = 32

    config = Config()

    touched = slice_per_layer_config(config, [0, 2], 4)

    assert touched == ["layer_types"]
    assert config.layer_types == ["full", "full"]
    assert config.hidden_size == 32


def test_pruned_model_reloads_and_runs(deep_model_dir: Path, tmp_path: Path):
    """The end that matters: fewer layers, and still a servable checkpoint."""
    from transformers import AutoModelForCausalLM

    job = PruneJob(
        model=ModelSpec(id=str(deep_model_dir), device="cpu"),
        n_drop=2,
        calibration_texts=CALIBRATION,
        max_length=64,
        output_dir=tmp_path / "pruned",
    )

    artifact = run_prune(job)

    assert artifact.recipe.method == "prune2"
    assert artifact.artifact_bytes and artifact.artifact_bytes > 0

    reloaded = AutoModelForCausalLM.from_pretrained(job.output_dir)
    assert reloaded.config.num_hidden_layers == 4
    assert len(layer_list(reloaded)[1]) == 4

    import torch

    with torch.no_grad():
        logits = reloaded(torch.tensor([[1, 2, 3]])).logits
    assert logits.shape[:2] == (1, 3)


def test_artifact_key_tracks_the_calibration_data(deep_model_dir: Path):
    """Different text scores the blocks differently, so it is different weights."""
    model = ModelSpec(id=str(deep_model_dir))
    one = PruneJob(model=model, n_drop=2, calibration_texts=CALIBRATION)
    two = PruneJob(model=model, n_drop=2, calibration_texts=CALIBRATION[:2])
    same = PruneJob(model=model, n_drop=2, calibration_texts=list(CALIBRATION))

    assert one.artifact_key != two.artifact_key
    assert one.artifact_key == same.artifact_key
    assert (
        PruneJob(model=model, n_drop=3, calibration_texts=CALIBRATION).artifact_key
        != one.artifact_key
    )
