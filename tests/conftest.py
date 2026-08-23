"""Shared fixtures.

The important one is ``tiny_model``: a real (randomly initialised) Llama and a
real trained tokenizer, saved to disk. It exercises the genuine load ->
evaluate -> record path in about a second and needs no network, so the
end-to-end behaviour is covered by ordinary unit tests rather than only by
manual runs against a downloaded model.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

CORPUS_LINES = [
    "The quick brown fox jumps over the lazy dog.",
    "Machine learning models are evaluated on held-out data.",
    "Quantization reduces the memory footprint of a neural network.",
    "Perplexity measures how well a language model predicts a sample.",
    "Deployment backends such as vLLM serve models at scale.",
    "A trustworthy baseline must be reproducible across runs.",
]


@pytest.fixture(scope="session")
def tiny_model_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A minimal but genuine causal LM saved to disk."""
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

    directory = tmp_path_factory.mktemp("tiny-model")

    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.train_from_iterator(
        CORPUS_LINES * 20,
        trainers.BpeTrainer(
            vocab_size=300,
            special_tokens=["<unk>", "<pad>", "<eos>"],
            show_progress=False,
        ),
    )

    fast_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="<unk>",
        pad_token="<pad>",
        eos_token="<eos>",
        bos_token="<eos>",
    )
    fast_tokenizer.save_pretrained(directory)

    config = LlamaConfig(
        vocab_size=len(fast_tokenizer),
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=128,
        bos_token_id=fast_tokenizer.bos_token_id,
        eos_token_id=fast_tokenizer.eos_token_id,
        pad_token_id=fast_tokenizer.pad_token_id,
    )
    LlamaForCausalLM(config).save_pretrained(directory)

    return directory


@pytest.fixture(scope="session")
def text_corpus_file(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("data") / "corpus.txt"
    path.write_text("\n".join(CORPUS_LINES * 4), encoding="utf-8")
    return path


@pytest.fixture(scope="session")
def jsonl_corpus_file(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("data") / "corpus.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for line in CORPUS_LINES:
            handle.write(json.dumps({"text": line}) + "\n")
    return path


@pytest.fixture(scope="session")
def mc_dataset_file(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("data") / "evals.jsonl"
    rows = [
        {
            "id": "q1",
            "context": "Question: What jumps over the lazy dog?\nAnswer:",
            "choices": [" The quick brown fox.", " A deployment backend."],
            "answer_index": 0,
        },
        {
            "id": "q2",
            "context": "Question: What reduces memory footprint?\nAnswer:",
            "choices": [" Perplexity.", " Quantization."],
            "answer_index": 1,
        },
        {
            "id": "q3",
            "context": "Question: What serves models at scale?\nAnswer:",
            "choices": [" vLLM.", " The lazy dog."],
            "answer_index": 0,
        },
    ]
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    return path
