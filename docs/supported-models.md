# Supported models

What AutoDistiller runs, what it refuses, and what it has actually been measured
against. The [README](../README.md) summarises this; here it is in full, with the
distinction that matters kept visible:

| | |
|---|---|
| **Measured** | run end to end on this hardware, numbers recorded |
| **Checked** | dimensions verified against the real checkpoint, not run |
| **Refused** | recognised and declined, with a reason |

"Checked" is not a weaker claim about whether it works — it is a claim about what
has been *proven*. A refusal is a supported outcome: the tool telling you it
cannot describe something beats a confident wrong number.

## Ask the tool

Nothing below is a hard-coded allow-list. Support falls out of three rules —
which Auto class loads the architecture, whether the memory arithmetic describes
it, and whether a runtime can serve it — so the fastest answer for a model not
listed here is to ask:

```bash
uv run autodistiller candidates --model <your-model>
```

```bash
uv run autodistiller methods --model <your-model>
```

The first prints the dimensions it read and every configuration it would try.
The second prints which compression methods apply, each with its reason.

---

## Decoder-only language models

The main path. Any Hugging Face causal LM: dimensions come from `config.json` and
loading goes through `AutoModelForCausalLM`, with no per-architecture handling.

**Measured** on an 8 GiB card:

| Model | Family | Params |
|---|---|---|
| `openai-community/gpt2` | GPT-2 | 0.15B |
| `Qwen/Qwen3-0.6B` | Qwen 3 | 0.60B |
| `TinyLlama/TinyLlama-1.1B-Chat-v1.0` | Llama | 1.10B |
| `meta-llama/Llama-3.2-1B-Instruct` | Llama 3 | 1.24B |
| `deepseek-ai/deepseek-coder-1.3b-instruct` | DeepSeek | 1.35B |
| `stabilityai/stablelm-2-1_6b` | StableLM | 1.64B |
| `HuggingFaceTB/SmolLM2-1.7B-Instruct` | Llama | 1.71B |
| `EleutherAI/pythia-1.4b` | GPT-NeoX | 1.82B |
| `ibm-granite/granite-3.1-2b-instruct` | Granite | 2.53B |
| `google/gemma-2-2b-it` | Gemma 2 | 2.61B |
| `meta-llama/Llama-3.2-3B-Instruct` | Llama 3 | 3.21B |
| `microsoft/Phi-3-mini-4k-instruct` | Phi-3 | 3.82B |
| `google/gemma-3-4b-it` | Gemma 3 (VLM) | 3.88B |
| `Qwen/Qwen3-4B` | Qwen 3 | 4.02B |
| `mistralai/Mistral-7B-Instruct-v0.3` | Mistral | 7.25B |
| `allenai/OLMo-2-1124-7B` | OLMo 2 | 7.30B |
| `meta-llama/Llama-3.1-8B-Instruct` | Llama 3 | 8.03B |
| `tiiuae/falcon-7b-instruct` | Falcon | 10.87B |

**Vision-language models** are read by their language tower. Gemma 3 keeps the
decoder in a nested `text_config`, and that is the half that gets quantized and
whose KV cache dominates memory. The vision encoder is not counted, so a VLM's
memory estimate is of its language half.

**Mixture-of-experts** models are counted by their *resident* parameters, not
their active ones: Qwen3-30B-A3B is 3B active and 30B on the card, and only the
second number decides whether it fits.

All 11 compression methods apply — 6 through vLLM, 5 through llama.cpp.

**Gated models** (Llama, Gemma) need `hf auth login` plus access granted on the
model page.

---

## Encoder / embedding models

BERT-family encoders go through the same spine: evaluate, compress, screen,
benchmark, rank, export. Loaded with `AutoModel`, because what an embedding task
needs is the hidden states underneath a head rather than a head.

**Measured** end to end, including against a real vLLM pooling server:

| Model | Params |
|---|---|
| `BAAI/bge-small-en-v1.5` | 33M |
| `google-bert/bert-base-uncased` | 109M |

**Checked** — recognised as encoders, and the parameter estimate compared
against the real checkpoint:

| Model | Architecture | Estimate vs actual |
|---|---|---|
| `BAAI/bge-base-en-v1.5` | `BertModel` | 99.35% |
| `intfloat/e5-base-v2` | `BertModel` | 99.35% |
| `thenlper/gte-base` | `BertModel` | 99.35% |
| `sentence-transformers/all-MiniLM-L6-v2` | `BertModel` | 99.21% |
| `FacebookAI/roberta-base` | `RobertaForMaskedLM` | 99.39% |

That is the original BERT block: two-matrix feed-forward, learned absolute
position table. **A modernised encoder is recognised but mis-sized** — see the
gap below before pointing the memory screen at one.

Five of the eleven methods apply: `int8`, `int8-weight-only`, `int4-gptq`, `fp8`,
`fp8-static`. Tasks are `stsb` (similarity) and `scifact` (retrieval, nDCG@10).
The search runs over sequence length × batch size × depth.

---

## Vision transformers

Image classifiers load through `AutoModelForImageClassification` — the head is
the point, since the label it predicts is what gets scored — and carry an image
processor where a text model carries a tokenizer.

**Measured** end to end:

| Model | Architecture | Params | Top-1 |
|---|---|---|---|
| `google/vit-base-patch16-224` | `ViTForImageClassification` | 86M | 78.03% |
| `facebook/deit-base-patch16-224` | `ViTForImageClassification` | 86M | 78.37% |
| `facebook/dinov2-small-imagenet1k-1-layer` | `Dinov2ForImageClassification` | 22M | 80.08% |

Top-1 on the `imagenet` preset, whose images are re-encoded and so score about
two points below a published figure — see the README for why, and use
`imagenet-original` for a number comparable with a paper's.

**Checked** — parameter estimates against the real checkpoints:

| Model | Architecture | Estimate vs actual |
|---|---|---|
| `google/vit-base-patch16-224` | `ViTForImageClassification` | 99.86% |
| `WinKawaks/vit-small-patch16-224` | `ViTForImageClassification` | 99.72% |
| `microsoft/beit-base-patch16-224` | `BeitForImageClassification` | 99.37% |
| `facebook/dinov2-small-imagenet1k-1-layer` | `Dinov2ForImageClassification` | 98.00% |
| `facebook/data2vec-vision-base-ft1k` | `Data2VecVisionForImageClassification` | recognised |

Two of the eleven methods apply: `int8-weight-only` and `fp8`, the two that need
no calibration pass. Tasks are `imagenet` and `imagenet-original`. **No serving
backend runs these**, so the pipeline stops after quality and size — there are no
throughput or latency numbers for a vision model.

---

## Refused, and why

Each of these is recognised and declined with a message naming the reason. None
of them is a crash, and none produces a number.

| What | Why |
|---|---|
| **Staged and convolutional image classifiers** — Swin, ConvNeXt, ResNet, MobileViT, EfficientNet | Their width changes every stage, so one `hidden_size` and one block count do not describe them. The message names the missing config fields. |
| **Encoder-decoder models** — T5, BART, Whisper, DETR | The encoder half has no KV cache and is not counted, so the estimate would be wrong rather than missing. |
| **Vision models that are not classifiers** — CLIP, SigLIP, DINOv2 backbones, SegFormer | No head this tool knows how to score. Refused by the shape estimator, and `evaluate` fails at the Auto class with `Unrecognized configuration class` before any weights are fetched. |
| **`int4-awq` on an encoder or a vision model** | AWQ smooths activations through per-architecture mappings and llmcompressor registers 31, every one a decoder. On anything else it matches nothing and divides by zero. |
| **Calibrated methods on a vision model** — `int8`, `int4-gptq`, `int4-awq`, `fp8-static` | Calibration means pushing sample text through a tokenizer, and a vision tower has neither. |
| **Depth pruning on a vision model** | Block influence is scored against calibration text, for the same reason. |
| **GGUF on an encoder** | Untried, so not offered. This is the weakest refusal here — "not tested", not "impossible". |
| **Non-NVIDIA accelerators** | The capability rules are keyed on CUDA compute capability. |
| **timm-format checkpoints** (`timm/…`) | Their `config.json` is timm's own, not a Hugging Face model config. Transformers refuses it with a clear `ImportError` unless `timm` is installed; AutoDistiller has not been tested against the wrapper that appears when it is. |

---

## Known gaps

Two places where the tool is not wrong so much as imprecise, both found by
running it and neither fixed yet.

**Modernised encoders are under-estimated.** The encoder arithmetic describes
the original BERT block, and several newer embedding models have replaced its
feed-forward with a gated one — three matrices where the estimate counts two —
or dropped the position table for rotary embeddings. They are recognised as
encoders, so the kind is right and the shape is not:

| Model | Estimate vs actual |
|---|---|
| `answerdotai/ModernBERT-base` | 90.75% |
| `jinaai/jina-embeddings-v2-base-en` | 83.48% |
| `nomic-ai/nomic-embed-text-v1.5` | 80.42% |

Under, which is the dangerous direction: a screen that under-estimates admits a
candidate that then runs out of memory at serve time. Evaluation and compression
are unaffected — those load the real weights — so this is a caution about
`candidates` and `optimize`, not about the numbers those produce.

**DeBERTa-v3 is read as a decoder.** `microsoft/deberta-v3-base` ships a
`config.json` with no `architectures` key, and the rule for that case — written
when every model here was an LLM — is "assume decoder". The result is a
confident wrong shape: 0.21B parameters and a KV cache it does not have. Its
`model_type` says `deberta-v2` and is not consulted. Other encoders are
unaffected, because they all name their architecture.

**A DINOv2 classifier reports the wrong sequence length.** Its config advertises
`image_size: 518` (1370 patch tokens) because that is what its position table was
trained at, while its image processor crops to 224 (257 tokens) — so that is what
actually runs. The recorded evaluation context says 1370, and the memory screen
models five times the activations it needs. The estimate errs high, so it would
reject a configuration that fits rather than accept one that does not, and no
runtime serves vision anyway. The accuracy numbers are unaffected. ViT, DeiT and
BEiT are unaffected: their config and their processor agree.
