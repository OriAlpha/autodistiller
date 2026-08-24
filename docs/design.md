# Design notes

Why AutoDistiller works the way it does. The [README](../README.md) covers what it does;
this covers the reasoning, and the failures that shaped it.

---

## Why every run records so much

A run record carries the config, the resolved model commit, an architecture fingerprint, dataset
content fingerprints, library versions, and the hardware it ran on. That is not bookkeeping for its
own sake:

- **Comparability is checkable.** `compare` refuses to score a comparison where the two runs used
  different data, and warns when the hardware or software stack moved.
- **The experiment cache needs it.** Reusing a measurement is only safe if you can prove the inputs
  were identical. The config hash and the hardware and software fingerprints are that proof.
- **It is the long-term differentiator.** The defensible asset is measured knowledge: which
  configurations work on which models, GPUs, backends and software stacks.

### On performance numbers

The baseline inference step reports tokens/sec. It is tagged `runtime: "transformers"` and
`is_deployment_claim: false`, and the CLI says so every time it prints them. Transformers timings
are a smoke test, not serving performance. Deployment numbers get measured inside the deployment
backend — that is Phase 2.

---

---

## The experiment cache

Nothing is measured twice. `evaluate`, `compress` and `optimize` all check first, and reuse an
identical earlier result instead of repeating it:

```bash
uv run autodistiller history
```

```bash
uv run autodistiller history --model Qwen3-0.6B --json
```

An experiment is reusable only when everything that could have moved the number is unchanged: the
config (model, tasks, datasets, seed, compression recipe), the hardware, and the software stack.
Change the GPU or upgrade torch and the cache misses, as it should. Pass `--refresh` to any of the
three commands to measure again anyway.

The stack half of that key is deliberately narrow — `autodistiller`, `torch`, `transformers`,
`tokenizers`, `datasets`, CUDA and the Python minor version. Keying on every installed package is
defensible in theory and useless in practice: a `safetensors` patch bump would throw away every
result without changing any of them.

Three things are cached, in cost order:

| What | Keyed on | Where |
|---|---|---|
| Compressed artifacts | model, method, calibration data, `ignore`, dtype | `artifacts/<model>-<method>-<key>/` |
| Evaluations | config fingerprint + hardware + stack | `runs/<run_id>/record.json` |
| Deployment benchmarks | served weights, backend, request shape, context length, KV dtype | `runs/<run_id>/record.json` |

Artifact directories carry the recipe key because the recipe *is* the identity of the weights.
`Qwen3-0.6B-int4-gptq` alone is not: compress that model and method with two different calibration
sets and you get two genuinely different artifacts, and one path for both means the second silently
replaces the first.

`runs/index.jsonl` holds one row per record — the keys and a summary, no metrics — so a lookup does
not have to parse every run ever done. It is derived state; delete it and it rebuilds, or force it
with `autodistiller history --rebuild`. It is also the shape a shared benchmark database would want:
flat rows carrying a complete key rather than a local file layout.

---

---

## Trade-offs and recommendations

A single winning score cannot be checked. `optimize` therefore also prints the
configurations where you cannot improve one thing without losing another, and names
the options a reader is likely to want:

```
Pareto frontier - Quality retention vs Peak VRAM vs TTFT p50 vs Peak throughput
 Candidate      Quality retention   Peak VRAM   TTFT p50   Peak throughput   Verdict
 baseline                 100.00%    7.00 GiB      110ms         780 tok/s   Pareto-optimal
 fp8                       98.45%    5.00 GiB       60ms        1010 tok/s   Pareto-optimal
 int4-awq                  94.10%    4.00 GiB       40ms        1320 tok/s   Pareto-optimal
 int8                      97.02%    5.00 GiB       70ms         990 tok/s   dominated
```

`int8` is dominated because `fp8` beats it on every axis at once — there is no
reading of the numbers under which you would pick it. The other three are real
trade-offs, and each named recommendation says what choosing it costs:

```
 Option                 Candidate   Wins on                      Frontier   Gives up
 best quality           baseline    quality retention 100.00%    yes        peak vram 7.00 GiB against a best of 4.00 GiB; ...
 fastest (throughput)   int4-awq    peak throughput 1320 tok/s   yes        quality retention 94.10% against a best of 100.00%
```

Two rules keep this honest:

- **A candidate is never ranked on a number nobody measured.** Treating an
  unmeasured throughput as either the best or the worst value would put a
  candidate on the frontier for a reason that is not a measurement. Those are
  listed as "not measured on every axis" instead.
- **An axis never mixes measured and estimated values.** Peak VRAM from a real
  serving run and VRAM predicted by arithmetic are different quantities. When
  nothing was benchmarked the whole axis falls back to estimates and is labelled
  `VRAM (estimated)`; it never compares one against the other.

Early stopping and trade-off analysis pull against each other — the first
qualifying candidate is the only one measured, so there is nothing to compare it
to. Use `--no-stop-early` when you want the frontier, and `--no-pareto` when you
only want the winner.

---

---

## Two backends

`--backend vllm` and `--backend llama.cpp` search different spaces, because they
serve different formats:

| | vLLM | llama.cpp |
|---|---|---|
| Format | compressed-tensors | GGUF |
| Methods | `int8`, `int4-gptq`, `int4-awq`, `fp8`, … | `gguf-q8-0`, `gguf-q6-k`, `gguf-q5-k-m`, `gguf-q4-k-m`, `gguf-q3-k-m` |
| Built by | llmcompressor | `convert_hf_to_gguf.py` + `llama-quantize` |
| Artifact | a directory | a single `.gguf` |
| KV cache types | `auto`, `fp8` | `auto` |
| Default port | 8000 | 8080 |

Picking a method picks the toolchain, so `--method gguf-q4-k-m` routes to
llama.cpp on its own. The search filters itself: a GGUF method is never offered
to vLLM, and llama.cpp is never offered an fp8 KV cache it has no concept of.

```bash
uv run autodistiller optimize --model Qwen/Qwen3-0.6B --backend llama.cpp   --launch-preset native-llamacpp --objective throughput
```

llama.cpp is not pip installable, so point AutoDistiller at a built checkout with
`--llama-cpp` or `LLAMA_CPP_DIR`. When it is missing, the error names which half —
the converter script or the binary — rather than failing inside a subprocess.

### Sizing GGUF honestly

A K-quant is a mix of widths, not its headline number: `gguf-q4-k-m` averages about
4.85 bits per weight. GGUF also quantizes the embeddings, which compressed-tensors
leaves at 16-bit — so at the same nominal width a GGUF artifact is smaller.

Both matter for screening. Published bits-per-weight figures are measured on
7B-class models where embeddings are a rounding error; Qwen3-0.6B carries 26% of
its parameters in a 151936-entry embedding, and llama.cpp keeps those tensors
above the headline type. Estimating from the headline number alone under-reports
by around 15%, and a memory screen that under-estimates produces candidates that
OOM at serve time. Estimates land within 10% of published sizes for this model.

### Quality screening

Quality is screened by loading the GGUF through Transformers, which dequantizes
it: the weights come back carrying the quantization error, which is what a quality
comparison needs. It is not a claim about llama.cpp's inference kernels — those
are measured where they run, in `llama-server`, and reported as the deployment
numbers.

---

---

## Export and deploy

A measured recommendation is only worth something if someone else can deploy it
and rebuild it:

```bash
uv run autodistiller export <run_id>
```

That writes three files beside the weights, so the directory you would serve is
the one that explains itself:

| File | What it carries |
|---|---|
| `autodistiller-manifest.json` | Recipe, calibration fingerprint, metrics, benchmark, hardware, stack |
| `DEPLOY.md` | How to serve it, what it scored, how to rebuild it |
| `autodistiller-config.yaml` | The exact config, which re-hashes to the same experiment |

Export **checks** deployability rather than asserting it, and exits non-zero if
the artifact would not load:

```
config      PASS  config.json present
weights     PASS  1 file(s), 0.70 GiB
tokenizer   PASS  tokenizer.json, tokenizer_config.json
format      PASS  compressed-tensors, which vllm has kernels for

Serve it    vllm serve artifacts/Qwen3-0.6B-fp8-7deef795 --port 8000 --max-model-len 2048
```

Nothing is converted — llmcompressor already writes a Hugging Face directory, so
the artifact *is* the export. What was missing was the provenance tying it to the
measurements that justify it, and a verified claim that a server can load it.
An artifact that benchmarks beautifully and then cannot be served is the failure
these checks exist to catch.

`optimize --export DIR` does the same for the winning configuration, so you never
have to work out which run id it was. Add `--copy-weights` to assemble a bundle
you can move; without it the bundle refers to the weights where they already are.

### GGUF

GGUF artifacts export the same way. They carry their own config and tokenizer, so
the Hugging Face checks do not apply and are not run — asking for a `tokenizer.json`
inside a GGUF directory would report a working artifact as broken.

There is still no conversion *from* a compressed-tensors artifact: GGUF carries
its own quantization schemes and llama.cpp converts from unquantized weights. The
manifest says so and points at the command that builds one from the source model
instead.

---

---

## Metrics

**Perplexity** (`perplexity`, `nll_per_token`, `bits_per_byte`) — strided windows, so every token is
scored exactly once and with as much left context as the window allows. Naive chunking scores the
first token of every chunk with no context at all, which inflates the number. `bits_per_byte` is
tokenizer-independent and stays meaningful when a candidate ships a different tokenizer.

**Multiple choice** (`acc`, `acc_norm`) — each candidate answer is scored by log-probability and the
highest-scoring one wins. No sampling, so results are exactly reproducible. `acc_norm` normalizes by
answer length so longer answers are not penalized for having more tokens.

Both report a standard error, which `compare` uses to distinguish a real regression from noise.

---

---

## Tasks

Run `uv run autodistiller tasks` for the live list.

**Presets** — `wikitext2`, `wikitext103`, `arc_easy`, `arc_challenge`, `hellaswag`, `piqa`

**Your own data**

| Syntax | Meaning |
|---|---|
| `ppl:corpus.txt` | perplexity over a local text file |
| `ppl:corpus.jsonl` | perplexity over a local JSONL corpus (`text` field) |
| `mc:evals.jsonl` | multiple choice over a local JSONL file |

The multiple-choice schema is one JSON object per line:

```json
{"id": "q1", "context": "Question: What is 2+2?\nAnswer:", "choices": [" 3", " 4"], "answer_index": 1}
```

Choices keep their own leading space: they are appended to the context verbatim so tokenization
matches what a real prompt would produce.

For full control, use a config file — see
[`examples/configs/baseline.yaml`](../examples/configs/baseline.yaml):

```bash
uv run autodistiller evaluate --config examples/configs/baseline.yaml
```

---

---

## Reproducibility

Runs are seeded (Python, NumPy, torch, CUDA), cuDNN autotuning is pinned off, and the resolved
config is written next to every result:

```bash
uv run autodistiller evaluate --model Qwen/Qwen3-0.6B --save-config my-baseline.yaml
uv run autodistiller evaluate --config my-baseline.yaml   # same numbers
```

The config hash covers everything that can move a metric and excludes what cannot (`label`,
`output_dir`).

---

---

## On isolation

AutoDistiller imports neither vLLM nor llmcompressor. Serving runtimes and compression backends
have heavy, mutually incompatible pins — llmcompressor caps `transformers<=5.14.1` while
AutoDistiller runs 5.15.x — and quietly downgrading a library would change the stack every recorded
baseline was measured against. So both run in their own environments: vLLM over HTTP, compression
as a subprocess. A useful side effect is that llama.cpp needs no new benchmark client in Phase 9,
because it speaks the same OpenAI API.

