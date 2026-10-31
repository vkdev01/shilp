# Step 2 — Reshaping the model

The step Ollama does not have. Four kinds of change, in the order they must happen.

---

## The four kinds of change, and why order matters

| Change | Command | Touches | Reversible? |
|---|---|---|---|
| Parameters and template | `shilp build -f Shilpfile` | manifest only | yes — delete the tag |
| Format (HF → GGUF) | `shilp convert` | new file, lossless | yes |
| Precision | `shilp quantize` | new file, lossy | yes |
| Structure | `shilp surgery …` | new weight files, lossy | yes — the source is untouched |
| Behaviour (fine-tune) | `shilp merge-lora` | new weight files | yes |

**If you intend to do more than one, the order is:**

```
merge LoRA  →  surgery  →  convert  →  quantize  →  build (params + template)
```

The reason is that quantization is a lossy *compression* step. Merging an adapter into already-quantized weights bakes the adapter's update through a rounding error you cannot undo. Surgery before conversion, because the tools that edit safetensors are mature and the tools that edit GGUF are not.

The reverse order is not a crime, it is just worse — and if you only want to run a model, skip to `build`.

---

## A. Parameters and template — the `Shilpfile`

The cheapest change, and usually the one that fixes the most. A `Shilpfile` describes a model in one readable file:

```
# Shilpfile
FROM hf:Qwen/Qwen2.5-1.5B-Instruct-GGUF:q4_k_m

PARAMETER temperature 0.6
PARAMETER top_p 0.95
PARAMETER num_ctx 8192
PARAMETER num_predict 512
PARAMETER repeat_penalty 1.1
PARAMETER stop "<|im_end|>"

SYSTEM """You are a terse assistant. One sentence unless asked for more."""

TEMPLATE """<|im_start|>system
{{ .System }}<|im_end|>
<|im_start|>user
{{ .Prompt }}<|im_end|>
<|im_start|>assistant
"""
```

```bash
shilp build -f Shilpfile --tag terse
shilp show qwen2.5-1.5b-instruct-gguf:terse
```

Directives:

| Directive | Meaning |
|---|---|
| `FROM` | `hf:org/repo[:quant]` (pulls if needed), an existing `name:tag` (copies its layers), or a local `.gguf` path |
| `PARAMETER` | sampling and limits; unknown names are rejected at build time, not ignored at inference time |
| `TEMPLATE` | Ollama-style: `{{ .System }}`, `{{ .Prompt }}`, `{{ .Response }}` |
| `SYSTEM` | system prompt, applied when the request has none |
| `ADAPTER` | path to a LoRA adapter to merge (used by `merge-lora`) |

Values may be `"quoted"` or `"""triple-quoted"""` for multi-line. `#` starts a comment.

**Why parameters live in the manifest rather than in the client.** Because every client that forgets to send `temperature` gets the model author's intended behaviour, and because changing behaviour becomes a one-file diff instead of a code change. A request can still override anything per call.

**Why the template is worth editing.** Most "this local model is dumb" reports are a wrong chat template — the model is being handed text in a shape it was never trained on. `shilp show` prints the template the model is currently using, and the one baked into the GGUF, so you can compare.

---

## B. Format — `convert`

HF safetensors → one GGUF file.

```bash
shilp pull hf:Qwen/Qwen2.5-1.5B-Instruct
shilp convert qwen2.5-1.5b-instruct:main --out ./qwen-f16.gguf --out-type f16
```

Requires a llama.cpp checkout, because `convert_hf_to_gguf.py` tracks every new architecture and reimplementing it would be a full-time job:

```bash
git clone https://github.com/ggml-org/llama.cpp ~/llama.cpp
export SHILP_LLAMACPP_DIR=~/llama.cpp
```

`--out-type` is `f32`, `f16` or `bf16`. Use `bf16` for models that shipped in bfloat16 — converting to f16 first and then to bf16 wastes a step and a little precision.

**Memory:** conversion holds the model in RAM. A 1.5B model needs ~4 GB; a 7B model needs ~16 GB. This is the step that sends laptop users to a rented GPU or Colab.

---

## C. Precision — `quantize`

```bash
shilp quantize ./qwen-f16.gguf --out ./qwen-q4.gguf --qtype Q4_K_M
# wrote ./qwen-q4.gguf  1.0 GB  (Q4_K_M)
#   estimated resident memory: 1.1 GB
```

| Type | Trade-off |
|---|---|
| `Q2_K` | smallest that is still coherent; noticeably dumber |
| `Q3_K_M` | fits a 13B on 8 GB; quality drops on code and maths |
| **`Q4_K_M`** | **the default: best size/quality balance** |
| `Q5_K_M` | near-F16 quality, ~20% larger than Q4_K_M |
| `Q6_K` | essentially lossless, ~35% larger |
| `Q8_0` | reference quality, barely worth it over Q6_K |
| `F16` | no quantization |

K-quants keep per-block scales rather than one scale per tensor, which is why `Q4_K_M` is close to lossless in conversation and why `Q2_K` is not. The `_M` means "medium": some tensors (attention output, feed-forward down-projections) are kept at higher precision because they are the sensitive ones.

---

## D. Structure — `surgery`

These edit a Hugging Face directory: tensors and `config.json`. The source is copied, never modified, so every edit is undone by deleting the output directory.

### Prune layers — make the model smaller and worse

```bash
shilp surgery prune ./qwen-1.5b --out ./qwen-1.5b-24L --keep 24
# { "layers_before": 28, "layers_after": 24, "tensors_dropped": 32 }
# note: Depth pruning removes capability, not just size.
```

Drops every transformer block at index ≥ `keep`. Memory and latency fall roughly in proportion, and so does quality — the last layers do a lot of work, so pruning the *tail* is gentler than pruning the middle. This is worth doing only with a fine-tune afterwards, or with an eval that proves the drop is acceptable. `shilp` reports what it removed; it does not pretend to know whether the result is still good.

### Truncate vocabulary

```bash
shilp surgery vocab ./model --out ./model-32k --size 32000
```

Slices the embedding matrix and the LM head to the first N rows. Correct **only** when the tokenizer really has that many tokens — after retraining a tokenizer on a narrow domain, for instance. As a shortcut to a smaller file it is a bug: the ids no longer decode to the text the model expects.

### Extend context with RoPE scaling

```bash
shilp surgery rope ./model --out ./model-32k-ctx --factor 2.0
```

Config-only, no tensors touched. Doubles the position range by rescaling RoPE frequencies. Quality holds reasonably to about 1.5× the training length and degrades past it, no matter what the config says. Test with a long-context probe — a needle-in-a-haystack or a summarise-this-transcript task — not by asking the model whether it remembers.

### Set the chat template (HF side)

```bash
shilp surgery template ./model --out ./model-tpl --file template.jinja
```

Writes `chat_template` into `tokenizer_config.json` in Jinja form. Use this when you are converting to GGUF afterwards and want the template baked in; use the `Shilpfile` `TEMPLATE` directive when you want it per-tag.

---

## E. Behaviour — `merge-lora`

```bash
pip install -e ".[finetune]"
shilp merge-lora ./qwen-1.5b ./my-lora-adapter --out ./qwen-1.5b-tuned
# next: shilp convert + shilp quantize, in that order
```

Folds a PEFT adapter into the base weights with `merge_and_unload`, then saves a plain HF directory. A merged model is one file that any GGUF toolchain understands, with zero inference-time overhead — the cost is that you cannot swap adapters per request, and you need the base model in float precision once.

**What a laptop can actually do here:** training a 1.5B LoRA needs ~8–12 GB of GPU memory with 4-bit loading and gradient checkpointing, so an 8 GB Apple Silicon machine can do it slowly on 0.5–1.5B models, and cannot do it on 7B. Merging is cheap by comparison. If fine-tuning is the goal, plan on Colab, Kaggle, or a rented A10/T4 for the training and bring the adapter home.

---

## The recipe is the deliverable

Every one of these steps appends to `recipes/<name>/<tag>.json`:

```json
{
  "ref": "qwen2.5-1.5b-instruct-gguf:terse",
  "parent": "sha256:…",
  "steps": [
    { "step": "pull", "spec": "hf:Qwen/Qwen2.5-1.5B-Instruct-GGUF", "revision": "main" },
    { "step": "quantize", "quant_type": "Q4_K_M" },
    { "step": "build", "params": { "temperature": 0.6 }, "template": true }
  ]
}
```

```bash
shilp show qwen2.5-1.5b-instruct-gguf:terse   # prints manifest + recipe
```

This is what makes the pipeline reproducible rather than a series of shell commands you will not remember. The `parent` digest also means a derived model can never silently drift from the base it claims to come from.

## Next

[Step 3 — serving it](03-serve.md).
