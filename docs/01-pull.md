# Step 1 — Downloading a model from Hugging Face

What is actually on the Hub, how `shilp pull` gets it, and where it lands.

---

## What a model repo contains

A Hugging Face model repo is a git repo with a fixed-ish set of files. The ones that matter:

| File | What it is | Do you need it? |
|---|---|---|
| `*.safetensors` (+ `.index.json`) | the weights, in float precision, possibly sharded | only if you intend to change the model |
| `config.json` | architecture: layer count, hidden size, vocab size, rope settings | yes, for conversion |
| `tokenizer.json`, `tokenizer_config.json` | vocabulary and the **chat template** | yes |
| `*.gguf` | the same weights, already converted and possibly quantized | yes, if you just want to run it |
| `*.bin`, `*.pt` | legacy PyTorch pickles | no — skipped |
| `README.md` | the model card | no |

The distinction that matters:

- **safetensors repo** (e.g. `Qwen/Qwen2.5-1.5B-Instruct`) — the original, float weights. Needed for surgery, LoRA merging, or conversion. Cannot be served directly by this project.
- **GGUF repo** (e.g. `Qwen/Qwen2.5-1.5B-Instruct-GGUF`, `bartowski/...-GGUF`) — already converted, usually with several quantizations as separate files. Runnable immediately, on CPU.

Most popular models have both. If your only goal is to serve something today, take the GGUF.

---

## Commands

### See what you would download, first

```bash
shilp pull hf:Qwen/Qwen2.5-1.5B-Instruct-GGUF:q4_k_m --dry-run
```

```
Qwen/Qwen2.5-1.5B-Instruct-GGUF @ main  (1.0 GB)
   1.0 GB  qwen2.5-1.5b-instruct-q4_k_m.gguf
```

A dry run is worth the three seconds: it is where you discover the repo publishes six quants and you were about to grab the wrong one.

### Pull a runnable model

```bash
shilp pull hf:Qwen/Qwen2.5-1.5B-Instruct-GGUF:q4_k_m
# pulled qwen2.5-1.5b-instruct-gguf:q4_k_m  1.0 GB
#   ready to serve:  shilp serve --model qwen2.5-1.5b-instruct-gguf:q4_k_m
```

The reference syntax is `hf:<owner>/<repo>[:<quant>]`. The quant token is matched case- and separator-insensitively, so `q4_k_m` finds `Q4_K_M`.

### Pull float weights, to change them later

```bash
shilp pull hf:Qwen/Qwen2.5-1.5B-Instruct
# pulled qwen2.5-1.5b-instruct:main  (3.1 GB)
#   safetensors stored — next: shilp build -f Shilpfile  (or convert + quantize)
```

This stores the weights, the config and the tokenizer. Nothing is runnable yet — that is step 2.

### Pin a revision

```bash
shilp pull hf:Qwen/Qwen2.5-1.5B-Instruct --revision a09a35458c702b33eeacc393d103063234e8bc28
```

`main` moves. If you intend to publish results from a model, pin the commit sha — a dependency that can change underneath you is not a dependency. The revision is recorded in the manifest and the recipe either way, so at minimum you can tell later which one you got.

### Gated models

Llama, Gemma and a few others require accepting terms on the Hub and a token:

```bash
export HF_TOKEN=hf_...
shilp pull hf:meta-llama/Llama-3.2-3B-Instruct-GGUF:q4_k_m
```

Without a token you get a `502 hub_error` naming the repo — not a silent empty download.

---

## Where it goes

```
$SHILP_HOME/                       default ~/.shilp
  blobs/sha256-<hex>               file content, named by its own hash
  manifests/<name>/<tag>.json      what the model is
  recipes/<name>/<tag>.json        how it was made
  tmp/                             scratch during a pull; emptied afterwards
  logs/shilp.db                    telemetry
```

Three consequences worth understanding:

1. **The same file is stored once.** Pulling two tags that share a weights file costs the bytes once.
2. **You cannot corrupt a model by copying files around.** A blob's name is its SHA-256. If the bytes change, the name no longer matches and the loader refuses.
3. **Deleting a manifest does not delete blobs immediately** — `store.delete()` runs garbage collection, which removes only blobs nothing else references. `shilp gc --dry-run`-style output is available via `shilp gc` without `--yes`.

### Naming

The default name is the repo name, lowercased; the default tag is the quant (for GGUF) or the revision (for safetensors). Override either:

```bash
shilp pull hf:Qwen/Qwen2.5-1.5B-Instruct-GGUF:q4_k_m --name qwen --tag small
shilp serve --model qwen:small
```

---

## Choosing a quantization for your machine

Approximate sizes. The rule of thumb: **the model plus its KV cache must fit in RAM, with ~2 GB left for the OS.**

| Quant | 1.5B | 3B | 7–8B | Quality |
|---|---|---|---|---|
| `Q8_0` | 1.6 GB | 3.2 GB | 8.1 GB | indistinguishable from F16 |
| `Q6_K` | 1.3 GB | 2.5 GB | 6.6 GB | effectively lossless |
| `Q5_K_M` | 1.1 GB | 2.2 GB | 5.3 GB | near-lossless |
| **`Q4_K_M`** | **1.0 GB** | **2.0 GB** | **4.4 GB** | **the default; best size/quality** |
| `Q3_K_M` | 0.8 GB | 1.6 GB | 3.7 GB | noticeable on code and maths |
| `Q2_K` | 0.6 GB | 1.2 GB | 2.9 GB | smallest that stays coherent |

On an 8 GB Mac, `Q4_K_M` up to ~3B, or ~7B if you keep the context modest (`num_ctx 4096`). On 16 GB, 7–8B `Q4_K_M` is comfortable. On 24 GB+, 13–14B `Q4_K_M`.

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `no GGUF file matching quant 'q4_k_m'` | the repo uses different quant names | the error lists what the repo actually publishes |
| `... is split across shards, which shilp does not join yet` | multi-part GGUF | pick a repo publishing one file |
| `has no .safetensors weights` | GGUF-only repo pulled as raw | add `:<quant>` to the reference |
| `502 hub_error` on a gated repo | no token, or terms not accepted | `export HF_TOKEN=…` after accepting on the Hub |
| Download stalls then resumes | `huggingface_hub` resumes partial downloads | rerun; it continues rather than restarting |

## Next

[Step 2 — reshaping the model](02-reshape.md).
