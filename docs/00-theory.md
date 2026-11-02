# 00 — Inference from first principles

**What the bare minimum looks like, what llama.cpp actually does, and why this project delegates to it.**

This is the foundational document in `docs/`. The other four describe *how to use* shilp. This one describes *what is happening underneath it*, from the tokenizer upward, and answers the question a reviewer will ask: if a language model is just matrix multiplication, why does a model server need a C++ runtime at all?

Every number below is tagged, because the difference matters:

| Tag | Meaning |
|---|---|
| **measured** | produced by a program run for this document, on the machine described |
| **computed** | derived from a model's published `config.json` using the formulas here, with no weights loaded |
| **expected** | a prediction from the bandwidth model, in the range real implementations achieve |

The measured numbers come from a deliberately modest machine — 2 vCPU, 2 GB RAM, no GPU, Linux x86_64 — which is the point: everything here reproduces on a laptop.

---

## 1. The problem, stated precisely

**Inference** is the act of turning a sequence of token ids into a probability distribution over the next token, and then choosing one. Repeated, that is text generation.

**Serving** is making inference available to programs that are not the program holding the weights: a network boundary, a request/response contract, and a lifecycle for the model in memory.

Every inference system that has ever existed, from a 10-line script to a 400-GPU cluster, performs the same five functions. They are the minimum:

| # | Function | Question it answers |
|---|---|---|
| 1 | **Tokenise** | How does text become integers? |
| 2 | **Forward** | How do integers become scores? |
| 3 | **Cache** | How do we avoid recomputing what we already computed? |
| 4 | **Sample** | How does a score become a chosen token? |
| 5 | **Transport** | How does a token reach the client as it is produced? |

Nothing in that list is optional. A system missing (3) is quadratically slow; one missing (5) makes users wait for the entire answer before seeing any of it; one missing (1) is not a language model at all.

What *is* optional is **who implements each one**. That is the entire subject of this document: `transformers` implements all five in Python, llama.cpp implements all five in C++, and shilp implements (5) and the lifecycle while delegating (1)–(4).

---

## 2. The theory, with numbers

### 2.1 Tokenisation

A **tokenizer** maps text to integers using a fixed **vocabulary**. The vocabulary is built by **byte-pair encoding**: start from 256 raw bytes, repeatedly merge the most frequent adjacent pair, stop at the target vocabulary size (typically 32k–152k).

The consequence is that token count is not word count, and the ratio is not constant. **Measured**, using the GPT-2 tokenizer:

| Input | Tokens | Characters | Note |
|---|---|---|---|
| `hello world` | 2 | 11 | common words collapse |
| `Hello, world!` | 4 | 13 | punctuation is its own token |
| `getUserAccountById` | 4 | 18 | camelCase splits on subwords |
| `नमस्ते दुनिया` | **23** | 12 | two words, 23 tokens — Devanagari was rare in training |

That last row is the most important number in this section. The same two words cost 11× more in Hindi than in English, because the tokenizer's merges were learned on a corpus dominated by English. You pay for this twice: in context window, and in compute proportional to token count. It is also why multilingual models train their own tokenizer rather than reusing an English one.

Two derived quantities matter for everything that follows:

```
prompt_tokens = len(tokenize(prompt))
cost          ≈ prompt_tokens × price_per_token        (hosted)
latency       ∝ prompt_tokens                           (prefill, §2.4)
```

### 2.2 The forward pass

A **parameter** is one learned number. A forward pass reads all of them and produces one **logit** per vocabulary entry.

The shapes, for a decoder-only transformer with `L` layers, hidden size `H`, intermediate (MLP) size `I`, `H_q` query heads, `H_kv` key/value heads, head dimension `d = H / H_q`, vocabulary `V`:

```
attention per layer : H·(H_q·d) + 2·H·(H_kv·d) + (H_q·d)·H
MLP per layer       : 3·H·I        (gated: gate, up, down)
                     2·H·I        (ungated)
embeddings          : V·H  (tied) or 2·V·H  (untied)
norms               : ≈ 2·L·H

total ≈ L·(attention + MLP) + embeddings + norms
```

This formula is worth trusting only if it reproduces known counts. **Computed** from the published configs:

| Model | Formula | Published | Match |
|---|---|---|---|
| Qwen2.5-1.5B-Instruct | 1.54 B | 1.54 B | ✓ |
| Qwen2.5-7B-Instruct | 7.62 B | 7.62 B | ✓ |

The first version of this formula gave 1.78 B for the 1.5B model — a 15% error — because it assumed untied embeddings. Qwen2.5-1.5B sets `tie_word_embeddings: true`, so the embedding matrix is counted once, not twice. That single flag accounts for the whole discrepancy (V·H = 233 M parameters). *A parameter estimate that does not reproduce the model card is wrong, and the error is usually a config flag you did not read.*

Where the parameters live matters for pruning (see [02-reshape.md](02-reshape.md)). **Computed** for Qwen2.5-1.5B:

| Component | Parameters | Share |
|---|---|---|
| MLP (gate, up, down) | 1,156 M | 74.9% |
| Embeddings (tied) | 233 M | 15.1% |
| Attention (q, k, v, o) | 154 M | 10.0% |
| Norms | 0.09 M | 0.0% |

Three quarters of this model is feed-forward. That is why quantisation effort concentrates on MLP tensors, and why depth pruning (§[11] in the book) removes MLP capacity in proportion to the layers dropped.

### 2.3 Attention and the KV cache

Attention lets a token weight earlier tokens. Each token emits a **query**, a **key** and a **value**; scores are `softmax(QKᵀ/√d)` used to weight a sum of values, masked so position *i* sees only ≤ *i*.

The **KV cache** stores the keys and values of every token already processed, so each new token computes only its own. Its size is exact arithmetic:

```
bytes per token = 2 (K and V) × L × H_kv × d × bytes_per_element
```

**Computed** at fp16, with the context cost that follows:

| Model | L | H_kv | d | Bytes/token | 4k ctx | 8k ctx | 32k ctx |
|---|---|---|---|---|---|---|---|
| Qwen2.5-1.5B | 28 | 2 | 128 | 28 KiB | 0.11 GiB | 0.22 GiB | 0.88 GiB |
| Qwen2.5-7B | 28 | 4 | 128 | 56 KiB | 0.22 GiB | 0.44 GiB | 1.75 GiB |
| Llama-3-8B | 32 | 8 | 128 | 128 KiB | 0.50 GiB | 1.00 GiB | 4.00 GiB |

Two readings. First, **grouped-query attention is why these are small**: Llama-3-8B has 32 query heads but only 8 KV heads, cutting the cache 4× against the naive version. Second, **context length is a memory decision, not a quality setting**. At 32k, the cache for an 8B model costs as much as the weights did before quantisation. `num_ctx 32768` "because the model supports it" is how laptops start swapping.

### 2.4 The two regimes: prefill and decode

This is the single most important distinction in inference engineering, and it is a distinction between two *different bottlenecks*.

**Prefill** processes the whole prompt. Every position is independent given the causal mask, so the work parallelises across the sequence and the matrix multiplies become large and wide. Prefill is **compute-bound**: it saturates the arithmetic units.

**Decode** produces one token per forward pass, and each pass depends on the previous one. There is no parallelism across tokens, so every pass reads the entire model from memory to perform comparatively little arithmetic. Decode is **memory-bandwidth-bound**.

That gives a hard ceiling:

```
tokens/second ≤ memory_bandwidth / bytes_read_per_token
              ≈ bandwidth / model_size_in_bytes
```

**Computed** for Q4_K_M (4.89 effective bits/weight) using this formula:

| Model | File size | KV/token | M1 (68 GB/s) | M2 Pro (200) | M3 Max (400) | x86 DDR4 (40) |
|---|---|---|---|---|---|---|
| Qwen2.5-1.5B | 0.94 GB | 28 KiB | 72 | 212 | 425 | 43 |
| Qwen2.5-7B | 4.66 GB | 56 KiB | 15 | 43 | 86 | 9 |

**Expected** real throughput is 50–70% of these ceilings — the gap is imperfect bandwidth utilisation, attention over the cache, and sampling overhead. So a 7B Q4_K_M on an M2 Pro should measure ~21–30 tok/s, and a 1.5B ~106–149.

The model also predicts something counter-intuitive and useful: **on a GPU, throughput per user rises with load**, because reading the weights once can serve many sequences (continuous batching). On a single laptop stream there is nothing to batch, which is why a GPU's headline advantage is mostly a *concurrency* advantage, not a single-request one.

And it predicts when it does **not** apply: if the model is small enough to sit in cache, it is not bandwidth-bound at all. **Measured** on the 2 GB test box with a 0.4 MiB model: 69 tok/s, against a formula prediction of ~178,000. The formula is right and the model is out of scope — 0.4 MiB fits in L2. *Always check the assumptions of a performance model before quoting its output.*

### 2.5 Sampling

The forward pass ends with **logits**, one per vocabulary entry. Sampling turns those into a choice.

**Temperature** divides logits before the softmax; **top-k** keeps the k best; **top-p** (nucleus) keeps the smallest set whose cumulative probability exceeds p; **min-p** keeps tokens with `p ≥ min_p × p_max`. With logits `[3.0, 2.0, 1.0, 0.5]`, **computed**:

| Temperature | p₁ | p₂ | p₃ | p₄ | top-p 0.9 keeps |
|---|---|---|---|---|---|
| 0.5 | 0.862 | 0.117 | 0.016 | 0.006 | 1 token |
| 1.0 | 0.631 | 0.232 | 0.085 | 0.052 | 3 tokens |
| 2.0 | 0.442 | 0.268 | 0.163 | 0.127 | 4 tokens |

`T = 0` is greedy decoding: always the argmax. It is deterministic and usually bad for chat, because the highest-probability continuation is often the most generic one — and it loops. **Measured** on a small random model, greedy decoding produced `stairs stairs stairs stairs…` for twelve consecutive tokens. That is the degenerate case of the same behaviour that makes greedy output flat and repetitive on real models.

### 2.6 The memory equation

Everything above composes into the equation that decides whether a model fits:

```
resident_bytes ≈ weight_bytes + KV_cache_bytes + runtime_overhead
```

**Computed** for two real models at three precisions, with the cache at 8k context:

| Model | Precision | Weights | KV @ 8k | Runtime | Total |
|---|---|---|---|---|---|
| Qwen2.5-1.5B | fp32 | 5.75 GiB | 0.22 | ~0.7 | ~6.7 GiB |
| Qwen2.5-1.5B | fp16 | 2.88 GiB | 0.22 | ~0.7 | ~3.8 GiB |
| Qwen2.5-1.5B | Q4_K_M | 0.88 GiB | 0.22 | ~0.7 | ~1.8 GiB |
| Qwen2.5-7B | fp16 | 14.18 GiB | 0.44 | ~0.7 | ~15.3 GiB |
| Qwen2.5-7B | Q4_K_M | 4.34 GiB | 0.44 | ~0.7 | ~5.5 GiB |

The `runtime` term is not a rounding error, and §3.6 measures it: a Python + PyTorch process carrying a 0.4 MiB model still occupied **707 MiB** of RSS. Framework overhead is frequently larger than the model for anything under ~1B parameters. Quantisation shrinks the first term; a leaner runtime shrinks the third; only careful `num_ctx` shrinks the second.

### 2.7 What is actually on disk: safetensors and GGUF

Both formats hold the same numbers — the weights. They are **containers**, not different models, and the two engines in this repository each read exactly one of them. Understanding the split explains most of the confusion around "why is my model 14 GB here and 4 GB there".

| | safetensors | GGUF |
|---|---|---|
| What it is | a bare tensor container | a runtime container for GGML / llama.cpp |
| Header | 8-byte length, then JSON: per-tensor dtype, shape, byte offsets | magic `GGUF`, version, metadata KV block, tensor-info table |
| What is inside | tensors only | tensors **plus** architecture, context length, RoPE settings, tokenizer vocab and merges, chat template, quantisation type |
| Files per model | a directory: `config.json`, `tokenizer.json`, `tokenizer_config.json`, `generation_config.json`, plus sharded weight files | one file |
| Precision | whatever the model was trained in: fp32, fp16, bf16 | fp32 / fp16 **and** a type system for quantised blocks (`Q4_K`, `Q6_K`, `Q8_0`, i-quants) |
| Memory-mapped | designed for lazy loading; PyTorch can mmap it | the normal path — only touched pages are read |
| Read by | PyTorch, `transformers`, MLX, training code | llama.cpp and everything built on it: Ollama, LM Studio, shilp |
| Origin | Hugging Face, 2022, to replace pickle checkpoints | llama.cpp, 2023, successor to the older GGML format |

#### Format is not precision

This is the conflation worth killing: **GGUF is not a compression format.** It is a container that *can* hold quantised weights. A `F16` GGUF is essentially the same size as the same model in `F16` safetensors — converting is a format change and lossless. The size difference everyone quotes comes from quantisation, which happens *after* conversion:

```
safetensors fp16   →  convert_hf_to_gguf  →  GGUF f16  →  llama-quantize  →  GGUF Q4_K_M
   2.88 GiB             (lossless)              2.88 GiB     (lossy)            0.88 GiB
```

Qwen2.5-1.5B, from its config: 2.88 GiB at fp16 **either way**, 0.88 GiB at Q4_K_M. The 3.3× is quantisation's doing, not the container's.

The second half of the difference is **self-description**. A safetensors file is weights and nothing else: it does not know it is a Qwen, does not know its context length, carries no tokenizer and no chat template. Those live in sibling JSON files — which is why a Hugging Face repo is a directory and a GGUF is a single download. GGUF carries the manual printed on the casing, which is exactly why `shilp pull` can register a model from one file.

#### What each engine needs

Neither engine reads the other's format, which is the whole reason the loader has two branches:

```bash
shilp pull hf:Qwen/Qwen2.5-1.5B-Instruct               # → safetensors directory
shilp serve --engine transformers                       # PyTorch reads that

shilp pull hf:Qwen/Qwen2.5-1.5B-Instruct-GGUF:q4_k_m   # → one GGUF file
shilp serve                                             # llama.cpp reads that
```

`transformers` has no GGUF loader; llama.cpp has no safetensors loader. So `engine/loader.py` materialises a *directory* for one backend and hands the other a single blob path. Hand either engine the wrong shape and it names the fix rather than failing obscurely.

#### Which to use

**safetensors** when you intend to *change* the model — fine-tuning, merging a LoRA, pruning layers, editing the tokenizer. It is the training format, the whole ML ecosystem reads it, and any tensor is one `print` away. See [02-reshape.md](02-reshape.md).

**GGUF** when you intend to *run* it — quantised, one file, no Python in the hot path, mmap'd, portable across CPU, Metal, CUDA and Vulkan. This is what a laptop actually needs, and it is why llama.cpp is the default engine.

Conversion is one-way in practice. HF → GGUF at the same precision is lossless; GGUF → safetensors on a quantised file returns *approximations* of the original floats, not the originals. Convert once, late, after every change you wanted to make.

---

## 3. Path A — the bare minimum, in Python

The complete implementation is [`scripts/bare_inference.py`](../scripts/bare_inference.py). It is ~250 lines with comments, and it does the whole job: download, load, tokenise, generate, stream, serve. Run it before reading the rest; the point lands better after you have seen it work.

### 3.1 What you install

```bash
pip install torch transformers fastapi uvicorn
```

That is it. No compilation, no C++, no model conversion. On a CPU-only machine use the CPU wheel to avoid pulling ~2 GB of CUDA libraries you cannot use:

```bash
pip install --index-url https://download.pytorch.org/whl/cpu torch
```

### 3.2 Load the weights

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float32)
model.eval()
```

Three things this line does that are easy to miss:

- **Downloads** the checkpoint (safetensors) into `~/.cache/huggingface`, once.
- **Materialises every weight in RAM** as a dense tensor. There is no memory mapping; the process now owns the full weight bytes.
- **`dtype` decides how many bytes per parameter**: fp32 = 4, fp16/bf16 = 2. For a 1.5B model that is the difference between 5.75 GiB and 2.88 GiB.

`model.eval()` disables dropout. Without it, output is non-deterministic even at temperature 0.

### 3.3 Build the prompt

```python
if tokenizer.chat_template:
    prompt = tokenizer.apply_chat_template(messages, tokenize=False,
                                          add_generation_prompt=True)
else:
    prompt = f"Q: {user}\nA:"      # hope for the best
```

**Measured**: on a base model (`sshleifer/tiny-gpt2`), `chat_template` is `None`, and calling `apply_chat_template` raises `ValueError`. This is the failure the book's chapter 4 is about, seen from the code side: an instruct model is only well-behaved when the prompt matches the format it was trained on, and the format is *data that ships with the tokenizer*, not something you invent.

### 3.4 The generation loop

```python
with torch.no_grad():
    out = model.generate(**inputs, max_new_tokens=128)
```

`generate` is a loop over forward passes with the KV cache handled internally:

```
prefill:  logits = forward(prompt_tokens)          # one wide pass
loop:     next   = sample(logits[-1])
          logits = forward(next, past_kv=cache)    # one narrow pass
          cache  = cache ∪ new_kv
until EOS or max_new_tokens
```

**Measured** on the test box (2 vCPU, 2 GB), model of 102,714 parameters:

| Quantity | Value |
|---|---|
| Load (cold, including download) | 11.8 s |
| Weights in memory | 0.4 MiB (4.0 bytes/param = fp32) |
| Prompt tokens | 12 |
| Prefill (one pass over 12 tokens) | 30 ms |
| Decode, 32 tokens | 463 ms → **69.1 tok/s**, 14.5 ms/token |
| First decoded token | 12 ms after decode started |
| Peak RSS of the process | 707 MiB |

Read the last row against the second. The framework's footprint was **~1,700× the model**. For this model that is absurd; for a 7B model the same overhead is noise. The crossover is somewhere around 1B parameters, and it is the reason "just use transformers" and "just use llama.cpp" can both be right answers to the same question.

### 3.5 Serve it

Forty lines of FastAPI, in the script:

```python
@app.post("/v1/chat/completions")
def chat(req: ChatRequest):
    prompt = build_prompt(tokenizer, req.messages)
    def events():
        for chunk in stream_tokens(tokenizer, model, prompt, ...):
            yield f"data: {json.dumps({...})}\n\n"
        yield "data: [DONE]\n\n"
    return StreamingResponse(events(), media_type="text/event-stream")
```

`stream_tokens` drives `generate` in a **background thread** and consumes a `TextIteratorStreamer`. That threading is not incidental: `generate` blocks for the entire answer, so without a second thread there is nothing to stream. This is the same problem shilp solves more carefully in `server/stream.py` — and the same reason that file exists.

**Measured**: 8 tokens arrived as 9 separate SSE chunks; the non-streaming path returned the same text with `tok_per_s: 151.4` (higher than the 69.1 above only because fewer tokens were generated, so fixed per-call overhead is amortised less — a reminder that short runs are dominated by overhead).

### 3.6 The four walls of the bare path

Path A is a complete inference server. It is also, on a laptop, unusable for anything larger than ~1B parameters. The walls are specific:

**Wall 1 — memory.** Weights are dense and unshared: fp32 is 4 bytes per parameter, and the whole thing must be resident. Qwen2.5-7B at fp32 is 30 GB. There is no quantisation in this path at all; PyTorch has no Q4_K_M.

**Wall 2 — speed.** `transformers` on CPU performs general-purpose matmuls on float32 tensors. It has no 4-bit kernels, no NEON/AVX quantised dot products, no Metal. Measured against the bandwidth ceiling, a fp32 path is 4–8× larger *and* its kernels are slower per byte, so the deficit compounds: at fp32, Qwen2.5-1.5B's ceiling on an M2 Pro is 32 tok/s, versus 212 tok/s at Q4_K_M. **Computed.**

**Wall 3 — format.** There is no GGUF in this path. The checkpoint is a directory of sharded safetensors plus configs, which is fine for training and awkward for distribution: nothing can `mmap` it, nothing else can read it, and every consumer needs the same Python stack.

**Wall 4 — concurrency.** One model, one process, one GIL for the Python parts. There is no eviction policy, no request queue, no notion of a second model. `uvicorn --workers 4` would load four copies of the weights.

None of these are flaws in `transformers`. It is a **research library**, optimised for breadth of architectures, hackability and training — not for running one model cheaply for a long time. Judging it by serving criteria is a category error, and the reverse is equally true.
---

## 4. What llama.cpp actually is

llama.cpp is a C/C++ inference runtime for models stored in the GGUF format. It began in 2023 as an inference implementation of the LLaMA architecture on Apple Silicon and grew into the de-facto portable runtime for open weights: MIT licensed, no Python in the hot path, and running on hardware from a Raspberry Pi to a multi-GPU workstation.

It is a **library and a set of tools**, not a service. The repository ships:

| Component | Role |
|---|---|
| `ggml` | the tensor library: allocation, quantised dot products, SIMD kernels |
| the runtime | graph construction, KV cache, sampling, tokenizer |
| `convert_hf_to_gguf.py` | Hugging Face safetensors → GGUF |
| `llama-quantize` | re-quantise a GGUF to another precision |
| `llama-server` | a reference HTTP server, not a production scheduler |

### 4.1 The seven things it provides

Map these against the five functions from §1 and the four walls from §3.6:

**1. Quantised compute kernels.** This is the core contribution. A `Q4_K_M` weight is not a float; it is a 4-bit integer plus a per-block scale. Multiplying it requires dequantise-and-dot-product fused in SIMD, hand-written per architecture: NEON on ARM, AVX2/AVX512 on x86, Metal Shading Language on Apple GPUs, CUDA on NVIDIA. This is tens of thousands of lines of code that exists nowhere else in a usable form, and it is what turns Wall 1 and Wall 2 from "impossible" into "routine".

**2. The GGUF container.** One file holding metadata, tokenizer, chat template and weights, laid out so tensors are aligned and directly readable. Self-describing: no sidecar configs, no Python needed to interpret it.

**3. Memory mapping.** The weights file is `mmap`ed rather than read. The OS pages in only the tensors actually touched, so resident memory tracks the working set rather than the file size, and *loading* is nearly free. This is why `llama.cpp` starts in milliseconds where `transformers` spends seconds parsing and materialising.

**4. KV cache management.** Allocation sized to `n_ctx`, per-layer storage, and context shifting when the window fills. Wall 1's second term, handled.

**5. A sampler.** Temperature, top-k, top-p, min-p, typical, mirostat, repetition penalties, and grammar-constrained decoding via GBNF.

**6. Tokenizer and chat templates in C++.** BPE and SentencePiece implemented natively, plus Jinja chat-template rendering — so the correct prompt is produced by the runtime, not by a Python wrapper that might disagree with it.

**7. Hardware backends.** CPU, Metal, CUDA, Vulkan, ROCm, SYCL behind one interface, chosen at build time or runtime.

### 4.2 What it is not

- **Not a scheduler.** `llama-server` handles multiple slots, but there is no paged-attention continuous batching. If you need high concurrency on a GPU, that is vLLM's job (§8).
- **Not a training framework.** It applies LoRA adapters at inference; it does not train them.
- **Not a model store.** No tags, no manifests, no provenance. That is shilp's layer (§7).
- **Not an API standard.** Its OpenAI-compatible endpoint is a convenience; the API surfaces in shilp exist because clients expect *specific* dialects, not because llama.cpp defines them.

---

## 5. Path B — the same job with llama.cpp

In shilp, the whole of Path B is two commands:

```bash
shilp pull hf:Qwen/Qwen2.5-1.5B-Instruct-GGUF:q4_k_m
shilp serve --model qwen2.5-1.5b-instruct-gguf:q4_k_m
```

And underneath, the file that actually talks to the engine is `src/shilp/engine/llamacpp.py` — **146 lines**, of which the load is:

```python
self._llm = Llama(
    model_path=str(gguf_path),   # mmap'd, quantised, already converted
    n_ctx=num_ctx,
    n_threads=threads or None,
    n_gpu_layers=gpu_layers,
)
```

…and the generation is a loop over the binding's chunk iterator. There is no matrix multiplication in this repository. There is no tokenizer, no sampler, no KV cache code.

### 5.1 What changes, concretely

| | Path A (transformers) | Path B (llama.cpp) |
|---|---|---|
| Install | `pip install torch transformers` (~700 MB, no compiler) | `pip install llama-cpp-python` (compiles C++, 3–10 min) |
| Model artifact | safetensors directory | one GGUF file |
| Precision | fp32 / fp16 / bf16 | 2–8 bit K-quants, plus f16/f32 |
| Qwen2.5-1.5B on disk | 5.75 GiB (fp32) / 2.88 GiB (fp16) | **0.88 GiB** (Q4_K_M) |
| Qwen2.5-7B on disk | 30.5 GiB (fp32) / 14.2 GiB (fp16) | **4.34 GiB** (Q4_K_M) |
| Load | parse + materialise, seconds | mmap, milliseconds |
| Decode ceiling, 1.5B, M2 Pro | 32 tok/s (fp32) / 65 (fp16) | **212 tok/s** (Q4_K_M) |
| Decode ceiling, 7B, M2 Pro | 7 tok/s (fp32) / 13 (fp16) | **43 tok/s** (Q4_K_M) |
| Apple GPU | not used | Metal |
| Chat template | Python, from the tokenizer | in the runtime, or from the manifest |
| Change the weights | native (it is the training format) | convert → quantise → edit |
| Debug a kernel | read Python | read C++ |

The two figures in bold are the whole argument. Q4_K_M makes a 7B model **3.3× smaller and 3.3× faster** than fp16 — and against fp32, **7× smaller and 6× faster**. Those are not tuning gains; they are the difference between a model that runs on your laptop and one that does not.

### 5.2 What it costs

Honesty requires the other column, and it is real:

- **A compiler in the install path.** `pip install llama-cpp-python` builds C++; a missing toolchain turns setup into a debugging session. shilp's answer is `shilp doctor` and the `fake` engine, so the API can be developed before the engine exists.
- **A conversion step.** You cannot serve safetensors directly. That is exactly why `shilp convert` and `shilp quantize` exist, and why [02-reshape.md](02-reshape.md) insists on doing surgery *before* conversion.
- **Opaque internals.** When quality is bad, you are reading C++ or guessing. With `transformers` you can `print` a tensor.
- **Quantisation is lossy.** 4-bit is not free (§2.6): prose survives, arithmetic and long-form code degrade first. This project ships the trade-off table and, so far, no eval harness to measure the damage — the honest gap named in [04-roadmap.md](04-roadmap.md).

### 5.3 Both paths ship

The bare path in §3 is not a teaching sketch that gets thrown away. It is
`src/shilp/engine/transformers_engine.py` — a second implementation of the same
`Engine` protocol, selectable at runtime:

```bash
shilp serve --model qwen:main --engine transformers
shilp serve --model qwen:q4_k_m                # llama.cpp, the default
```

Nothing else changes. Not the loader, not either API surface, not the telemetry,
not the eviction policy. That is the test of whether a boundary is real: the
diff is one file, plus a branch in the factory and a materialisation step in the
loader that turns a manifest's blobs into a directory `transformers` can read.

It also makes the trade-off measurable on your own machine rather than argued
from this document: serve the same prompt through both engines and read
`shilp usage`. Expect `transformers` to show roughly 3–7× the resident memory and
3–7× lower tok/s for the same model, which is exactly what the bandwidth model in
§2.4 predicts, since dense fp16 weights are 3–7× larger than the same weights at
Q4_K_M.

What each engine needs is not the same, and each says so when handed the wrong
shape: llama.cpp wants one GGUF blob, `transformers` wants a safetensors
directory. `shilp pull hf:org/model` stores the latter, `shilp pull
hf:org/model-GGUF:q4_k_m` the former.

---

## 6. Why this project delegates instead of implementing

The defensible version of the argument, in three parts.

**1. The work being delegated is not the work this project is about.** Of 3,336 non-blank, non-comment lines in `src/`, the file that touches the engine is 146 — **4.4%**. The other 96% is the model lifecycle: a content-addressed store, tagged manifests, recipes, conversion and surgery orchestration, residency policy, two API dialects, telemetry, metrics, tests. Those are the problems that are unsolved for a user and interesting to solve. Quantised matmul is solved, and better than I would solve it.

**2. The boundary is an interface, not a dependency.** Nothing outside `engine/llamacpp.py` imports `llama_cpp`. The server depends on `Engine.chat() -> Iterator[Chunk]`. Two consequences make the claim testable rather than rhetorical:

- `engine/fake.py` implements the protocol with no model, and the *entire* HTTP surface — streaming, status codes, telemetry, metrics, eviction — is tested against it, in milliseconds, with no weights and no network.
- Adding a vLLM or MLX backend is one new file plus one branch in the loader. Nothing else changes.

**3. It is the same call every serious system makes.** Ollama is a Go service over llama.cpp. LM Studio is a UI over llama.cpp. `llama-cpp-python` itself is a ctypes binding over the C API. The stack is:

```
FastAPI  →  shilp Engine protocol  →  llama-cpp-python (ctypes)  →  llama.cpp (C++)  →  kernels
```

Three layers of indirection before arithmetic — and that is normal. You would not reimplement TCP to write an HTTP service. The engineering judgement is *where* to draw the line, and the line here is at "turn tokens into logits".

What would it mean *not* to delegate? Two options, both worse for this project's purpose: write your own kernels (months, slower, and pointless unless kernels are what you want to learn), or own the layer above — which is precisely what shilp does.

---

## 7. When llama.cpp is the wrong answer

A document that only argues one side is a sales pitch. Four cases where Path B is the wrong choice:

| Situation | Use instead | Why |
|---|---|---|
| Many concurrent users on one GPU | **vLLM** | Paged attention + continuous batching; throughput per user rises with load. llama.cpp has no equivalent |
| Apple Silicon only, maximum speed | **MLX** | Tuned for unified memory; competitive with or ahead of Metal in llama.cpp |
| Training, fine-tuning, or research on a new architecture | **transformers** | It *is* the training format; llama.cpp cannot train |
| A brand-new architecture published last week | **transformers** | Community ports to GGUF take days; `trust_remote_code` works today |
| Serverless, sub-100 ms cold start, small models | **ONNX Runtime / CTranslate2** | Smaller runtime, faster cold start than either path here |

The pattern: llama.cpp optimises for **one model, one machine, many formats, minimal dependencies**. The moment your constraint is concurrency, training, or a brand-new architecture, a different tool is correct.

---

## 8. Worked example: one request, both paths

The same request through both paths: **Qwen2.5-1.5B**, 200-token prompt, 100-token answer, on an M2 Pro (200 GB/s), 16 GB RAM. All figures **computed** from the formulas above; the throughput ranges are **expected** (50–70% of ceiling).

| | Path A, fp16 | Path A, fp32 | Path B, Q4_K_M |
|---|---|---|---|
| Weights on disk | 2.88 GiB | 5.75 GiB | **0.88 GiB** |
| Resident, incl. runtime + 8k cache | ~3.8 GiB | ~6.7 GiB | **~1.8 GiB** |
| Decode ceiling | 65 tok/s | 32 tok/s | **212 tok/s** |
| Expected decode | 33–46 tok/s | 16–22 tok/s | **106–149 tok/s** |
| 100-token answer takes | 2.2–3.0 s | 4.5–6.3 s | **0.7–0.9 s** |
| Cold start | seconds (parse + copy) | seconds | **~milliseconds (mmap)** |
| Fits in 16 GB alongside an OS? | yes | yes | yes, comfortably |

Two footnotes that keep this honest.

**Cold start is not free for Path B either.** `mmap` makes *loading* instant, but the first forward pass still faults pages in from disk. The saving is that you pay it lazily, per tensor, inside prefill — not as an up-front copy of the whole file. On a cold page cache, TTFT for both paths includes reading ~1 GB from disk; the difference is that Path A reads it *twice* (into page cache, then into anonymous memory) while Path B reads it once.

**The 1.5B case flatters Path A**, because 2.88 GiB fits in RAM either way. Run the same comparison at 7B and Path A's fp16 ceiling drops to 13 tok/s while Path B holds 43 — a 3.3× gap that decides whether the thing is usable for chat. At 13B and above, fp16 on a 16 GB laptop stops fitting at all, and the argument stops being about speed.

---

## 9. Glossary

Terms used in this document. The full 52-term glossary is in the book (`shilp-book.docx`).

**Attention** — mechanism weighting other tokens' contributions via query/key similarity. **Backend** — hardware path in llama.cpp (CPU, Metal, CUDA, Vulkan). **Bandwidth-bound** — limited by bytes moved per second rather than arithmetic. Decode is this. **BPE** — byte-pair encoding; the algorithm that builds a vocabulary by merging frequent byte pairs. **Chat template** — the format-specific string layout an instruct model was trained on; ships with the tokenizer. **Compute-bound** — limited by arithmetic throughput. Prefill is this. **Continuous batching** — admitting new requests into an in-flight batch at token boundaries; vLLM's main advantage. **Decode** — the phase producing one token per forward pass. **Dense weights** — a full-precision tensor, as opposed to a quantised block. **dtype** — the numeric type of a tensor; determines bytes per parameter. **GGUF** — llama.cpp's single-file container: metadata, tokenizer, template, aligned quantised tensors. **GQA** — grouped-query attention; several query heads share one KV head, shrinking the cache. **KV cache** — stored keys and values of processed tokens, so each new token computes only itself. **K-quant** — llama.cpp's block quantisation family with per-super-block scales and mixed precision. **Logits** — unnormalised scores, one per vocabulary entry. **min-p** — keeps tokens with probability ≥ min_p × p_max. **mmap** — mapping a file into virtual memory so the OS pages in only what is touched. **Prefill** — the parallel pass over the prompt; sets TTFT. **Quantisation** — reducing weight precision; lossy, and the reason models fit on laptops. **Resident set size (RSS)** — physical memory a process actually holds. **Sampling** — choosing a token from the distribution. **safetensors** — pickle-free tensor format: 8-byte header, JSON header, raw data. **Softmax** — exponentiate and normalise logits into probabilities. **temperature** — divides logits before softmax. **token** — one integer id from the vocabulary; ≈¾ of an English word. **top-k / top-p** — truncation filters. **TTFT** — time to first token; dominated by prefill. **vocabulary** — the fixed set of token ids a model knows.

---

## 10. Exercises

Each one is designed to break an assumption rather than confirm it.

1. **Reproduce the ceiling.** Run `scripts/bare_inference.py generate` on your machine with a model that fits, and compare measured tok/s against `bandwidth / bytes`. If you beat the ceiling, your model is in cache — check the size. If you are far under it, something other than bandwidth is the limit.

2. **Find the crossover.** At what model size does PyTorch's runtime overhead stop mattering relative to the weights? You measured 707 MiB of RSS for a 0.4 MiB model. Compute the size at which overhead is 10% of resident memory, then check it against a real 1.5B load.

3. **Break the tokenizer.** Tokenise the same sentence in English, Hindi, and a language with a non-Latin script, and in each case report tokens-per-character. Then explain, in terms of BPE merges, why the ratios differ.

4. **Prove the template matters.** Serve the same instruct model twice through Path A — once with `apply_chat_template`, once with a hand-written `Q: … A:` format — and compare a three-turn conversation. The degradation is the point.

5. **Make greedy loop.** Generate 200 tokens at `temperature=0` on a small model and look for repetition. Then raise temperature to 0.7 and re-run. Explain what the argmax is doing.

6. **Measure the conversion.** Convert one model to GGUF at f16 and at Q4_K_M, and report file size, ceiling, and — for the same 10 prompts — how often the Q4 answer differs *in substance* from the f16 answer. This is the eval harness from [04-roadmap.md](04-roadmap.md), built by hand.

---

## 11. Further reading

Primary sources for the concepts used above. Listed by title so they are findable regardless of link rot.

- **Attention Is All You Need** — Vaswani et al., 2017. arXiv:1706.03762. The transformer.
- **Neural Machine Translation of Rare Words with Subword Units** — Sennrich et al., 2016. arXiv:1508.07909. BPE for text.
- **LoRA: Low-Rank Adaptation of Large Language Models** — Hu et al., 2021. arXiv:2106.09685.
- **FlashAttention** — Dao et al., 2022. arXiv:2205.14135. IO-aware exact attention; the reason long context became practical.
- **Efficient Memory Management for LLM Serving with PagedAttention** — Kwon et al., 2023. arXiv:2309.06180. The vLLM paper; explains what continuous batching buys.
- **GPTQ** — Frantar et al., 2022. arXiv:2210.17323. Post-training quantisation, one layer at a time.
- **llama.cpp** — `github.com/ggml-org/llama.cpp`. Read `ggml-quants.c` if you want to see what a quantised dot product actually costs.
- **Hugging Face `transformers` docs** — the reference for Path A, and the only place with authoritative chat-template documentation.

For this repository's own material: [architecture.md](architecture.md) for the design decisions, [01-pull.md](01-pull.md) → [03-serve.md](03-serve.md) for the pipeline, and `shilp-book.docx` for the 20-chapter treatment.
