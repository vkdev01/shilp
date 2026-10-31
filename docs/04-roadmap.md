# Roadmap — what is missing, and why

A roadmap is only useful if it says what is *not* there. Each item below is a real gap in the current code, with the reason it was left and the shape the fix would take.

---

## 1. A real evaluation harness

**Gap.** `surgery prune` and `quantize` change quality and shilp cannot measure the change. Right now the honest advice is "test it yourself", which is not engineering.

**Fix.** `shilp eval <ref> --suite gsm8k,lite --n 200`, reporting accuracy per task against the parent model, stored in the recipe. A small suite (50–200 items) that runs in minutes on a laptop would be enough to catch a bad prune or an over-aggressive quant. This is the highest-value item on the list, because it turns every other lossy operation from "trust me" into "measured".

## 2. Continuous batching, or a vLLM backend

**Gap.** One model, one request at a time. On a laptop that is correct. On a CUDA box it wastes most of the GPU.

**Fix.** Two options, and the engine protocol was shaped for the second:

- Implement `Engine` for vLLM, which brings continuous batching and paged attention for free. Roughly one file, plus a `gpu` extra in `pyproject.toml`.
- Add request queueing with a micro-batch window inside the loader, which helps llama.cpp but cannot match paged attention.

## 3. Cancellation

**Gap.** A client that disconnects mid-stream leaves the worker thread finishing a generation nobody reads. One generation of wasted compute per abandoned request.

**Fix.** llama.cpp exposes no cancellation callback through the Python bindings, so the practical approach is a cooperative flag: the pump loop checks an `asyncio.Event` between chunks and breaks. That works because chunks arrive per token; the cost is up to one token of overrun.

## 4. Tool calling

**Gap.** No `tool_calls` parsing on either surface. A model that emits a JSON tool call returns text.

**Fix.** Parse the model's tool-call syntax (per-family: Hermes JSON, Llama's `<|python_tag|>`, Qwen's `<tool_call>`) into OpenAI's `tool_calls` shape, and add a `tools` passthrough to the template renderer. Needs a per-architecture registry, which is why it is not a five-line change.

## 5. Sharded GGUFs

**Gap.** `select_files` rejects multi-part GGUF repos with a clear message. Larger models increasingly ship that way.

**Fix.** Download all shards, verify the split count and index, store them as separate layers of one manifest, and hand llama.cpp the first path (it resolves siblings itself).

## 6. Layer surgery on GGUF directly

**Gap.** Surgery happens on safetensors, which requires the float model — 3 GB for 1.5B, 15 GB for 7B. You cannot prune a model you could otherwise run.

**Fix.** GGUF is a documented container: a header, a metadata KV block, tensor info, then tensor data. Dropping layers means rewriting the tensor-info table and the offsets, and decrementing `block_count`. Doable and fiddly; worth doing after the eval harness exists so the result can be measured.

## 7. Authentication and multi-tenancy

**Gap.** No auth, binds to localhost. Deliberate, but it means the thing cannot be shared.

**Fix.** An optional bearer token (`SHILP_API_KEY`) checked by a dependency, plus per-key usage rows so `shilp usage --key` works. Small; skipped only because a single-user local server does not need it and half-built auth is worse than none.

## 8. A Docker image and a `compose` file

**Gap.** No container. `pip install` plus a llama.cpp build is two steps too many for a demo.

**Fix.** Multi-stage build: compile llama.cpp in one stage, copy the binary and `convert_hf_to_gguf.py` into a slim runtime, mount `SHILP_HOME` as a volume so models survive the container. Apple Silicon users should still run natively — Metal does not pass through to Docker.

## 9. A fine-tuning pipeline that closes the loop

**Gap.** `merge-lora` exists but nothing produces an adapter. The pipeline is download → reshape → serve, with the training step assumed to happen elsewhere.

**Fix.** `shilp finetune --base <ref> --data <jsonl> --method lora|grpo`, emitting an adapter that `merge-lora` consumes. This is the natural next project, and it connects to the reasoning-model work: GRPO on a small model, evaluated by the harness in item 1, served by the engine that already exists. That would make the four steps a closed loop: *pull → tune → quantize → serve → measure → tune again*.

## 10. Small things

- `shilp pull --resume` reporting byte progress through a callback instead of three coarse phases.
- A `--json` flag on every CLI command, for scripting.
- Speculative decoding (a small draft model plus a large target), which llama.cpp supports and the loader could own.
- An OpenAI-compatible `/v1/audio/*` is out of scope; that is a different kind of model.

---

## What I would do next, in order

1. **Evaluation harness.** Everything lossy becomes defensible.
2. **Cancellation.** Cheap, removes a real waste.
3. **vLLM backend.** Proves the engine abstraction was worth having.
4. **Fine-tuning pipeline.** Turns four steps into a loop.
5. **Docker.** Distribution, not engineering — do it when someone else needs to run this.
