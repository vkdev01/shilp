"""Merge a LoRA adapter back into the base weights.

Why merge instead of serving the adapter separately: a merged model is one file
that any GGUF toolchain understands, with zero inference-time overhead. The cost
is that you cannot switch adapters per request, and you need the base model in
float precision once, which is the part a laptop usually cannot do for 7B+.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..utils import MissingDependency, ShilpError


class MergeError(ShilpError):
    """The adapter does not fit the base model, or a dependency is missing."""


@dataclass(frozen=True)
class MergeReport:
    base: Path
    adapter: Path
    output: Path
    dtype: str


def merge_lora(
    base_dir: Path,
    adapter_dir: Path,
    output_dir: Path,
    *,
    dtype: str = "float16",
) -> MergeReport:
    """Load base + adapter, fold the adapter in, save a plain HF model dir.

    Requires the `finetune` extra: torch, transformers, peft.
    """
    try:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ModuleNotFoundError as exc:  # pragma: no cover - install dependent
        raise MissingDependency(
            "LoRA merging needs torch, transformers and peft: pip install -e '.[finetune]'"
        ) from exc

    torch_dtype = getattr(torch, dtype, None)
    if torch_dtype is None:
        raise MergeError(f"unknown dtype {dtype!r}; try float16 or bfloat16")

    base_dir, adapter_dir, output_dir = Path(base_dir), Path(adapter_dir), Path(output_dir)
    if output_dir.exists():
        raise MergeError(f"{output_dir} already exists; pick another path")

    try:
        model = AutoModelForCausalLM.from_pretrained(
            str(base_dir), torch_dtype=torch_dtype, low_cpu_mem_usage=True
        )
        model = PeftModel.from_pretrained(model, str(adapter_dir))
        # merge_and_unload folds the low-rank update into the base matrices and
        # returns a plain model with no PEFT wrappers left behind.
        model = model.merge_and_unload()
    except Exception as exc:
        raise MergeError(f"merge failed: {exc}") from exc

    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(output_dir), safe_serialization=True)
    try:
        AutoTokenizer.from_pretrained(str(adapter_dir)).save_pretrained(str(output_dir))
    except Exception:
        # Many adapters ship no tokenizer; the base one is the right fallback.
        AutoTokenizer.from_pretrained(str(base_dir)).save_pretrained(str(output_dir))

    return MergeReport(base=base_dir, adapter=adapter_dir, output=output_dir, dtype=dtype)
