"""Step 2 of the pipeline: change the model.

Nothing here is magic. Each module wraps one operation, records what it did, and
returns a report so the change shows up in the model's recipe.

    convert.py   HF safetensors -> GGUF          (format change, lossless)
    quantize.py  GGUF F16 -> GGUF Q4_K_M         (precision change, lossy)
    surgery.py   prune layers, resize vocab, ... (structure change, lossy)
    merge.py     LoRA adapter -> merged weights  (behaviour change)

The honest ordering, if you intend to fine-tune: merge LoRA *first*, then
convert, then quantize. Quantizing before merging bakes rounding error into the
adapter's target weights.
"""

from .convert import convert_to_gguf
from .merge import merge_lora
from .quantize import QUANT_TYPES, quantize
from .surgery import prune_layers, set_rope_scaling, truncate_vocab, write_chat_template

__all__ = [
    "convert_to_gguf",
    "quantize",
    "QUANT_TYPES",
    "prune_layers",
    "truncate_vocab",
    "set_rope_scaling",
    "write_chat_template",
    "merge_lora",
]
