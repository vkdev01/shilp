"""Structural edits to a model's weights and config — the "internal changes" step.

These operate on a Hugging Face directory (safetensors + config.json), because
that is the only place the change can be made cleanly. A GGUF is a compiled
artifact: you convert to GGUF *after* surgery, not before.

What is here and what it costs:

    prune_layers      drop the last N transformer blocks. Cuts memory and time
                      roughly in proportion, and costs real quality — the last
                      layers do a lot of work. Fine-tune after pruning, or don't
                      bother.
    truncate_vocab    keep the first N embedding rows. Useful when you have
                      retrained a tokenizer; destructive otherwise.
    set_rope_scaling  extend context by rescaling RoPE positions. Config-only,
                      no tensors touched. Quality holds near the original
                      training length and degrades past it.
    write_chat_template  set the tokenizer's chat template. No weight change,
                      usually the highest-value "modification" of all, because
                      most bad answers from a local model are a wrong template.

Every function copies the source directory and rewrites only what it must, so
the original stays intact and the change is reversible by deleting the output.
"""

from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from ..utils import ShilpError

#: Block-name conventions across architectures. Add a row, not a special case.
LAYER_PATTERNS = (
    r"model\.layers\.(\d+)\.",
    r"transformer\.h\.(\d+)\.",
    r"model\.decoder\.layers\.(\d+)\.",
    r"language_model\.model\.layers\.(\d+)\.",
)

EMBEDDING_PATTERNS = (
    r"(^|\.)embed_tokens\.weight$",
    r"(^|\.)wte\.weight$",
    r"(^|\.)word_embeddings\.weight$",
)
LM_HEAD_PATTERNS = (r"(^|\.)lm_head\.weight$", r"(^|\.)embed_out\.weight$")


class SurgeryError(ShilpError):
    """The requested edit does not apply to this model."""


@dataclass
class SurgeryReport:
    """What changed, in terms a human can check."""

    operation: str
    source: Path
    output: Path
    detail: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def to_recipe_step(self) -> dict[str, Any]:
        return {"step": self.operation, **self.detail, "warnings": self.warnings}


# --- small helpers -----------------------------------------------------
def _layer_index(key: str) -> int | None:
    for pattern in LAYER_PATTERNS:
        match = re.search(pattern, key)
        if match:
            return int(match.group(1))
    return None


def _matches_any(key: str, patterns: tuple[str, ...]) -> bool:
    return any(re.search(p, key) for p in patterns)


def read_config(model_dir: Path) -> dict[str, Any]:
    path = Path(model_dir) / "config.json"
    if not path.is_file():
        raise SurgeryError(f"{model_dir} has no config.json")
    return json.loads(path.read_text(encoding="utf-8"))


def _write_config(model_dir: Path, config: dict[str, Any]) -> None:
    (Path(model_dir) / "config.json").write_text(
        json.dumps(config, indent=2) + "\n", encoding="utf-8"
    )


def _shards(model_dir: Path) -> list[Path]:
    """Weight files, sharded or single, in a stable order."""
    files = sorted(Path(model_dir).glob("*.safetensors"))
    if not files:
        raise SurgeryError(f"{model_dir} has no .safetensors files")
    return files


def _rewrite_shards(
    model_dir: Path,
    transform: Callable[[str, Any], Any | None],
) -> dict[str, int]:
    """Apply `transform(key, tensor)` to every tensor; None means drop it.

    One shard is held in memory at a time. A 7B model in two shards therefore
    needs ~7 GB of RAM for the largest shard, not ~15 GB for the whole model —
    worth knowing before you run this on a laptop.
    """
    try:
        import numpy as np
        from safetensors import safe_open
        from safetensors.numpy import save_file
    except ModuleNotFoundError as exc:  # pragma: no cover - install dependent
        raise SurgeryError(
            "safetensors and numpy are required for surgery: pip install safetensors numpy"
        ) from exc

    stats = {"kept": 0, "dropped": 0, "rewritten": 0}
    weight_map: dict[str, str] = {}
    index_path = model_dir / "model.safetensors.index.json"
    index = json.loads(index_path.read_text(encoding="utf-8")) if index_path.is_file() else None

    for shard in _shards(model_dir):
        tensors: dict[str, Any] = {}
        with safe_open(str(shard), framework="numpy") as handle:
            for key in handle.keys():
                original = handle.get_tensor(key)
                out = transform(key, original)
                if out is None:
                    stats["dropped"] += 1
                    continue
                if out.shape != original.shape:
                    stats["rewritten"] += 1
                tensors[key] = np.ascontiguousarray(out)
                stats["kept"] += 1
        if not tensors:
            shard.unlink()  # every tensor in this shard was dropped
            continue
        save_file(tensors, str(shard), metadata={"format": "pt"})
        del tensors
        weight_map.update({k: shard.name for k in _keys_of(shard)})

    if index is not None:
        # Keep only entries whose tensors survived, pointing at the same shard.
        index["weight_map"] = {
            k: v for k, v in index["weight_map"].items() if v in weight_map
        }
        index["metadata"] = {
            **index.get("metadata", {}),
            "total_size": sum(p.stat().st_size for p in _shards(model_dir)),
        }
        index_path.write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")
    return stats


def _keys_of(shard: Path) -> list[str]:
    from safetensors import safe_open

    with safe_open(str(shard), framework="numpy") as handle:
        return list(handle.keys())


def _copy_model_dir(source: Path, output: Path) -> None:
    source, output = Path(source), Path(output)
    if not source.is_dir():
        raise SurgeryError(f"no such model directory: {source}")
    if output.exists():
        raise SurgeryError(f"{output} already exists; delete it or pick another path")
    shutil.copytree(source, output)


# --- the operations ----------------------------------------------------
def prune_layers(
    source: Path,
    output: Path,
    *,
    keep_layers: int,
) -> SurgeryReport:
    """Drop every transformer block at index >= `keep_layers`.

    Keeping 24 of 28 layers removes ~14% of the weights and usually costs more
    quality than a step down in quantization. Measure before you ship it.
    """
    config = read_config(source)
    total = config.get("num_hidden_layers")
    if not isinstance(total, int):
        raise SurgeryError("config.json has no num_hidden_layers; cannot prune")
    if not 1 <= keep_layers < total:
        raise SurgeryError(f"keep_layers must be between 1 and {total - 1}, got {keep_layers}")

    _copy_model_dir(source, output)

    def transform(key: str, tensor: Any) -> Any | None:
        index = _layer_index(key)
        if index is not None and index >= keep_layers:
            return None
        return tensor

    stats = _rewrite_shards(Path(output), transform)
    config["num_hidden_layers"] = keep_layers
    _write_config(Path(output), config)

    return SurgeryReport(
        operation="prune_layers",
        source=Path(source),
        output=Path(output),
        detail={
            "layers_before": total,
            "layers_after": keep_layers,
            "tensors_dropped": stats["dropped"],
        },
        warnings=[
            "Depth pruning removes capability, not just size. "
            "Re-evaluate before serving this as a general-purpose model."
        ],
    )


def truncate_vocab(
    source: Path,
    output: Path,
    *,
    new_vocab_size: int,
) -> SurgeryReport:
    """Keep the first `new_vocab_size` rows of the embedding and LM head.

    Only correct when the tokenizer really has that many tokens. It is the right
    move after retraining a tokenizer on a narrow domain and the wrong move as a
    shortcut to a smaller file.
    """
    config = read_config(source)
    current = config.get("vocab_size")
    if not isinstance(current, int):
        raise SurgeryError("config.json has no vocab_size")
    if not 1 <= new_vocab_size < current:
        raise SurgeryError(f"new_vocab_size must be < {current}, got {new_vocab_size}")

    _copy_model_dir(source, output)

    def transform(key: str, tensor: Any) -> Any | None:
        if _matches_any(key, EMBEDDING_PATTERNS + LM_HEAD_PATTERNS):
            if tensor.shape[0] < new_vocab_size:
                raise SurgeryError(
                    f"{key} has {tensor.shape[0]} rows, fewer than {new_vocab_size}"
                )
            return tensor[:new_vocab_size]
        return tensor

    stats = _rewrite_shards(Path(output), transform)
    config["vocab_size"] = new_vocab_size
    _write_config(Path(output), config)

    return SurgeryReport(
        operation="truncate_vocab",
        source=Path(source),
        output=Path(output),
        detail={
            "vocab_before": current,
            "vocab_after": new_vocab_size,
            "tensors_rewritten": stats["rewritten"],
        },
        warnings=[
            "The tokenizer must be trimmed to match, or ids will decode to the "
            "wrong text. Update tokenizer.json before serving."
        ],
    )


def set_rope_scaling(
    source: Path,
    output: Path,
    *,
    factor: float,
    rope_type: str = "linear",
) -> SurgeryReport:
    """Extend the usable context by rescaling RoPE positions. Config-only."""
    if factor <= 1.0:
        raise SurgeryError(f"factor must be > 1.0, got {factor}")
    config = read_config(source)
    _copy_model_dir(source, output)
    config["rope_scaling"] = {"type": rope_type, "factor": float(factor)}
    _write_config(Path(output), config)
    return SurgeryReport(
        operation="set_rope_scaling",
        source=Path(source),
        output=Path(output),
        detail={"rope_type": rope_type, "factor": factor},
        warnings=[
            "Beyond ~1.5x the training length, quality degrades no matter what "
            "the config claims. Test with a long-context probe, not a vibe check."
        ],
    )


def write_chat_template(source: Path, output: Path, *, template: str) -> SurgeryReport:
    """Set `chat_template` in tokenizer_config.json (Jinja, HF dialect)."""
    _copy_model_dir(source, output)
    path = Path(output) / "tokenizer_config.json"
    if not path.is_file():
        raise SurgeryError(f"{output} has no tokenizer_config.json")
    config = json.loads(path.read_text(encoding="utf-8"))
    previous = config.get("chat_template")
    config["chat_template"] = template
    path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    return SurgeryReport(
        operation="write_chat_template",
        source=Path(source),
        output=Path(output),
        detail={"had_template": previous is not None, "template_chars": len(template)},
        warnings=[],
    )
