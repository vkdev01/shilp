"""GGUF precision reduction — the step that makes a model fit on your machine.

A 7B model is ~15 GB at F16 and ~4.4 GB at Q4_K_M. Quantization is the reason a
laptop can serve anything at all, and the reason quality is no longer a
foregone conclusion: the K-quants keep per-block scales, so the loss is small
enough that most people cannot tell Q4_K_M from F16 in conversation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .toolchain import Toolchain, run

#: The quant types worth knowing, with the trade-off stated.
#: Anything `llama-quantize` accepts works; these are the sensible defaults.
QUANT_TYPES: dict[str, str] = {
    "Q2_K": "smallest that is still coherent; noticeably dumber",
    "Q3_K_M": "fits a 13B on 8 GB; quality drops on code and maths",
    "Q4_K_M": "the default: best size/quality balance",
    "Q5_K_M": "near-F16 quality, ~20% larger than Q4_K_M",
    "Q6_K": "essentially lossless, ~35% larger than Q4_K_M",
    "Q8_0": "reference quality, barely worth it over Q6_K",
    "F16": "no quantization; convert already produces this",
}

DEFAULT_QUANT = "Q4_K_M"


@dataclass(frozen=True)
class QuantizeReport:
    source: Path
    output: Path
    quant_type: str
    size_bytes: int
    #: Rough resident-memory estimate: file size plus KV cache and overhead.
    est_resident_bytes: int


def quantize(
    source: Path,
    output: Path,
    *,
    quant_type: str = DEFAULT_QUANT,
    toolchain: Toolchain | None = None,
    num_threads: int | None = None,
) -> QuantizeReport:
    """Quantize one GGUF file into another. Source is left untouched."""
    source = Path(source)
    if not source.is_file():
        raise FileNotFoundError(f"no such GGUF file: {source}")
    chain = toolchain or Toolchain.discover()
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    cmd = [str(chain.binary("llama-quantize"))]
    if num_threads:
        cmd += ["--threads", str(num_threads)]
    cmd += [str(source), str(output), quant_type]
    run(cmd)

    size = output.stat().st_size
    return QuantizeReport(
        source=source,
        output=output,
        quant_type=quant_type,
        size_bytes=size,
        # ~15% headroom for the KV cache at 4k context plus runtime buffers.
        est_resident_bytes=int(size * 1.15),
    )
