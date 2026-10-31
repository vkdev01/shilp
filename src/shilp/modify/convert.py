"""HF safetensors -> GGUF. A format change, so nothing is lost.

After this the model is a single file that llama.cpp can mmap. That is what
makes CPU inference reasonable at all: the OS page cache does the loading, and
resident memory is only the pages actually touched.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

from .toolchain import Toolchain, run

#: `f32` and `f16` are the only two the converter guarantees; `bf16` is the
#: smaller of the two and what most modern releases ship.
OUT_TYPES = ("f32", "f16", "bf16")


@dataclass(frozen=True)
class ConvertReport:
    source: Path
    output: Path
    out_type: str
    size_bytes: int


def convert_to_gguf(
    source_dir: Path,
    output: Path,
    *,
    out_type: str = "f16",
    toolchain: Toolchain | None = None,
) -> ConvertReport:
    """Convert a directory of HF safetensors into one GGUF file.

    `source_dir` must hold `config.json` plus the weights and tokenizer, which is
    exactly what `shilp pull hf:org/repo` leaves in the store.
    """
    if out_type not in OUT_TYPES:
        raise ValueError(f"out_type must be one of {OUT_TYPES}, got {out_type!r}")
    source_dir = Path(source_dir)
    if not (source_dir / "config.json").is_file():
        raise FileNotFoundError(f"{source_dir} has no config.json; is this an HF model dir?")

    chain = toolchain or Toolchain.discover()
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    # The converter writes `<model name>-<out_type>.gguf` next to --outfile's
    # directory, so point it at a scratch dir and rename afterwards.
    staging = output.parent / f".convert-{output.stem}"
    staging.mkdir(parents=True, exist_ok=True)
    run(
        [
            sys.executable,
            str(chain.convert_script),
            str(source_dir),
            "--outtype",
            out_type,
            "--outdir",
            str(staging),
        ],
        cwd=chain.root,
    )
    produced = sorted(staging.glob("*.gguf"))
    if not produced:
        raise FileNotFoundError("the converter produced no .gguf file")
    produced[0].replace(output)
    for leftover in staging.glob("*"):
        leftover.unlink(missing_ok=True)
    staging.rmdir()

    return ConvertReport(
        source=source_dir,
        output=output,
        out_type=out_type,
        size_bytes=output.stat().st_size,
    )
