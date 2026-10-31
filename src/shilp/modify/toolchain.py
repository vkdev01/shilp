"""Shell out to the tools that already do this well.

`convert_hf_to_gguf.py` and `llama-quantize` live in a llama.cpp checkout. We do
not reimplement them: they track every new architecture, and a wrong GGUF header
is a debugging session nobody wants. What shilp adds is discovery (find the
checkout, explain what is missing), argument building, and provenance.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from ..utils import MissingDependency, ShilpError


class ToolError(ShilpError):
    """An external tool ran and failed; `stderr` holds its own explanation."""

    def __init__(self, message: str, stderr: str = ""):
        super().__init__(message)
        self.stderr = stderr


@dataclass(frozen=True)
class Toolchain:
    """Where llama.cpp's scripts and binaries are."""

    root: Path

    @classmethod
    def discover(cls, root: Path | None = None) -> "Toolchain":
        """Find a llama.cpp checkout: argument, then $SHILP_LLAMACPP_DIR, then PATH."""
        if root is None:
            env = os.environ.get("SHILP_LLAMACPP_DIR")
            root = Path(env).expanduser() if env else None
        if root is None:
            # A checkout that also has the converter is the useful one.
            for candidate in (Path.cwd(), Path.home() / "llama.cpp"):
                if (candidate / "convert_hf_to_gguf.py").is_file():
                    root = candidate
                    break
        if root is None:
            raise MissingDependency(
                "llama.cpp not found. Clone it and point shilp at it:\n"
                "  git clone https://github.com/ggml-org/llama.cpp ~/llama.cpp\n"
                "  export SHILP_LLAMACPP_DIR=~/llama.cpp\n"
                "(needed only for convert/quantize, not for serving a GGUF)"
            )
        if not Path(root).is_dir():
            raise MissingDependency(f"SHILP_LLAMACPP_DIR={root} is not a directory")
        return cls(root=Path(root))

    @property
    def convert_script(self) -> Path:
        path = self.root / "convert_hf_to_gguf.py"
        if not path.is_file():
            raise MissingDependency(f"{path} is missing; is {self.root} a llama.cpp checkout?")
        return path

    def binary(self, name: str) -> Path:
        """Locate a built binary, newest build layout first."""
        for rel in (f"build/bin/{name}", f"bin/{name}", name):
            candidate = self.root / rel
            if candidate.is_file() and os.access(candidate, os.X_OK):
                return candidate
        on_path = shutil.which(name)
        if on_path:
            return Path(on_path)
        raise MissingDependency(
            f"{name} not built. In {self.root} run:\n"
            "  cmake -B build && cmake --build build --config Release -j"
        )


def run(cmd: list[str], *, cwd: Path | None = None) -> None:
    """Run a command, streaming nothing but raising a readable error on failure."""
    try:
        proc = subprocess.run(
            cmd, cwd=cwd, capture_output=True, text=True, check=False
        )
    except FileNotFoundError as exc:
        raise MissingDependency(f"cannot execute {cmd[0]}: {exc}") from exc
    if proc.returncode != 0:
        tail = "\n".join((proc.stderr or proc.stdout or "").strip().splitlines()[-12:])
        raise ToolError(f"{Path(cmd[0]).name} failed (exit {proc.returncode})", tail)
