'''Parse a `Shilpfile` — the text file that says what a model is.

It is deliberately close to Ollama's Modelfile, because that format got the
important part right: the model is described in one readable file, and building
is reproducible. A minimal one:

    FROM hf:Qwen/Qwen2.5-1.5B-Instruct-GGUF:q4_k_m
    PARAMETER temperature 0.6
    PARAMETER num_ctx 8192
    PARAMETER stop "<|im_end|>"
    SYSTEM <block>You are a terse assistant.<block>
    TEMPLATE <block><|im_start|>system
    {{ .System }}<|im_end|>
    <|im_start|>user
    {{ .Prompt }}<|im_end|>
    <|im_start|>assistant
    <block>

(`<block>` stands for three double quotes.) Two value delimiters, both familiar:
a triple-quoted block spans lines, a single `"..."` is one line, and `#` starts a
comment when it is not inside a value.
'''

from __future__ import annotations

import shlex
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .utils import ShilpError

#: Parameters the engine understands. Anything else is rejected at build time
#: rather than silently ignored at inference time.
NUMERIC_PARAMS = {
    "temperature",
    "top_p",
    "top_k",
    "min_p",
    "repeat_penalty",
    "repeat_last_n",
    "num_ctx",
    "num_predict",
    "seed",
    "presence_penalty",
    "frequency_penalty",
}
LIST_PARAMS = {"stop"}
BOOL_PARAMS = {"stream"}
KNOWN_PARAMS = NUMERIC_PARAMS | LIST_PARAMS | BOOL_PARAMS

KNOWN_DIRECTIVES = {"FROM", "PARAMETER", "TEMPLATE", "SYSTEM", "ADAPTER", "COMMENT"}


class ShilpfileError(ShilpError):
    """The Shilpfile is malformed; the message names the offending line."""


@dataclass
class Shilpfile:
    """The parsed form of a Shilpfile."""

    #: `hf:org/repo`, `hf:org/repo:quant`, `name:tag`, or a local path.
    source: str
    params: dict[str, Any] = field(default_factory=dict)
    template: str | None = None
    system: str | None = None
    #: Path to a LoRA adapter to merge in, relative to the Shilpfile.
    adapter: str | None = None
    #: `#` lines, kept so `shilp show` can echo the author's notes.
    comments: list[str] = field(default_factory=list)


def _coerce(key: str, raw: str) -> Any:
    """Turn a parameter's text value into the type the engine expects."""
    if key in LIST_PARAMS:
        return raw
    if key in BOOL_PARAMS:
        low = raw.lower()
        if low not in {"true", "false"}:
            raise ValueError(f"{key} expects true/false, got {raw!r}")
        return low == "true"
    try:
        return float(raw) if "." in raw else int(raw)
    except ValueError as exc:
        raise ValueError(f"{key} expects a number, got {raw!r}") from exc


def _read_block(text: str, start: int) -> tuple[str, int]:
    """Read a value at `start`; returns (value, index just past it).

    Accepts a triple-quoted block, a single-quoted string, or a bare token.
    """
    triple = '"' * 3
    if text.startswith(triple, start):
        end = text.find(triple, start + 3)
        if end == -1:
            raise ShilpfileError("unterminated triple-quoted block")
        return text[start + 3 : end], end + 3
    if start < len(text) and text[start] == '"':
        end = start + 1
        while end < len(text):
            if text[end] == "\\":
                end += 2
                continue
            if text[end] == '"':
                return text[start + 1 : end], end + 1
            end += 1
        raise ShilpfileError("unterminated quoted value")
    end = start
    while end < len(text) and not text[end].isspace():
        end += 1
    return text[start:end], end


def parse(text: str) -> Shilpfile:
    """Parse Shilpfile source text into a `Shilpfile`."""
    source: str | None = None
    params: dict[str, Any] = {}
    stops: list[str] = []
    template: str | None = None
    system: str | None = None
    adapter: str | None = None
    comments: list[str] = []

    pos = 0
    line_no = 0
    n = len(text)

    while pos < n:
        line_no += 1
        while pos < n and text[pos] in " \t\r":
            pos += 1
        if pos >= n:
            break
        if text[pos] == "\n":
            pos += 1
            continue
        if text[pos] == "#":
            end = text.find("\n", pos)
            end = n if end == -1 else end
            comments.append(text[pos + 1 : end].strip())
            pos = end
            continue

        word_end = pos
        while word_end < n and not text[word_end].isspace():
            word_end += 1
        raw_directive = text[pos:word_end]
        directive = raw_directive.upper()
        if directive not in KNOWN_DIRECTIVES:
            raise ShilpfileError(
                f"line {line_no}: unknown directive {raw_directive!r} "
                f"(known: {', '.join(sorted(KNOWN_DIRECTIVES))})"
            )
        pos = word_end
        while pos < n and text[pos] in " \t":
            pos += 1

        if directive in {"FROM", "TEMPLATE", "SYSTEM", "ADAPTER"}:
            if pos >= n:
                raise ShilpfileError(f"line {line_no}: {directive} is missing a value")
            value, pos = _read_block(text, pos)
            if directive == "FROM":
                source = value.strip()
            elif directive == "TEMPLATE":
                template = value
            elif directive == "SYSTEM":
                system = value
            else:
                adapter = value.strip()
        elif directive == "PARAMETER":
            rest_end = text.find("\n", pos)
            rest_end = n if rest_end == -1 else rest_end
            try:
                parts = shlex.split(text[pos:rest_end], comments=True)
            except ValueError as exc:
                raise ShilpfileError(f"line {line_no}: {exc}") from exc
            if len(parts) != 2:
                raise ShilpfileError(
                    f"line {line_no}: PARAMETER expects '<name> <value>', "
                    f"got {text[pos:rest_end].strip()!r}"
                )
            key, raw = parts
            if key not in KNOWN_PARAMS:
                raise ShilpfileError(
                    f"line {line_no}: unknown parameter {key!r} "
                    f"(known: {', '.join(sorted(KNOWN_PARAMS))})"
                )
            if key in LIST_PARAMS:
                stops.append(raw)  # repeated PARAMETER stop lines accumulate
            else:
                try:
                    params[key] = _coerce(key, raw)
                except ValueError as exc:
                    raise ShilpfileError(f"line {line_no}: {exc}") from exc
            pos = rest_end
        else:  # COMMENT — reserved, and a no-op for now
            end = text.find("\n", pos)
            pos = n if end == -1 else end

        while pos < n and text[pos] in " \t\r":
            pos += 1
        if pos < n and text[pos] == "\n":
            pos += 1

    if not source:
        raise ShilpfileError("a Shilpfile must have a FROM line")
    if stops:
        params["stop"] = stops
    return Shilpfile(
        source=source,
        params=params,
        template=template,
        system=system,
        adapter=adapter,
        comments=comments,
    )


def parse_file(path: Path) -> Shilpfile:
    return parse(Path(path).read_text(encoding="utf-8"))
