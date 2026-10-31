"""Render an Ollama-style template to a prompt string.

Ollama's Modelfile templates are Go templates with three variables:

    {{ .System }}    the system prompt
    {{ .Prompt }}    the user's turn
    {{ .Response }}  where the model starts writing (usually empty)

Supporting exactly those three is enough to copy a template from a model card
and get the same formatting, and it keeps the implementation honest: anything
fancier belongs in a Jinja template inside the GGUF, which llama.cpp handles.
"""

from __future__ import annotations

import re

#: `{{ .System }}`, `{{.System}}`, `{{- .System -}}` — whitespace trim markers
#: are accepted and ignored, because templates copied from model cards use them.
_VAR_RE = re.compile(r"\{\{-?\s*\.(System|Prompt|Response)\s*-?\}\}")


class TemplateError(ValueError):
    """The template refers to something this renderer does not provide."""


def render(template: str, *, system: str = "", prompt: str, response: str = "") -> str:
    """Substitute the three variables. Unknown `{{ .X }}` names are an error."""
    values = {"System": system, "Prompt": prompt, "Response": response}

    def substitute(match: re.Match[str]) -> str:
        return values[match.group(1)]

    rendered = _VAR_RE.sub(substitute, template)

    leftover = re.search(r"\{\{[^}]*\}\}", rendered)
    if leftover:
        raise TemplateError(
            f"unsupported template expression {leftover.group(0)!r}; "
            "shilp supports {{ .System }}, {{ .Prompt }} and {{ .Response }}"
        )
    return rendered
