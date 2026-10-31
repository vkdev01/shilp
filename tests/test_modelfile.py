"""Shilpfile parsing: the format is small, so the tests can be exhaustive."""

from __future__ import annotations

import pytest

from shilp.modelfile import ShilpfileError, parse

TRIPLE = '"' * 3

FULL = f"""
# a comment at the top
FROM hf:Qwen/Qwen2.5-1.5B-Instruct-GGUF:q4_k_m
PARAMETER temperature 0.6
PARAMETER num_ctx 8192
PARAMETER stop "<|im_end|>"
PARAMETER stop "<|endoftext|>"
SYSTEM {TRIPLE}You are terse.{TRIPLE}
TEMPLATE {TRIPLE}<|im_start|>system
{{{{ .System }}}}<|im_end|>
<|im_start|>user
{{{{ .Prompt }}}}<|im_end|>
<|im_start|>assistant
{TRIPLE}
"""


def test_parses_every_directive():
    spec = parse(FULL)
    assert spec.source == "hf:Qwen/Qwen2.5-1.5B-Instruct-GGUF:q4_k_m"
    assert spec.params["temperature"] == 0.6
    assert spec.params["num_ctx"] == 8192
    assert spec.params["stop"] == ["<|im_end|>", "<|endoftext|>"]
    assert spec.system == "You are terse."
    assert "{{ .System }}" in spec.template
    assert spec.comments == ["a comment at the top"]


def test_integers_stay_integers():
    spec = parse("FROM a:1\nPARAMETER num_predict 128\n")
    assert spec.params["num_predict"] == 128
    assert isinstance(spec.params["num_predict"], int)


def test_requires_from():
    with pytest.raises(ShilpfileError, match="FROM"):
        parse("PARAMETER temperature 0.1\n")


def test_unknown_directive_names_the_line():
    with pytest.raises(ShilpfileError, match="line 2"):
        parse("FROM a:1\nRUN apt-get install\n")


def test_unknown_parameter_is_rejected():
    with pytest.raises(ShilpfileError, match="unknown parameter"):
        parse("FROM a:1\nPARAMETER temprature 0.7\n")


def test_non_numeric_parameter_is_rejected():
    with pytest.raises(ShilpfileError, match="expects a number"):
        parse("FROM a:1\nPARAMETER temperature warm\n")


def test_quoted_value_with_spaces():
    spec = parse('FROM a:1\nPARAMETER stop "a long stop string"\n')
    assert spec.params["stop"] == ["a long stop string"]


def test_adapter_directive():
    spec = parse("FROM a:1\nADAPTER ./lora-out\n")
    assert spec.adapter == "./lora-out"
