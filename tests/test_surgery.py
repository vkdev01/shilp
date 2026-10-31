"""Surgery: each edit changes the tensors or the config it claims to, and nothing else."""

from __future__ import annotations

import json

import pytest

from shilp.modify.surgery import (
    SurgeryError,
    prune_layers,
    read_config,
    set_rope_scaling,
    truncate_vocab,
    write_chat_template,
)


def _keys(model_dir):
    from safetensors import safe_open

    with safe_open(str(model_dir / "model.safetensors"), framework="numpy") as handle:
        return list(handle.keys())


def test_prune_drops_the_requested_layers(tiny_hf_model, tmp_path):
    out = tmp_path / "pruned"
    report = prune_layers(tiny_hf_model, out, keep_layers=1)

    keys = _keys(out)
    assert not any(".layers.1." in k for k in keys)
    assert any(".layers.0." in k for k in keys)
    assert read_config(out)["num_hidden_layers"] == 1
    assert report.detail["tensors_dropped"] == 1
    assert report.warnings  # quality warning is part of the contract


def test_prune_leaves_the_source_untouched(tiny_hf_model, tmp_path):
    before = sorted(_keys(tiny_hf_model))
    prune_layers(tiny_hf_model, tmp_path / "pruned", keep_layers=1)
    assert sorted(_keys(tiny_hf_model)) == before
    assert read_config(tiny_hf_model)["num_hidden_layers"] == 2


def test_prune_rejects_impossible_layer_counts(tiny_hf_model, tmp_path):
    with pytest.raises(SurgeryError, match="keep_layers"):
        prune_layers(tiny_hf_model, tmp_path / "x", keep_layers=5)


def test_truncate_vocab_slices_both_matrices(tiny_hf_model, tmp_path):
    out = tmp_path / "small-vocab"
    report = truncate_vocab(tiny_hf_model, out, new_vocab_size=6)

    from safetensors import safe_open

    with safe_open(str(out / "model.safetensors"), framework="numpy") as handle:
        assert handle.get_tensor("model.embed_tokens.weight").shape[0] == 6
        assert handle.get_tensor("lm_head.weight").shape[0] == 6
    assert read_config(out)["vocab_size"] == 6
    assert report.detail["tensors_rewritten"] == 2


def test_rope_scaling_is_config_only(tiny_hf_model, tmp_path):
    out = tmp_path / "rope"
    set_rope_scaling(tiny_hf_model, out, factor=2.0)
    config = read_config(out)
    assert config["rope_scaling"] == {"type": "linear", "factor": 2.0}
    assert sorted(_keys(out)) == sorted(_keys(tiny_hf_model))


def test_rope_rejects_a_shrinking_factor(tiny_hf_model, tmp_path):
    with pytest.raises(SurgeryError, match="factor"):
        set_rope_scaling(tiny_hf_model, tmp_path / "x", factor=0.5)


def test_chat_template_is_written(tiny_hf_model, tmp_path):
    out = tmp_path / "templated"
    write_chat_template(tiny_hf_model, out, template="<|u|>{{ .Prompt }}<|a|>")
    config = json.loads((out / "tokenizer_config.json").read_text())
    assert config["chat_template"] == "<|u|>{{ .Prompt }}<|a|>"
    assert config["model_max_length"] == 2048  # everything else is preserved


def test_refuses_to_overwrite_an_existing_directory(tiny_hf_model, tmp_path):
    target = tmp_path / "exists"
    target.mkdir()
    with pytest.raises(SurgeryError, match="already exists"):
        prune_layers(tiny_hf_model, target, keep_layers=1)
