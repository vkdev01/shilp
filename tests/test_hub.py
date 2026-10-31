"""Hub selection and import: which files, and what lands in the store."""

from __future__ import annotations

import pytest

from shilp.hub import HubError, HfSpec, HubFile, pull, preview, select_files
from shilp.store import MT_GGUF, Store


def test_spec_parses_quant():
    spec = HfSpec.parse("hf:bartowski/Qwen2.5-1.5B-Instruct-GGUF:Q4_K_M")
    assert spec.repo_id == "bartowski/Qwen2.5-1.5B-Instruct-GGUF"
    assert spec.quant == "q4_k_m"
    assert spec.is_gguf_repo


def test_spec_requires_owner():
    with pytest.raises(HubError, match="owner"):
        HfSpec.parse("hf:not-a-repo")


def test_selects_the_requested_quant():
    files = [
        HubFile("m-Q4_K_M.gguf", 100),
        HubFile("m-Q8_0.gguf", 200),
        HubFile("README.md", 10),
    ]
    spec = HfSpec.parse("hf:org/repo:q4_k_m")
    assert [f.name for f in select_files(files, spec)] == ["m-Q4_K_M.gguf"]


def test_missing_quant_lists_what_exists():
    files = [HubFile("m-Q8_0.gguf", 200)]
    spec = HfSpec.parse("hf:org/repo:q4_k_m")
    with pytest.raises(HubError, match="Q8_0"):
        select_files(files, spec)


def test_rejects_sharded_gguf_with_a_reason():
    files = [HubFile("m-Q4_K_M-00001-of-00002.gguf", 1), HubFile("m-Q4_K_M-00002-of-00002.gguf", 1)]
    spec = HfSpec.parse("hf:org/repo:q4_k_m")
    with pytest.raises(HubError, match="shards"):
        select_files(files, spec)


def test_safetensors_repo_picks_weights_and_config():
    files = [
        HubFile("model-00001-of-00002.safetensors", 100),
        HubFile("model.safetensors.index.json", 5),
        HubFile("config.json", 5),
        HubFile("pytorch_model.bin", 999),  # legacy, must be skipped
    ]
    spec = HfSpec.parse("hf:org/repo")
    names = [f.name for f in select_files(files, spec)]
    assert "pytorch_model.bin" not in names
    assert "model-00001-of-00002.safetensors" in names


def test_pull_registers_a_runnable_model(store: Store, fake_hub):
    spec = HfSpec.parse("hf:acme/tiny-GGUF:q4_k_m")
    manifest = pull(store, spec, hub=fake_hub)

    assert manifest.layer_of(MT_GGUF) is not None
    assert manifest.metadata["quantization"] == "q4_k_m"
    assert manifest.metadata["source"] == "hf:acme/tiny-GGUF"
    assert store.resolve(manifest.ref).size == manifest.size

    recipe = store.load_recipe(manifest.ref)
    assert recipe["steps"][0]["step"] == "pull"


def test_pull_marks_safetensors_as_needing_a_convert(store: Store):
    from tests.conftest import FakeHub

    hub = FakeHub(
        {
            "model.safetensors": b"weights",
            "config.json": b"{}",
        }
    )
    manifest = pull(store, HfSpec.parse("hf:acme/raw"), hub=hub)
    assert manifest.layer_of(MT_GGUF) is None
    assert "convert" in manifest.metadata["next_step"]


def test_preview_reports_size_without_writing(store: Store, fake_hub):
    info = preview(HfSpec.parse("hf:acme/tiny-GGUF:q4_k_m"), hub=fake_hub)
    assert info["total_bytes"] > 0
    assert store.list() == []
