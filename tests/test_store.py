"""Store and utility behaviour: refs, digests, dedupe, manifests, gc."""

from __future__ import annotations

import json

import pytest

from shilp.store import MT_CONFIG, MT_GGUF, Manifest, Store, StoreError
from shilp.utils import RefError, human_size, parse_ref


def test_parse_ref_defaults_to_latest():
    assert parse_ref("qwen") == ("qwen", "latest")
    assert parse_ref("qwen:q4") == ("qwen", "q4")
    assert parse_ref("team/model:v2") == ("team/model", "v2")


def test_parse_ref_rejects_hub_refs():
    with pytest.raises(RefError):
        parse_ref("hf:Qwen/Qwen2.5-1.5B-Instruct")


def test_human_size():
    assert human_size(512) == "512 B"
    assert human_size(1536).endswith("KB")
    assert human_size(4 * 1024**3) == "4.0 GB"


def test_put_blob_dedupes_identical_files(store: Store, tmp_path):
    first = tmp_path / "a.gguf"
    first.write_bytes(b"same bytes")
    second = tmp_path / "b.gguf"
    second.write_bytes(b"same bytes")

    layer_a = store.put_blob(first)
    layer_b = store.put_blob(second)

    assert layer_a.digest == layer_b.digest
    assert len(list(store.blobs.iterdir())) == 1
    assert layer_a.name == "a.gguf"  # the first name is kept


def test_manifest_roundtrip(store: Store, tmp_path):
    weights = tmp_path / "w.gguf"
    weights.write_bytes(b"GGUF")
    layer = store.put_blob(weights)
    manifest = Manifest(
        name="tiny",
        tag="q4",
        layers=[layer],
        params={"temperature": 0.2, "stop": ["</s>"]},
        metadata={"source": "hf:org/repo"},
    )
    store.save_manifest(manifest)

    loaded = store.resolve("tiny:q4")
    assert loaded.params["temperature"] == 0.2
    assert loaded.layers[0].digest == layer.digest
    assert loaded.human_size.endswith("B")
    assert loaded.ref == "tiny:q4"


def test_missing_model_is_a_clear_error(store: Store):
    with pytest.raises(StoreError, match="not found"):
        store.resolve("nope:latest")


def test_list_is_newest_first(store: Store, tmp_path):
    weights = tmp_path / "w.gguf"
    weights.write_bytes(b"GGUF")
    layer = store.put_blob(weights)
    store.save_manifest(Manifest(name="older", created="2024-01-01T00:00:00Z", layers=[layer]))
    store.save_manifest(Manifest(name="newer", created="2025-01-01T00:00:00Z", layers=[layer]))
    assert [m.name for m in store.list()] == ["newer", "older"]


def test_gc_keeps_referenced_blobs(store: Store, tmp_path):
    used = tmp_path / "used.gguf"
    used.write_bytes(b"used")
    orphan = tmp_path / "orphan.gguf"
    orphan.write_bytes(b"orphan")

    store.put_blob(used)
    store.save_manifest(Manifest(name="keeper", layers=[store.put_blob(used)]))
    store.put_blob(orphan)

    orphans = store.gc(dry_run=True)
    assert len(orphans) == 1
    store.gc(dry_run=False)
    assert len(list(store.blobs.iterdir())) == 1


def test_delete_removes_manifest_and_blob(store: Store, tmp_path):
    weights = tmp_path / "w.gguf"
    weights.write_bytes(b"GGUF")
    store.save_manifest(Manifest(name="gone", layers=[store.put_blob(weights)]))
    store.delete("gone")
    assert store.list() == []
    assert list(store.blobs.iterdir()) == []


def test_put_bytes_stores_a_config(store: Store):
    layer = store.put_bytes(b'{"a":1}', "config.json", MT_CONFIG)
    assert store.read_blob_text(layer.digest) == '{"a":1}'
    assert json.loads(store.read_blob_text(layer.digest))["a"] == 1


def test_weights_layer_is_found_by_media_type(store: Store):
    weights = store.put_bytes(b"GGUF", "m.gguf", MT_GGUF)
    manifest = Manifest(name="m", layers=[store.put_bytes(b"{}", "c.json", MT_CONFIG), weights])
    assert manifest.layer_of(MT_GGUF).digest == weights.digest
