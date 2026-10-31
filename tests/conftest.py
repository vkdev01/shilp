"""Shared fixtures: a throwaway store, a fake hub, and an app with no model."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from shilp.config import Settings, reset_settings_cache
from shilp.hub import HubFile, HfSpec
from shilp.store import Store


@pytest.fixture(autouse=True)
def isolated_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Every test gets its own SHILP_HOME, so nothing touches ~/.shilp."""
    home = tmp_path / "shilp-home"
    monkeypatch.setenv("SHILP_HOME", str(home))
    monkeypatch.setenv("SHILP_ENGINE", "fake")
    reset_settings_cache()
    yield home
    reset_settings_cache()


@pytest.fixture
def settings() -> Settings:
    return Settings.from_env()


@pytest.fixture
def store(settings: Settings) -> Store:
    return Store(settings.home)


class FakeHub:
    """Stands in for Hugging Face: writes the files a repo would contain."""

    def __init__(self, files: dict[str, bytes]):
        self.files = files
        self.calls: list[str] = []

    def list_files(self, spec: HfSpec, token: str | None = None) -> list[HubFile]:
        self.calls.append(spec.repo_id)
        return [HubFile(name=n, size=len(b)) for n, b in self.files.items()]

    def download(self, spec: HfSpec, patterns, local_dir: Path, token: str | None = None) -> Path:
        local_dir = Path(local_dir)
        local_dir.mkdir(parents=True, exist_ok=True)
        for name, data in self.files.items():
            if name in set(patterns):
                target = local_dir / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(data)
        return local_dir


@pytest.fixture
def fake_hub() -> FakeHub:
    """A GGUF repo: one weights file plus a config."""
    return FakeHub(
        {
            "model-Q4_K_M.gguf": b"GGUF\x00fake-weights" * 16,
            "config.json": json.dumps({"architecture": "qwen2"}).encode(),
            "README.md": b"# a model",
        }
    )


@pytest.fixture
def tiny_hf_model(tmp_path: Path) -> Path:
    """A two-layer HF model directory small enough to prune in a test."""
    np = pytest.importorskip("numpy")
    from safetensors.numpy import save_file

    model_dir = tmp_path / "tiny-model"
    model_dir.mkdir()
    tensors = {}
    for layer in range(2):
        tensors[f"model.layers.{layer}.self_attn.q_proj.weight"] = np.ones((4, 4), dtype=np.float32)
    tensors["model.embed_tokens.weight"] = np.arange(20, dtype=np.float32).reshape(10, 2)
    tensors["lm_head.weight"] = np.arange(20, dtype=np.float32).reshape(10, 2)
    tensors["model.norm.weight"] = np.ones(2, dtype=np.float32)
    save_file(tensors, str(model_dir / "model.safetensors"))

    (model_dir / "config.json").write_text(
        json.dumps({"num_hidden_layers": 2, "vocab_size": 10, "hidden_size": 2}), encoding="utf-8"
    )
    (model_dir / "tokenizer_config.json").write_text(
        json.dumps({"chat_template": "{{ .Prompt }}", "model_max_length": 2048}), encoding="utf-8"
    )
    return model_dir


@pytest.fixture
def client(settings: Settings):
    """An app wired to the fake engine and the temp store."""
    from fastapi.testclient import TestClient

    from shilp.server.app import create_app

    app = create_app(settings=settings, engine="fake")
    with TestClient(app) as test_client:
        yield test_client
