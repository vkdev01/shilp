"""The browser pages: status at `/`, chat at `/chat`, and their static assets."""

from __future__ import annotations

import pytest

from shilp.store import Manifest, Store


@pytest.fixture
def seeded(store: Store, tmp_path):
    weights = tmp_path / "tiny-Q4_K_M.gguf"
    weights.write_bytes(b"GGUF" * 32)
    store.save_manifest(
        Manifest(
            name="tiny",
            tag="q4",
            layers=[store.put_blob(weights)],
            metadata={"source": "hf:acme/tiny-GGUF", "quantization": "Q4_K_M"},
        )
    )
    return store


def test_status_page_lists_models_and_links_to_chat(client, seeded):
    res = client.get("/")
    assert res.status_code == 200
    assert res.headers["content-type"].startswith("text/html")
    assert "tiny:q4" in res.text
    assert "Q4_K_M" in res.text
    assert 'href="/chat?model=tiny%3Aq4"' in res.text
    assert "/static/css/shilp.css" in res.text


def test_status_page_without_models_explains_the_next_step(client):
    assert "shilp pull" in client.get("/").text


def test_status_page_escapes_store_metadata(client, store: Store, tmp_path):
    weights = tmp_path / "x.gguf"
    weights.write_bytes(b"GGUF")
    store.save_manifest(
        Manifest(
            name="evil",
            tag="latest",
            layers=[store.put_blob(weights)],
            metadata={"source": "<script>alert(1)</script>"},
        )
    )
    text = client.get("/").text
    assert "<script>alert(1)</script>" not in text
    assert "&lt;script&gt;" in text


def test_chat_page_renders_controls_for_both_surfaces(client, seeded):
    res = client.get("/chat")
    assert res.status_code == 200
    text = res.text
    assert 'value="ollama"' in text and 'value="openai"' in text
    assert '<option value="tiny:q4">' in text
    assert 'id="composer"' in text
    assert "/static/js/chat.js" in text
    assert "/static/js/theme.js" in text


def test_chat_page_without_models_disables_the_picker(client):
    text = client.get("/chat").text
    assert "No models stored" in text
    assert "make pull" in text


@pytest.mark.parametrize(
    ("path", "content_type"),
    [
        ("/static/css/shilp.css", "text/css"),
        ("/static/js/chat.js", "javascript"),
        ("/static/js/theme.js", "javascript"),
    ],
)
def test_static_assets_are_served(client, path, content_type):
    res = client.get(path)
    assert res.status_code == 200
    assert content_type in res.headers["content-type"]
    assert res.text.strip()


def test_theme_defines_both_palettes(client):
    css = client.get("/static/css/shilp.css").text
    assert ":root {" in css
    assert '[data-theme="dark"]' in css
