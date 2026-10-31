"""Engine-side behaviour: template rendering and the resident-model policy."""

from __future__ import annotations

import asyncio
import dataclasses

import pytest

from shilp.config import Settings
from shilp.engine.base import ChatMessage, GenParams
from shilp.engine.fake import FakeEngine
from shilp.engine.loader import ModelLoader, ModelNotReady
from shilp.engine.template import TemplateError, render
from shilp.store import Manifest, Store


# --- template ----------------------------------------------------------
def test_render_substitutes_all_three_variables():
    out = render(
        "<s>{{ .System }}</s><u>{{ .Prompt }}</u><a>{{ .Response }}",
        system="be brief",
        prompt="hello",
    )
    assert out == "<s>be brief</s><u>hello</u><a>"


def test_render_tolerates_trim_markers():
    assert render("{{- .Prompt -}}", prompt="x") == "x"


def test_render_rejects_unsupported_expressions():
    with pytest.raises(TemplateError, match="unsupported"):
        render("{{ .Tools }}", prompt="x")


# --- params ------------------------------------------------------------
def test_params_merge_lets_the_request_win():
    params = GenParams.merge({"temperature": 0.1, "num_ctx": 4096}, {"temperature": 0.9})
    assert params.temperature == 0.9
    assert params.num_ctx == 4096


def test_params_ignore_explicit_nulls():
    params = GenParams.merge({"temperature": 0.1}, {"temperature": None})
    assert params.temperature == 0.1


def test_params_accept_a_single_stop_string():
    params = GenParams.merge({}, {"stop": "</s>"})
    assert params.stop == ("</s>",)


# --- fake engine -------------------------------------------------------
def _manifest(ref="tiny:q4", params=None) -> Manifest:
    name, tag = ref.split(":")
    return Manifest(name=name, tag=tag, params=params or {"num_predict": 5})


def test_fake_engine_streams_words_then_done():
    engine = FakeEngine(_manifest())
    chunks = list(engine.chat([ChatMessage(role="user", content="hi")], GenParams(num_predict=4)))
    assert chunks[-1].done
    assert chunks[-1].completion_tokens == 4
    assert "".join(c.text for c in chunks[:-1]).strip()


def test_fake_engine_is_deterministic():
    engine = FakeEngine(_manifest())
    messages = [ChatMessage(role="user", content="same question")]
    first = "".join(c.text for c in engine.chat(messages, GenParams()))
    second = "".join(c.text for c in engine.chat(messages, GenParams()))
    assert first == second


# --- loader ------------------------------------------------------------
def _loader(store: Store, settings: Settings, max_resident: int = 1) -> ModelLoader:
    tuned = dataclasses.replace(settings, max_resident=max_resident, engine="fake")
    return ModelLoader(store, tuned)


def test_acquire_loads_once_and_reuses(store: Store, settings: Settings):
    store.save_manifest(_manifest("a:1"))
    loader = _loader(store, settings)

    async def run() -> tuple[int, int]:
        async with loader.acquire("a:1") as engine:
            first = id(engine)
        async with loader.acquire("a:1") as engine:
            second = id(engine)
        return first, second

    first, second = asyncio.run(run())
    assert first == second, "the second acquire must reuse the resident engine"
    assert len(loader.loaded()) == 1


def test_lru_eviction_when_a_second_model_arrives(store: Store, settings: Settings):
    store.save_manifest(_manifest("a:1"))
    store.save_manifest(_manifest("b:1"))
    loader = _loader(store, settings, max_resident=1)

    async def run():
        async with loader.acquire("a:1"):
            pass
        async with loader.acquire("b:1"):
            pass
        return loader.loaded()

    loaded = asyncio.run(run())
    assert [m["ref"] for m in loaded] == ["b:1"]


def test_a_model_in_use_is_never_evicted(store: Store, settings: Settings):
    store.save_manifest(_manifest("a:1"))
    store.save_manifest(_manifest("b:1"))
    loader = _loader(store, settings, max_resident=1)

    async def run():
        async with loader.acquire("a:1"):
            with pytest.raises(ModelNotReady):
                async with loader.acquire("b:1"):
                    pass

    asyncio.run(run())
    assert loader.is_loaded("a:1")


def test_unload_closes_everything(store: Store, settings: Settings):
    store.save_manifest(_manifest("a:1"))
    loader = _loader(store, settings)

    async def run():
        async with loader.acquire("a:1"):
            pass
        return await loader.unload()

    assert asyncio.run(run()) == ["a:1"]
    assert loader.loaded() == []
