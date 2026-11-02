"""The transformers engine: the sampler, the failure messages, the wiring.

The sampler is a pure function of logits and parameters, so most of this runs
without a model. The one test that needs weights is opt-in via SHILP_TEST_MODEL,
because it downloads from the Hub and a test suite that needs the network is a
test suite people stop running.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import pytest

needs_torch = pytest.mark.skipif(
    importlib.util.find_spec("torch") is None,
    reason="the sampler is pure torch",
)

from shilp.engine import ENGINES, load_engine  # noqa: E402
from shilp.engine.base import ChatMessage, GenParams  # noqa: E402
from shilp.engine.transformers_engine import (  # noqa: E402
    TransformersEngine,
    first_stop_index,
    sample_next,
)
from shilp.store import MT_SAFETENSORS, Manifest  # noqa: E402


def _manifest(ref: str = "tiny:base") -> Manifest:
    name, tag = ref.split(":")
    return Manifest(name=name, tag=tag)


# --- the sampler -------------------------------------------------------
@needs_torch
def test_temperature_zero_is_greedy():
    import torch
    logits = torch.tensor([0.1, 5.0, 0.2, 0.3])
    assert sample_next(logits, GenParams(temperature=0.0)) == 1


@needs_torch
def test_top_k_of_one_is_also_greedy():
    import torch
    logits = torch.tensor([0.1, 5.0, 4.9, 0.3])
    for _ in range(20):
        assert sample_next(logits, GenParams(temperature=1.0, top_k=1)) == 1


@needs_torch
def test_top_p_keeps_only_the_nucleus():
    import torch
    """With top_p small enough to cover one token, sampling is deterministic."""
    logits = torch.tensor([10.0, 0.0, 0.0, 0.0])
    for _ in range(20):
        assert sample_next(logits, GenParams(temperature=1.0, top_p=0.5)) == 0


@needs_torch
def test_min_p_filters_the_tail():
    import torch
    logits = torch.tensor([10.0, 0.0, 0.0, 0.0])
    for _ in range(20):
        assert sample_next(logits, GenParams(temperature=1.0, min_p=0.5)) == 0


@needs_torch
def test_sampling_actually_varies_at_high_temperature():
    import torch
    """A flat distribution must not collapse to one id, or sampling is broken."""
    logits = torch.zeros(50)
    seen = {sample_next(logits, GenParams(temperature=2.0)) for _ in range(200)}
    assert len(seen) > 1


@needs_torch
def test_seed_makes_sampling_reproducible():
    import torch
    logits = torch.tensor([1.0, 1.0, 1.0, 1.0])
    params = GenParams(temperature=1.0, seed=7)
    first = sample_next(logits, params, generator=torch.Generator().manual_seed(7))
    second = sample_next(logits, params, generator=torch.Generator().manual_seed(7))
    assert first == second


# --- stop sequences ----------------------------------------------------
def test_stop_matches_anywhere_not_just_the_end():
    """The bug this covers: a stop of " " never matched "stairs", because the
    text ends in "s" even though it contains a space."""
    assert first_stop_index(" stairs stairs", (" ",)) == 0


def test_stop_returns_the_earliest_of_several():
    assert first_stop_index("abc<eos>def</s>", ("</s>", "<eos>")) == 3


def test_no_stop_present_is_minus_one():
    assert first_stop_index("nothing here", ("<eos>", "</s>")) == -1


def test_empty_stop_strings_are_ignored():
    assert first_stop_index("text", ("",)) == -1


# --- construction and failure modes ------------------------------------
def test_engines_registry_lists_all_three():
    assert set(ENGINES) == {"llamacpp", "transformers", "fake"}


def test_a_non_model_directory_gets_a_readable_error(tmp_path: Path):
    with pytest.raises(Exception) as exc:
        TransformersEngine(_manifest(), tmp_path)
    message = str(exc.value)
    assert "not a Hugging Face model directory" in message
    assert "llamacpp" in message  # names the alternative


def test_unknown_engine_lists_the_known_ones():
    with pytest.raises(ValueError, match="known:"):
        load_engine(_manifest(), Path("/tmp"), engine="tensorrt")


def test_factory_builds_the_transformers_engine_for_a_real_dir(tmp_path: Path):
    """A config.json is enough to get past the directory check; the load then
    fails on the missing weights, which is a different error and proves the
    routing works."""
    (tmp_path / "config.json").write_text('{"model_type": "gpt2"}', encoding="utf-8")
    with pytest.raises(Exception) as exc:
        load_engine(_manifest(), tmp_path, engine="transformers")
    assert "not a Hugging Face model directory" not in str(exc.value)


# --- pull must keep safetensors ----------------------------------------
def test_pull_keeps_safetensors_weights(store):
    """Regression: .safetensors used to be dropped, leaving a model with config
    and no weights — and a `convert` step with nothing to convert."""
    from shilp.hub import HfSpec, pull
    from tests.conftest import FakeHub

    hub = FakeHub({
        "model.safetensors": b"weights",
        "config.json": b"{}",
        "tokenizer.json": b"{}",
    })
    manifest = pull(store, HfSpec.parse("hf:acme/raw"), hub=hub)

    kinds = {layer.media_type for layer in manifest.layers}
    assert MT_SAFETENSORS in kinds, f"safetensors were dropped; got {kinds}"
    assert any(layer.name == "model.safetensors" for layer in manifest.layers)


# --- opt-in: a real model, if you want one -----------------------------
@needs_torch
@pytest.mark.skipif(
    not os.environ.get("SHILP_TEST_MODEL"),
    reason="set SHILP_TEST_MODEL=<hf repo> to run the end-to-end engine test",
)
def test_generates_from_a_real_model(tmp_path: Path):
    from huggingface_hub import snapshot_download

    model_dir = Path(snapshot_download(os.environ["SHILP_TEST_MODEL"]))
    engine = TransformersEngine(_manifest(), model_dir, num_ctx=512)

    chunks = list(engine.generate("The capital of France is", GenParams(num_predict=4)))
    assert chunks[-1].done is True
    assert chunks[-1].completion_tokens == 4
    assert "".join(c.text for c in chunks[:-1]).strip()

    chat = list(engine.chat([ChatMessage(role="user", content="hi")], GenParams(num_predict=2)))
    assert chat[-1].done
