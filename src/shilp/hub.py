"""Step 1 of the pipeline: get weights out of the Hugging Face Hub and into the store.

Two paths, and the difference matters on a laptop:

* **GGUF repo** (`hf:bartowski/Qwen2.5-1.5B-Instruct-GGUF:q4_k_m`) — someone has
  already converted and quantized it. One file, no toolchain, runs immediately.
  This is the path you want on CPU or Apple Silicon.
* **Raw HF repo** (`hf:Qwen/Qwen2.5-1.5B-Instruct`) — safetensors plus config.
  Needed if you intend to *change* the model, because the changes happen on the
  float weights, and only then do you convert to GGUF.

Both end in the same place: a manifest with content-addressed blobs. The hub is
behind a small protocol so tests can run without the network.
"""

from __future__ import annotations

import fnmatch
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Protocol

from .store import (MT_CONFIG, MT_GGUF, MT_SAFETENSORS, MT_TEMPLATE, Layer,
                    Manifest, Store)
from .utils import MissingDependency, ShilpError, human_size

#: Weights + the small files the loader and the converter both need.
WEIGHT_PATTERNS = (
    "*.safetensors",
    "*.safetensors.index.json",
    "*.json",
    "*.txt",
    "tokenizer*",
    "*.model",
)
#: Files that never matter for inference and can be large.
SKIP_PATTERNS = ("*.bin", "*.pt", "*.pth", "*.h5", "*.msgpack", "*.onnx", "original/*")


class HubError(ShilpError):
    """The Hub could not be reached, or the repo does not contain what we asked for."""


@dataclass(frozen=True)
class HubFile:
    name: str
    size: int


@dataclass(frozen=True)
class HfSpec:
    """A parsed `hf:` reference."""

    repo_id: str
    revision: str = "main"
    #: Optional quant token, as in `...-GGUF:q4_k_m`.
    quant: str | None = None

    @classmethod
    def parse(cls, ref: str) -> "HfSpec":
        if not ref.startswith("hf:"):
            raise HubError(f"not a Hugging Face ref: {ref!r} (expected hf:org/repo[:quant])")
        body = ref[3:]
        quant = None
        if body.count(":") == 1:
            body, quant = body.split(":")
        if "/" not in body:
            raise HubError(f"{ref!r} needs an owner: hf:owner/repo")
        return cls(repo_id=body, quant=quant.lower() if quant else None)

    @property
    def is_gguf_repo(self) -> bool:
        return self.quant is not None


class Hub(Protocol):
    """The two calls we need from a hub. Implemented by `HfHub` and by tests."""

    def list_files(self, spec: HfSpec, token: str | None = None) -> list[HubFile]: ...

    def download(
        self,
        spec: HfSpec,
        patterns: Iterable[str],
        local_dir: Path,
        token: str | None = None,
    ) -> Path: ...


class HfHub:
    """Real implementation, on top of `huggingface_hub`.

    The import is lazy so that `import shilp` never requires the Hub client, and
    so a missing package gives one clear sentence instead of a traceback.
    """

    def _api(self):
        try:
            from huggingface_hub import HfApi
        except ModuleNotFoundError as exc:  # pragma: no cover - depends on install
            raise MissingDependency(
                "huggingface_hub is not installed. Run: pip install huggingface_hub"
            ) from exc
        return HfApi()

    def list_files(self, spec: HfSpec, token: str | None = None) -> list[HubFile]:
        api = self._api()
        try:
            info = api.model_info(
                spec.repo_id, revision=spec.revision, token=token, files_metadata=True
            )
        except Exception as exc:
            raise HubError(f"cannot read {spec.repo_id}@{spec.revision}: {exc}") from exc
        files: list[HubFile] = []
        for sibling in info.siblings or []:
            name = sibling.rfilename
            size = getattr(sibling, "size", None) or 0
            files.append(HubFile(name=name, size=size))
        return files

    def download(
        self,
        spec: HfSpec,
        patterns: Iterable[str],
        local_dir: Path,
        token: str | None = None,
    ) -> Path:
        try:
            from huggingface_hub import snapshot_download
        except ModuleNotFoundError as exc:  # pragma: no cover
            raise MissingDependency(
                "huggingface_hub is not installed. Run: pip install huggingface_hub"
            ) from exc
        local_dir.mkdir(parents=True, exist_ok=True)
        try:
            return Path(
                snapshot_download(
                    repo_id=spec.repo_id,
                    revision=spec.revision,
                    allow_patterns=list(patterns),
                    ignore_patterns=list(SKIP_PATTERNS),
                    local_dir=local_dir,
                    token=token,
                )
            )
        except Exception as exc:
            raise HubError(f"download failed for {spec.repo_id}: {exc}") from exc


def select_files(files: list[HubFile], spec: HfSpec) -> list[HubFile]:
    """Pick the files we actually need for the requested path."""
    if spec.is_gguf_repo:
        quant = (spec.quant or "").replace("-", "_")
        picked = [
            f
            for f in files
            if f.name.lower().endswith(".gguf") and quant in f.name.lower().replace("-", "_")
        ]
        if not picked:
            available = sorted({f.name for f in files if f.name.lower().endswith(".gguf")})
            raise HubError(
                f"no GGUF file matching quant {spec.quant!r} in {spec.repo_id}. "
                f"Available: {', '.join(available) or 'none'}"
            )
        # Sharded GGUFs (00001-of-00003) are not handled; say so plainly.
        if len(picked) > 1 and any("-of-" in f.name for f in picked):
            raise HubError(
                f"{spec.repo_id}:{spec.quant} is split across shards, which shilp "
                "does not join yet. Pick a repo that publishes a single file."
            )
        return picked

    picked = [
        f
        for f in files
        if any(fnmatch.fnmatch(f.name, p) for p in WEIGHT_PATTERNS)
        and not any(fnmatch.fnmatch(f.name, s) for s in SKIP_PATTERNS)
    ]
    if not any(f.name.endswith(".safetensors") for f in picked):
        raise HubError(
            f"{spec.repo_id} has no .safetensors weights at revision "
            f"{spec.revision}. If it only publishes GGUF, use hf:{spec.repo_id}:<quant>."
        )
    return picked


def pull(
    store: Store,
    spec: HfSpec,
    *,
    hub: Hub | None = None,
    token: str | None = None,
    name: str | None = None,
    tag: str | None = None,
    workdir: Path | None = None,
) -> Manifest:
    """Download a repo and register it in the store.

    For a GGUF repo this produces a runnable model. For a raw HF repo it stores
    the safetensors and config so that `shilp convert` can pick them up.
    """
    hub = hub or HfHub()
    files = hub.list_files(spec, token=token)
    chosen = select_files(files, spec)
    total = sum(f.size for f in chosen)

    default_name = spec.repo_id.split("/")[-1].lower()
    model_name = name or default_name
    model_tag = tag or (spec.quant if spec.quant else spec.revision.replace("/", "-"))

    scratch = workdir or (store.home / "tmp" / f"pull-{model_name}-{model_tag}")
    if scratch.exists():
        shutil.rmtree(scratch)
    local = hub.download(spec, [f.name for f in chosen], scratch, token=token)

    manifest = Manifest(
        name=model_name,
        tag=model_tag,
        params={},
        metadata={
            "source": f"hf:{spec.repo_id}",
            "revision": spec.revision,
            "quant": spec.quant,
            "download_bytes": total,
            "download_human": human_size(total),
        },
    )

    for hub_file in chosen:
        path = local / hub_file.name
        if not path.is_file():  # partial download; skip rather than fail
            continue
        lower = path.name.lower()
        if lower.endswith(".gguf"):
            stored = store.put_blob(path, move=True)
            manifest.layers.append(
                Layer(stored.digest, stored.size, MT_GGUF, path.name)
            )
        elif lower.endswith(".safetensors"):
            # Without this the weights of a raw HF repo are silently dropped and
            # `convert` later fails with nothing to convert. MT_SAFETENSORS exists
            # for exactly this.
            stored = store.put_blob(path, move=True)
            manifest.layers.append(
                Layer(stored.digest, stored.size, MT_SAFETENSORS, path.name)
            )
        elif lower.endswith((".json", ".txt", ".model")) or "tokenizer" in lower:
            data = path.read_bytes()
            manifest.layers.append(
                store.put_bytes(data, path.name, MT_TEMPLATE if "template" in lower else MT_CONFIG)
            )

    if not manifest.layers:
        raise HubError(f"nothing usable was downloaded from {spec.repo_id}")

    weights = manifest.layer_of(MT_GGUF)
    if weights:
        manifest.metadata.setdefault("format", "gguf")
        manifest.metadata.setdefault("quantization", spec.quant or "unknown")
    else:
        manifest.metadata.setdefault("format", "safetensors")
        manifest.metadata.setdefault(
            "next_step", "run `shilp convert` then `shilp quantize`, or serve the float weights"
        )

    store.save_manifest(manifest)
    store.save_recipe(
        manifest,
        [
            {
                "step": "pull",
                "spec": f"hf:{spec.repo_id}",
                "revision": spec.revision,
                "files": [f.name for f in chosen],
                "bytes": total,
            }
        ],
    )
    shutil.rmtree(scratch, ignore_errors=True)
    return manifest


def preview(spec: HfSpec, *, hub: Hub | None = None, token: str | None = None) -> dict[str, Any]:
    """What `shilp pull --dry-run` prints: the files and the download size."""
    hub = hub or HfHub()
    files = hub.list_files(spec, token=token)
    chosen = select_files(files, spec)
    return {
        "repo_id": spec.repo_id,
        "revision": spec.revision,
        "quant": spec.quant,
        "files": [{"name": f.name, "size": f.size} for f in chosen],
        "total_bytes": sum(f.size for f in chosen),
        "total_human": human_size(sum(f.size for f in chosen)),
    }
