"""The command line, laid out along the four steps of the pipeline.

    shilp pull      step 1  download from Hugging Face
    shilp build     step 2  reshape: convert, quantize, merge, surgery
    shilp serve     step 3  run it behind an HTTP API
    shilp ps/usage  step 4  watch what it is doing

Every command prints what it did and where the result lives. Nothing is silent,
because a pipeline you cannot inspect is a pipeline you cannot debug.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Optional

import typer

from . import __version__
from .config import Settings, get_settings
from .export import materialise
from .hub import HfSpec, HubError, preview, pull
from .modelfile import ShilpfileError, parse_file
from .modify.convert import convert_to_gguf
from .modify.merge import merge_lora
from .modify.quantize import QUANT_TYPES, quantize
from .modify.surgery import (
    prune_layers,
    set_rope_scaling,
    truncate_vocab,
    write_chat_template,
)
from .modify.toolchain import Toolchain
from .store import MT_GGUF, Layer, Manifest, Store
from .utils import MissingDependency, ShilpError, human_size, parse_ref, sha256_bytes

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Pull a model from Hugging Face, reshape it, serve it like Ollama.",
)
surgery_app = typer.Typer(no_args_is_help=True, help="Structural edits to HF weights.")
app.add_typer(surgery_app, name="surgery")


# --- helpers -----------------------------------------------------------
def _settings() -> Settings:
    return get_settings()


def _store() -> Store:
    return Store(_settings().home)


def _ok(message: str) -> None:
    typer.echo(message)


def _fail(message: str) -> None:
    typer.secho(message, fg=typer.colors.RED, err=True)


def _confirm(question: str, assume_yes: bool) -> bool:
    if assume_yes:
        return True
    return typer.confirm(question, default=False)


def _digest(manifest: Manifest) -> str:
    return sha256_bytes(json.dumps(manifest.to_dict(), sort_keys=True).encode("utf-8"))


# --- 1. pull -----------------------------------------------------------
@app.command()
def pull(
    ref: str = typer.Argument(..., help="hf:org/repo, or hf:org/repo-GGUF:q4_k_m"),
    name: Optional[str] = typer.Option(None, help="Store as this name (default: repo name)"),
    tag: Optional[str] = typer.Option(None, help="Store as this tag (default: revision or quant)"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show files and size, download nothing"),
    revision: str = typer.Option("main", help="Branch, tag or commit sha to pin"),
) -> None:
    """Download a model from the Hugging Face Hub into the local store.

    Pin `--revision` to a commit sha for anything you intend to reproduce later:
    `main` moves, and a model that changes under you is not a dependency.
    """
    settings = _settings()
    spec = HfSpec.parse(ref if ref.startswith("hf:") else f"hf:{ref}")
    if revision != "main" and not spec.is_gguf_repo:
        spec = HfSpec(repo_id=spec.repo_id, revision=revision, quant=spec.quant)
    try:
        if dry_run:
            info = preview(spec, token=settings.hf_token)
            _ok(f"{info['repo_id']} @ {info['revision']}  ({info['total_human']})")
            for item in info["files"]:
                _ok(f"  {human_size(item['size']):>9}  {item['name']}")
            return
        manifest = pull(_store(), spec, token=settings.hf_token, name=name, tag=tag)
    except (HubError, MissingDependency, ShilpError) as exc:
        _fail(str(exc))
        raise typer.Exit(1)

    _ok(f"pulled {manifest.ref}  {manifest.human_size}")
    if manifest.layer_of(MT_GGUF) is None:
        _ok("  safetensors stored — next: shilp build -f Shilpfile  (or convert + quantize)")
    else:
        _ok(f"  ready to serve:  shilp serve --model {manifest.ref}")


@app.command("ls")
def list_models() -> None:
    """List stored models."""
    models = _store().list()
    if not models:
        _ok("no models yet — try: shilp pull hf:bartowski/Qwen2.5-1.5B-Instruct-GGUF:q4_k_m")
        return
    for m in models:
        quant = m.metadata.get("quantization", "-")
        _ok(f"{m.ref:<44} {quant:<9} {m.human_size:>9}  {m.metadata.get('source', '')}")


@app.command()
def show(ref: str = typer.Argument(...)) -> None:
    """Print a model's manifest and the recipe that produced it."""
    store = _store()
    manifest = store.resolve(ref)
    _ok(json.dumps(manifest.to_dict(), indent=2))
    recipe = store.load_recipe(manifest.ref)
    if recipe:
        _ok("\nrecipe:")
        for step in recipe["steps"]:
            _ok("  " + json.dumps(step))


@app.command()
def rm(
    ref: str = typer.Argument(...),
    yes: bool = typer.Option(False, "--yes", "-y", help="Do not ask"),
) -> None:
    """Delete a model and any blobs nothing else references."""
    store = _store()
    manifest = store.resolve(ref)
    if not _confirm(f"delete {manifest.ref} ({manifest.human_size})?", yes):
        raise typer.Exit(0)
    store.delete(manifest.ref)
    _ok(f"deleted {manifest.ref}")


@app.command()
def gc(yes: bool = typer.Option(False, "--yes", "-y")) -> None:
    """Delete blobs that no manifest references."""
    store = _store()
    orphans = store.gc(dry_run=True)
    if not orphans:
        _ok("nothing to collect")
        return
    for digest in orphans:
        _ok(f"  unreferenced {digest[:19]}…")
    if _confirm(f"delete {len(orphans)} blob(s)?", yes):
        store.gc(dry_run=False)
        _ok("collected")


# --- 2. build / reshape ------------------------------------------------
@app.command()
def build(
    file: Path = typer.Option(Path("Shilpfile"), "--file", "-f", help="Path to a Shilpfile"),
    name: Optional[str] = typer.Option(None),
    tag: Optional[str] = typer.Option(None, help="Tag to store the result under"),
) -> None:
    """Create a model from a Shilpfile: source + parameters + template.

    The source may be `hf:...` (pulls if needed), an existing `name:tag` (copies
    its layers, then applies the new parameters), or a local `.gguf` path.
    """
    store = _store()
    settings = _settings()
    try:
        spec = parse_file(file)
    except (ShilpfileError, FileNotFoundError) as exc:
        _fail(f"{file}: {exc}")
        raise typer.Exit(1)

    source = spec.source
    parent: str | None = None

    if source.startswith("hf:"):
        hf_spec = HfSpec.parse(source)
        if store.exists(name or hf_spec.repo_id.split("/")[-1].lower()):
            base = store.resolve(name or hf_spec.repo_id.split("/")[-1].lower())
        else:
            base = pull(store, hf_spec, token=settings.hf_token)
        parent = _digest(base)
    elif source.endswith(".gguf"):
        path = (file.parent / source).resolve()
        layer = store.put_blob(path)
        base = Manifest(name=name or path.stem, tag="latest", layers=[layer])
        base.metadata.update({"source": f"file:{path.name}", "format": "gguf"})
    else:
        base = store.resolve(source)
        parent = _digest(base)

    manifest = Manifest(
        name=name or base.name,
        tag=tag or base.tag,
        layers=list(base.layers),
        params={**base.params, **spec.params},
        template=spec.template or base.template,
        system=spec.system or base.system,
        metadata={**base.metadata, "built_from": base.ref},
        parent=parent,
    )
    store.save_manifest(manifest)
    store.save_recipe(
        manifest,
        [
            {
                "step": "build",
                "shilpfile": file.name,
                "from": base.ref,
                "params": spec.params,
                "template": bool(spec.template),
            }
        ],
    )
    _ok(f"built {manifest.ref}  ({manifest.human_size})")
    _ok(f"  serve:  shilp serve --model {manifest.ref}")


@app.command()
def convert(
    ref: str = typer.Argument(..., help="Stored model ref, or a local HF model directory"),
    out: Path = typer.Option(..., "--out", help="Where to write the .gguf"),
    out_type: str = typer.Option("f16", help="f32 | f16 | bf16"),
) -> None:
    """Convert stored HF safetensors into one GGUF file (needs llama.cpp)."""
    store = _store()
    source = Path(ref)
    if not source.is_dir():
        manifest = store.resolve(ref)
        source = materialise(store, manifest, store.home / "tmp" / f"export-{manifest.name}")
        _ok(f"exported {manifest.ref} to {source}")
    try:
        report = convert_to_gguf(source, out, out_type=out_type, toolchain=Toolchain.discover())
    except (MissingDependency, ShilpError) as exc:
        _fail(str(exc))
        raise typer.Exit(1)
    _ok(f"wrote {report.output}  {human_size(report.size_bytes)}")


@app.command("quantize")
def quantize_model(
    source: Path = typer.Argument(..., help="Input .gguf"),
    out: Path = typer.Option(..., "--out", help="Output .gguf"),
    qtype: str = typer.Option("Q4_K_M", "--qtype", help=", ".join(QUANT_TYPES)),
) -> None:
    """Reduce precision of a GGUF file."""
    try:
        report = quantize(source, out, quant_type=qtype, toolchain=Toolchain.discover())
    except (MissingDependency, ShilpError) as exc:
        _fail(str(exc))
        raise typer.Exit(1)
    _ok(f"wrote {report.output}  {human_size(report.size_bytes)}  ({qtype})")
    _ok(f"  estimated resident memory: {human_size(report.est_resident_bytes)}")



@app.command("merge-lora")
def merge_lora_cmd(
    base: Path = typer.Argument(..., help="Base model directory (safetensors)"),
    adapter: Path = typer.Argument(..., help="LoRA adapter directory"),
    out: Path = typer.Option(..., "--out"),
    dtype: str = typer.Option("float16"),
) -> None:
    """Fold a LoRA adapter into the base weights."""
    try:
        report = merge_lora(base, adapter, out, dtype=dtype)
    except (MissingDependency, ShilpError) as exc:
        _fail(str(exc))
        raise typer.Exit(1)
    _ok(f"merged adapter into {report.output} ({report.dtype})")
    _ok("  next: shilp convert + shilp quantize, in that order")


@surgery_app.command("prune")
def surgery_prune(
    source: Path = typer.Argument(...),
    out: Path = typer.Option(..., "--out"),
    keep: int = typer.Option(..., "--keep", help="Number of transformer layers to keep"),
) -> None:
    """Drop the last N transformer blocks."""
    report = prune_layers(source, out, keep_layers=keep)
    _ok(json.dumps(report.detail, indent=2))
    for warning in report.warnings:
        _fail(f"note: {warning}")


@surgery_app.command("vocab")
def surgery_vocab(
    source: Path = typer.Argument(...),
    out: Path = typer.Option(..., "--out"),
    size: int = typer.Option(..., "--size", help="New vocabulary size"),
) -> None:
    """Truncate the embedding and LM head to the first N tokens."""
    report = truncate_vocab(source, out, new_vocab_size=size)
    _ok(json.dumps(report.detail, indent=2))
    for warning in report.warnings:
        _fail(f"note: {warning}")


@surgery_app.command("rope")
def surgery_rope(
    source: Path = typer.Argument(...),
    out: Path = typer.Option(..., "--out"),
    factor: float = typer.Option(..., "--factor", help="Context extension factor, e.g. 2.0"),
    rope_type: str = typer.Option("linear"),
) -> None:
    """Rescale RoPE so the model accepts longer sequences."""
    report = set_rope_scaling(source, out, factor=factor, rope_type=rope_type)
    _ok(json.dumps(report.detail, indent=2))
    for warning in report.warnings:
        _fail(f"note: {warning}")


@surgery_app.command("template")
def surgery_template(
    source: Path = typer.Argument(...),
    out: Path = typer.Option(..., "--out"),
    file: Path = typer.Option(..., "--file", help="File holding the Jinja chat template"),
) -> None:
    """Set the tokenizer's chat template from a file."""
    template = Path(file).read_text(encoding="utf-8")
    report = write_chat_template(source, out, template=template)
    _ok(json.dumps(report.detail, indent=2))


# --- 3. serve ----------------------------------------------------------
@app.command()
def serve(
    model: Optional[str] = typer.Option(None, "--model", help="Preload this model at startup"),
    host: str = typer.Option("127.0.0.1"),
    port: int = typer.Option(8000),
    engine: str = typer.Option("llamacpp", help="llamacpp | fake"),
    reload: bool = typer.Option(False, "--reload"),
) -> None:
    """Run the API server. `--engine fake` runs the whole surface with no model."""
    import uvicorn

    if model:
        typer.echo(f"preloading {model}")
        from .server.app import create_app as _create

        application = _create(engine=engine)
        application.state.shilp.default_model = model
        uvicorn.run(application, host=host, port=port, log_level="info")
        return
    uvicorn.run(
        "shilp.server.app:create_app",
        host=host,
        port=port,
        reload=reload,
        factory=True,
        log_level="info",
    )


# --- 4. observe --------------------------------------------------------
@app.command()
def ps() -> None:
    """Show which models are resident (reads the running server's /api/ps)."""
    import json as _json
    import urllib.request

    settings = _settings()
    url = f"http://{settings.host}:{settings.port}/api/ps"
    try:
        with urllib.request.urlopen(url, timeout=3) as response:
            _ok(_json.dumps(_json.loads(response.read())["models"], indent=2))
    except Exception as exc:
        _fail(f"cannot reach {url}: {exc}")
        raise typer.Exit(1)


@app.command()
def usage() -> None:
    """Print aggregate latency and throughput from the local telemetry table."""
    from .server.telemetry import Telemetry

    telemetry = Telemetry(_settings().db_path)
    try:
        _ok(json.dumps(telemetry.summary(), indent=2))
    finally:
        telemetry.close()


@app.command()
def doctor() -> None:
    """Check the machine: store, engine, llama.cpp, disk."""
    settings = _settings()
    store = _store()
    _ok(f"shilp {__version__}")
    _ok(f"home        {settings.home}  ({'exists' if settings.home.exists() else 'not created yet'})")
    stats = store.stats()
    _ok(f"models      {stats['models']}  ({human_size(stats['bytes'])})")

    try:
        import llama_cpp  # noqa: F401

        _ok("engine      llama-cpp-python installed")
    except ModuleNotFoundError:
        _fail("engine      llama-cpp-python MISSING  →  pip install -e '.[engine]'")

    try:
        chain = Toolchain.discover()
        _ok(f"llama.cpp   {chain.root}")
        try:
            chain.binary("llama-quantize")
            _ok("            llama-quantize found")
        except MissingDependency:
            _fail("            llama-quantize not built (needed for quantization)")
    except MissingDependency:
        _fail("llama.cpp   not found (only needed for convert/quantize)")

    free = shutil.disk_usage(settings.home if settings.home.exists() else Path.home()).free
    _ok(f"disk free   {human_size(free)}")


@app.command()
def version() -> None:
    """Print the version."""
    _ok(__version__)


if __name__ == "__main__":  # pragma: no cover
    app()
