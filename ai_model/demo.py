"""
AI-model layer demo: a 4-stage chain, each stage its own signed OCI
artifact with its own AI-BOM chained back to the source stage's BOM hash.

  stage 0a  source checkpoint   (real HF weights)         BOM: aibomgen-cli
  stage 0b tokenizer           (vocab/config, same repo)  BOM: ours, links to stage 0a's BOM
  stage 1  ONNX export         (optimum, HF -> ONNX)      BOM: ours, links to stage 0b's BOM
  stage 2  quantized ONNX      (onnxruntime, INT8)        BOM: ours, links to stage 1's BOM

Each stage is pushed, signed, and attested as its own OCI artifact, forming a genuine
hash-chain of provenance rather than a pile of unrelated documents on one node.

Prerequisites:
    - cosign, oras on PATH
    - aibomgen-cli on PATH (https://github.com/idlab-discover/aibomgen-cli)
    - `uv sync --extra onnx-export` (installs optimum-onnx + onnxruntime)

Usage:
    uv run python -m ai_model.demo --registry localhost:5001 --model-id prajjwal1/bert-tiny
"""

import glob
import io
import json
import shutil
import subprocess
import tarfile
import tempfile
from dataclasses import dataclass
from pathlib import Path

import click
from cyclonedx.model.component import ComponentType
from huggingface_hub import HfApi, hf_hub_download
from rich.console import Console
from rich.json import JSON
from rich.panel import Panel
from rich.table import Table

from . import signing
from .aibom import Artifact, CompileProvenance, build_aibom, sha256_file, write_aibom

console = Console()

# Filenames covering vocab/merges/config for the tokenizer implementations
# `transformers.AutoTokenizer` supports (BPE, WordPiece, SentencePiece, ...).
TOKENIZER_FILENAMES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.txt",
    "vocab.json",
    "merges.txt",
    "special_tokens_map.json",
    "added_tokens.json",
    "spiece.model",
    "sentencepiece.bpe.model",
    "config.json",
)


def run(*args: str, cwd: Path | None = None) -> str:
    try:
        result = subprocess.run(args, cwd=cwd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or "").strip()
        raise click.ClickException(
            f"`{' '.join(args)}` failed (exit {exc.returncode}){': ' + detail if detail else ''}"
        ) from exc
    return result.stdout.strip()


def oras_push(registry_ref: str, artifact_path: Path, media_type: str) -> None:
    run(
        "oras", "push", "--plain-http", registry_ref,
        f"{artifact_path.name}:{media_type}",
        cwd=artifact_path.parent,
    )


def oras_resolve(registry_ref: str) -> str:
    return run("oras", "resolve", "--plain-http", registry_ref)


def require_tool(name: str, install_hint: str) -> None:
    if shutil.which(name) is None:
        raise click.ClickException(f"'{name}' not found on PATH. {install_hint}")


def find_weight_file(model_id: str) -> str:
    """Pick the checkpoint file to treat as the 'source artifact' (safetensors > bin)."""
    info = HfApi().model_info(model_id)
    names = [s.rfilename for s in info.siblings]
    for suffix in (".safetensors", ".bin"):
        for name in names:
            if name.endswith(suffix):
                return name
    raise click.ClickException(f"No .safetensors/.bin weight file found for {model_id}")


def _config_json_bytes(path: Path, model_id: str) -> bytes:
    """Return ``config.json``'s bytes, injecting a resolved ``model_type`` if
    the upstream file doesn't set one.
    """
    data = json.loads(path.read_text())
    if "model_type" not in data:
        from transformers import AutoConfig

        data["model_type"] = AutoConfig.from_pretrained(model_id).model_type
    return (json.dumps(data, indent=2, sort_keys=True) + "\n").encode()


def fetch_tokenizer(model_id: str, out_dir: Path) -> Path:
    """Download the tokenizer's vocab/config files (no model weights) for
    ``model_id`` and package them into a single, deterministic tar.gz artifact
    so it can be hashed, pushed, signed, and attested like any other stage.
    """
    info = HfApi().model_info(model_id)
    available = {s.rfilename for s in info.siblings}
    to_fetch = sorted(available & set(TOKENIZER_FILENAMES))
    if not to_fetch:
        raise click.ClickException(
            f"No recognized tokenizer files ({', '.join(TOKENIZER_FILENAMES)}) "
            f"found in {model_id}"
        )

    downloaded = [Path(hf_hub_download(repo_id=model_id, filename=name)) for name in to_fetch]

    out_dir.mkdir(parents=True, exist_ok=True)
    archive_path = out_dir / "tokenizer.tar.gz"
    with tarfile.open(archive_path, "w:gz") as tar:
        for path in downloaded:
            real_path = path.resolve()
            payload = (
                _config_json_bytes(real_path, model_id)
                if path.name == "config.json"
                else real_path.read_bytes()
            )
            info_entry = tarfile.TarInfo(name=path.name)
            info_entry.size = len(payload)
            # Zero out mtime/uid/gid/uname so the archive's bytes (and hash)
            # only depend on the tokenizer file contents, not on when/where
            # they happened to be downloaded.
            info_entry.mtime = 0
            info_entry.uid = info_entry.gid = 0
            info_entry.uname = info_entry.gname = ""
            tar.addfile(info_entry, io.BytesIO(payload))
    return archive_path


def generate_source_aibom(model_id: str, out_dir: Path) -> Path:
    require_tool(
        "aibomgen-cli",
        "Install from https://github.com/idlab-discover/aibomgen-cli/releases",
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    run("aibomgen-cli", "generate", "-m", model_id, "-o", f"{out_dir}/")

    matches = glob.glob(str(out_dir / "*.json"))
    if not matches:
        raise click.ClickException("aibomgen-cli did not produce an AI-BOM file")
    return Path(matches[0])


def export_onnx(model_id: str, out_dir: Path) -> Path:
    """Export ``model_id`` to ONNX via optimum-cli and return the single
    resulting .onnx file.

    This demo only supports architectures that optimum-cli exports as a
    *single* ONNX graph (e.g. plain text encoders like BERT/RoBERTa/
    DistilBERT/ALBERT, or vision encoders like ViT). Encoder-decoder models
    (T5, BART, Whisper, ...) and generation-with-past exports produce
    multiple .onnx files (encoder/decoder/decoder_with_past) and are
    explicitly out of scope - see README "Trying a different model".
    """
    require_tool(
        "optimum-cli",
        "Install with: uv sync --extra onnx-export (this repo) or "
        "'pip install optimum-onnx onnxruntime'",
    )
    run("optimum-cli", "export", "onnx", "-m", model_id, str(out_dir))

    onnx_files = sorted(out_dir.glob("*.onnx"))
    if not onnx_files:
        raise click.ClickException(f"optimum-cli did not produce any .onnx file in {out_dir}")
    if len(onnx_files) > 1:
        names = ", ".join(f.name for f in onnx_files)
        raise click.ClickException(
            f"optimum-cli produced multiple ONNX files for {model_id} ({names}). "
            "This demo only supports single-graph exports (plain encoder models); "
            f"{model_id} looks like an encoder-decoder or generation-with-past "
            "architecture. Try a plain text/vision encoder model instead "
            "(e.g. bert-base-uncased, distilbert-base-uncased, roberta-base, "
            "google/vit-base-patch16-224)."
        )
    return onnx_files[0]


def quantize_onnx(onnx_path: Path, out_path: Path) -> Path:
    try:
        from onnxruntime.quantization import QuantType, quantize_dynamic
    except ImportError as exc:
        raise click.ClickException(
            "onnxruntime.quantization not available. Install with: "
            "uv sync --extra onnx-export"
        ) from exc
    out_path.parent.mkdir(parents=True, exist_ok=True)
    quantize_dynamic(str(onnx_path), str(out_path), weight_type=QuantType.QInt8)
    return out_path


def _package_version(name: str) -> str:
    try:
        from importlib.metadata import version
        return version(name)
    except Exception:
        return "unknown"


@dataclass
class Stage:
    """A pushed, signed, attested OCI artifact - one node in the provenance chain."""

    name: str
    ref: str
    artifact_path: Path
    aibom_path: Path
    recorded_hash: str
    policy_path: Path
    tag_ref: str


@dataclass
class ChainResult:
    """The full signed 3-stage AI-model provenance chain, plus the keys used to sign it.

    Reused by other layers (e.g. container_app.demo) that need to consume an
    already-attested model artifact rather than re-implement the chain logic.
    """

    key: Path
    pub: Path
    source_stage: Stage
    onnx_stage: Stage
    quant_stage: Stage
    tokenizer_stage: Stage


def push_sign_attest(
    *, registry: str, repo: str, tag: str, artifact_path: Path, aibom_path: Path,
    media_type: str, key: Path, recorded_hash: str, policy_path: Path, stage_name: str,
) -> Stage:
    tag_ref = f"{registry}/{repo}:{tag}"
    oras_push(tag_ref, artifact_path, media_type)
    digest = oras_resolve(tag_ref)
    ref = f"{registry}/{repo}@{digest}"
    signing.sign(ref, key=key)
    signing.attest(ref, predicate_path=aibom_path, predicate_type="cyclonedx", key=key)
    return Stage(stage_name, ref, artifact_path, aibom_path, recorded_hash, policy_path, tag_ref)


def verify_stage(stage: Stage, *, pub: Path) -> bool:
    _print_hash_check(stage.recorded_hash, stage.artifact_path)
    return signing.verify(stage.ref, key=pub) and signing.verify_attestation_policy(
        stage.ref, key=pub, predicate_type="cyclonedx", policy_path=stage.policy_path
    )


def build_and_sign_chain(*, registry: str, model_id: str, work: Path) -> ChainResult:
    """Build, push, sign, and attest the 4-stage AI-model provenance chain.

    Stage 0 (source checkpoint, BOM via aibomgen-cli) -> Stage 0b (tokenizer,
    chained to the source BOM) -> Stage 1 (ONNX export) -> Stage 2 (INT8
    quantization), each compiled stage's AI-BOM chained to the previous
    stage's BOM hash. Returns the signed stages plus the keypair used to sign
    them, so callers can verify and/or consume the artifacts further.
    """
    name = model_id.split("/")[-1]
    repo = f"supply-chain-demo/{name}"
    compiled_policy = Path(__file__).parent / "policy.cue"
    source_policy = Path(__file__).parent / "source_policy.cue"
    tokenizer_policy = Path(__file__).parent / "tokenizer_policy.cue"

    key, pub = signing.generate_key_pair(work)

    console.rule(f"Stage 0: source checkpoint for {model_id} (via aibomgen-cli)")
    source_aibom_path = generate_source_aibom(model_id, work / "aibomgen")
    console.print(Panel(
        JSON(source_aibom_path.read_text()),
        title=f"Source AI-BOM: {source_aibom_path.name}",
    ))
    weight_filename = find_weight_file(model_id)
    weight_path = Path(hf_hub_download(repo_id=model_id, filename=weight_filename))
    console.print(f"Source checkpoint: {weight_path.name} ({weight_path.stat().st_size} bytes)")

    source_hash = sha256_file(weight_path)
    source_stage = push_sign_attest(
        registry=registry, repo=repo, tag="source-1.0.0",
        artifact_path=weight_path, aibom_path=source_aibom_path,
        media_type="application/vnd.huggingface.checkpoint", key=key,
        recorded_hash=source_hash, policy_path=source_policy, stage_name="source",
    )
    console.print(f"Pushed {source_stage.ref}")

    console.rule("Stage 0b: tokenizer (vocab/config, chained to the same source repo)")
    tokenizer_path = fetch_tokenizer(model_id, work / "tokenizer_out")
    console.print(f"Tokenizer archive: {tokenizer_path.name} ({tokenizer_path.stat().st_size} bytes)")

    tokenizer_bom = build_aibom(
        model_name=f"{name}-tokenizer", model_version="1.0.0", framework="transformers",
        output_artifact=Artifact.from_file(tokenizer_path),
        source_artifact=None,
        provenance=CompileProvenance(
            compiler_name="huggingface_hub.hf_hub_download",
            compiler_version=_package_version("huggingface_hub"),
            source_format="huggingface-hub-repo", target_format="tokenizer-tar-gz",
            target_device="n/a", parameters={"model_id": model_id},
        ),
        registry_ref=f"{registry}/{repo}",
        source_aibom=Artifact.from_file(source_aibom_path),
        component_type=ComponentType.DATA,
    )
    tokenizer_aibom_path = write_aibom(tokenizer_bom, work / "tokenizer_aibom.cdx.json")
    console.print(Panel(JSON.from_data(tokenizer_bom), title=f"Tokenizer AI-BOM: {tokenizer_aibom_path}"))

    tokenizer_hash = _bom_hash(tokenizer_bom)
    tokenizer_stage = push_sign_attest(
        registry=registry, repo=repo, tag="tokenizer-1.0.0",
        artifact_path=tokenizer_path, aibom_path=tokenizer_aibom_path,
        media_type="application/vnd.huggingface.tokenizer.tar+gzip", key=key,
        recorded_hash=tokenizer_hash, policy_path=tokenizer_policy, stage_name="tokenizer",
    )
    console.print(f"Pushed {tokenizer_stage.ref}")

    console.rule("Stage 1: compile to ONNX (via onnx-optimum)")
    onnx_path = export_onnx(model_id, work / "onnx_out")
    console.print(f"ONNX: {onnx_path.name} ({onnx_path.stat().st_size} bytes)")

    onnx_bom = build_aibom(
        model_name=name, model_version="1.0.0", framework="transformers",
        output_artifact=Artifact.from_file(onnx_path),
        source_artifact=Artifact.from_file(weight_path, name=weight_filename),
        provenance=CompileProvenance(
            compiler_name="optimum", compiler_version=_package_version("optimum"),
            source_format=Path(weight_filename).suffix.lstrip("."),
            target_format="onnx", target_device="cpu",
        ),
        registry_ref=f"{registry}/{repo}",
        source_aibom=Artifact.from_file(source_aibom_path),
    )
    onnx_aibom_path = write_aibom(onnx_bom, work / "onnx_aibom.cdx.json")
    console.print(Panel(JSON.from_data(onnx_bom), title=f"ONNX AI-BOM: {onnx_aibom_path}"))

    onnx_hash = _bom_hash(onnx_bom)
    onnx_stage = push_sign_attest(
        registry=registry, repo=repo, tag="onnx-1.0.0",
        artifact_path=onnx_path, aibom_path=onnx_aibom_path,
        media_type="application/vnd.onnx.model", key=key,
        recorded_hash=onnx_hash, policy_path=compiled_policy, stage_name="onnx",
    )
    console.print(f"Pushed {onnx_stage.ref}")

    console.rule("Stage 2: quantize (via onnxruntime)")
    quant_path = quantize_onnx(onnx_path, work / "quantized" / "model.quant.onnx")
    console.print(
        f"Quantized: {quant_path.name} ({quant_path.stat().st_size} bytes, "
        f"was {onnx_path.stat().st_size} bytes)"
    )

    quant_bom = build_aibom(
        model_name=name, model_version="1.0.0", framework="onnxruntime",
        output_artifact=Artifact.from_file(quant_path),
        source_artifact=Artifact.from_file(onnx_path),
        provenance=CompileProvenance(
            compiler_name="onnxruntime.quantization",
            compiler_version=_package_version("onnxruntime"),
            source_format="onnx", target_format="onnx-int8-dynamic",
            target_device="cpu-generic-edge",
            parameters={"weight_type": "QInt8", "mode": "dynamic"},
        ),
        registry_ref=f"{registry}/{repo}",
        source_aibom=Artifact.from_file(onnx_aibom_path),
    )
    quant_aibom_path = write_aibom(quant_bom, work / "quant_aibom.cdx.json")
    console.print(Panel(JSON.from_data(quant_bom), title=f"Quantized AI-BOM: {quant_aibom_path}"))

    quant_hash = _bom_hash(quant_bom)
    quant_stage = push_sign_attest(
        registry=registry, repo=repo, tag="quant-1.0.0",
        artifact_path=quant_path, aibom_path=quant_aibom_path,
        media_type="application/vnd.onnx.model.quantized", key=key,
        recorded_hash=quant_hash, policy_path=compiled_policy, stage_name="quantized",
    )
    console.print(f"Pushed {quant_stage.ref}")

    return ChainResult(key, pub, source_stage, onnx_stage, quant_stage, tokenizer_stage)


@click.command()
@click.option("--registry", default="localhost:5000", show_default=True)
@click.option("--model-id", default="prajjwal1/bert-tiny", show_default=True,
              help="Hugging Face model to fetch, BOM, and 'compile' through the chain.")
def demo(registry: str, model_id: str) -> None:
    if not signing.cosign_available():
        raise click.ClickException("cosign not found on PATH")

    name = model_id.split("/")[-1]
    repo = f"supply-chain-demo/{name}"

    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        chain = build_and_sign_chain(registry=registry, model_id=model_id, work=work)
        pub = chain.pub
        source_stage, onnx_stage, quant_stage, tokenizer_stage = (
            chain.source_stage, chain.onnx_stage, chain.quant_stage, chain.tokenizer_stage,
        )

        console.rule("Verify: signature + attestation policy (via cosign + CUE policies)")
        chain_ok = True
        for stage in (source_stage, tokenizer_stage, onnx_stage, quant_stage):
            console.print(f"[bold]{stage.name}[/]")
            chain_ok &= verify_stage(stage, pub=pub)
        _print_verdict(chain_ok)

        console.rule("Tamper: push a modified quantized artifact to the same tag")
        quant_path = quant_stage.artifact_path
        quant_hash = quant_stage.recorded_hash
        quant_path.write_bytes(quant_path.read_bytes() + b"\x00")
        tag_ref = f"{registry}/{repo}:quant-1.0.0"
        oras_push(tag_ref, quant_path, "application/vnd.onnx.model.quantized")
        tampered_digest = oras_resolve(tag_ref)
        tampered_ref = f"{registry}/{repo}@{tampered_digest}"

        _print_hash_check(quant_hash, quant_path)
        sig_ok = signing.verify(tampered_ref, key=pub)
        _print_verdict(sig_ok, expect_reject=True)


def _bom_hash(bom: dict) -> str:
    return next(
        h["content"] for h in bom["metadata"]["component"]["hashes"] if h["alg"] == "SHA-256"
    )


def _print_hash_check(recorded_hash: str, actual_path: Path) -> None:
    actual_hash = sha256_file(actual_path)
    match = recorded_hash == actual_hash
    table = Table(show_header=False, box=None)
    table.add_column(style="bold")
    table.add_column()
    table.add_row("BOM-recorded hash", f"[dim]{recorded_hash}[/dim]")
    table.add_row(
        "On-disk hash",
        f"[{'green' if match else 'red'}]{actual_hash}[/] {'(matches)' if match else '(mismatch)'}",
    )
    console.print(table)


def _print_verdict(ok: bool, *, expect_reject: bool = False) -> None:
    if expect_reject:
        console.print("[green]OK[/]: tampered artifact rejected" if not ok else "[red]ERROR[/]: tampered artifact verified")
    else:
        console.print("[green]ALLOW[/]: verified, ready to deploy" if ok else "[red]ERROR[/]: verification failed on a freshly signed artifact")


if __name__ == "__main__":
    demo()

