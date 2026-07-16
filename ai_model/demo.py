"""
AI-model layer demo: fetch a real Hugging Face model, generate a source-model AI-BOM
with the third-party `aibomgen-cli` tool, "compile" it (HF checkpoint -> ONNX, via
`optimum`), build our own compiled-model AI-BOM that chains back to the source BOM,
then sign -> attest (both BOMs) -> push -> verify -> tamper -> reject.

This mirrors the paper's claim that a compiled artifact's integrity "can only be
guaranteed through the attestation chain linking it back to the original auditable
source model" - here that chain is: aibomgen-cli's source-model AI-BOM -> our
compiled-model AI-BOM (which records the source BOM's hash) -> both cosign-attested
on the same pushed artifact.

Prerequisites (see ../README.md):
    - cosign, oras on PATH
    - aibomgen-cli on PATH (https://github.com/idlab-discover/aibomgen-cli)
    - `uv sync --extra onnx-export` (installs optimum-onnx + onnxruntime for the
      real ONNX export step)

Usage:
    uv run python -m ai_model.demo --registry localhost:5001 --model-id prajjwal1/bert-tiny
"""

"""
AI-model layer demo: a 3-stage compilation chain, each stage its own signed OCI
artifact with its own AI-BOM chained back to the previous stage's BOM hash.

  stage 0  source checkpoint   (real HF weights)        BOM: aibomgen-cli (3rd party)
  stage 1  ONNX export         (optimum, HF -> ONNX)     BOM: ours, links to stage 0's BOM
  stage 2  quantized ONNX      (onnxruntime, INT8)       BOM: ours, links to stage 1's BOM

Stage 2 stands in for the paper's device-specific compilation (ONNX -> TensorRT engine /
Axelera Metis output): it's a real, meaningful, CPU-only transformation (dynamic INT8
quantization measurably shrinks the model) that captures the same "quantization
choices" AI-BOM concern the paper describes, without requiring a reviewer to own an
NVIDIA GPU + JetPack or an Axelera accelerator + SDK. Swap in a real device compiler at
this stage if you have the hardware - the BOM/sign/attest/verify mechanism is identical.

Every stage is pushed, signed, and attested as its own OCI artifact (not just extra
attestations bolted onto one final digest) - this is what makes it a genuine hash-chain
of provenance rather than a pile of unrelated documents on one node.

Prerequisites (see ../README.md):
    - cosign, oras on PATH
    - aibomgen-cli on PATH (https://github.com/idlab-discover/aibomgen-cli)
    - `uv sync --extra onnx-export` (installs optimum-onnx + onnxruntime)

Usage:
    uv run python -m ai_model.demo --registry localhost:5001 --model-id prajjwal1/bert-tiny
"""

import glob
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import click
from huggingface_hub import HfApi, hf_hub_download
from rich.console import Console
from rich.json import JSON
from rich.panel import Panel
from rich.table import Table

from . import signing
from .aibom import Artifact, CompileProvenance, build_aibom, sha256_file, write_aibom

console = Console()


def run(*args: str, cwd: Path | None = None) -> str:
    result = subprocess.run(args, cwd=cwd, check=True, capture_output=True, text=True)
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


def generate_source_aibom(model_id: str, out_dir: Path) -> Path:
    require_tool(
        "aibomgen-cli",
        "Install from https://github.com/idlab-discover/aibomgen-cli/releases",
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    # aibomgen-cli's `-o` takes a *file* path and only uses the directory portion, so a
    # bare directory name without a trailing slash gets its last component stripped as
    # if it were a filename - always pass it with a trailing slash.
    run("aibomgen-cli", "generate", "-m", model_id, "-o", f"{out_dir}/")
    matches = glob.glob(str(out_dir / "*.json"))
    if not matches:
        raise click.ClickException("aibomgen-cli did not produce an AI-BOM file")
    return Path(matches[0])


def export_onnx(model_id: str, out_dir: Path) -> Path:
    require_tool(
        "optimum-cli",
        "Install with: uv sync --extra onnx-export (this repo) or "
        "'pip install optimum-onnx onnxruntime'",
    )
    run("optimum-cli", "export", "onnx", "-m", model_id, str(out_dir))
    onnx_path = out_dir / "model.onnx"
    if not onnx_path.is_file():
        raise click.ClickException(f"optimum-cli did not produce {onnx_path}")
    return onnx_path


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
    return Stage(stage_name, ref, artifact_path, aibom_path, recorded_hash, policy_path)


def verify_stage(stage: Stage, *, pub: Path) -> bool:
    _print_hash_check(stage.recorded_hash, stage.artifact_path)
    return signing.verify(stage.ref, key=pub) and signing.verify_attestation_policy(
        stage.ref, key=pub, predicate_type="cyclonedx", policy_path=stage.policy_path
    )


@click.command()
@click.option("--registry", default="localhost:5001", show_default=True)
@click.option("--model-id", default="prajjwal1/bert-tiny", show_default=True,
              help="Small Hugging Face model to fetch, BOM, and 'compile' through the chain.")
def demo(registry: str, model_id: str) -> None:
    if not signing.cosign_available():
        raise click.ClickException("cosign not found on PATH")

    name = model_id.split("/")[-1]
    repo = f"supply-chain-demo/{name}"
    compiled_policy = Path(__file__).parent / "policy.cue"
    source_policy = Path(__file__).parent / "source_policy.cue"

    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        key, pub = signing.generate_key_pair(work)

        # --- Stage 0: source checkpoint + aibomgen-cli's AI-BOM ---------------------
        console.rule(f"[bold cyan]Stage 0: source checkpoint for {model_id} (aibomgen-cli)")
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

        # --- Stage 1: HF checkpoint -> ONNX (optimum) -------------------------------
        console.rule("[bold cyan]Stage 1: compile to ONNX (optimum) - our AI-BOM links to Stage 0")
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

        # --- Stage 2: ONNX -> quantized ONNX (onnxruntime) --------------------------
        console.rule(
            "[bold cyan]Stage 2: quantize (onnxruntime, INT8, CPU-only) - "
            "stands in for TensorRT/Metis, our AI-BOM links to Stage 1"
        )
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

        # --- Verify the whole chain --------------------------------------------------
        console.rule("[bold cyan]Verify: signature + attestation policy at every stage (cosign + CUE)")
        chain_ok = True
        for stage in (source_stage, onnx_stage, quant_stage):
            console.print(f"[bold]{stage.name}[/]")
            chain_ok &= verify_stage(stage, pub=pub)
        _print_verdict(chain_ok)

        # --- Tamper the final (deployable) artifact and show rejection --------------
        console.rule("[bold cyan]Tamper: push a modified quantized artifact to the same tag")
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
        console.print("[green]OK: tampered artifact rejected[/]" if not ok else "[red]ERROR: tampered artifact verified![/]")
    else:
        console.print("[green]ALLOW: deploy[/]" if ok else "[red]ERROR: verification failed on a freshly signed artifact[/]")


if __name__ == "__main__":
    demo()





