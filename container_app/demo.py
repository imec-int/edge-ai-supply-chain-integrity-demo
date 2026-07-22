"""
Container/Application-layer demo, and the *last step* of the pipeline: it
consumes the already-attested AI-model chain from ai_model.demo, bakes the
verified quantized model into a signed application image, actually runs that
image, and calls its real inference endpoint. Finally, it tampers with the
model artifact and shows that the application refuses to (re)serve it.

Usage:
    uv run python -m container_app.demo --registry localhost:5000
"""

import json
import shutil
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

import click
from rich.console import Console

from ai_model import signing
from ai_model.demo import Stage, build_and_sign_chain, oras_push, oras_resolve, verify_stage

console = Console()
SCRIPT_DIR = Path(__file__).parent


def run(*args: str, cwd: Path | None = None) -> str:
    try:
        result = subprocess.run(args, cwd=cwd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or "").strip()
        raise click.ClickException(
            f"`{' '.join(args)}` failed (exit {exc.returncode}){': ' + detail if detail else ''}"
        ) from exc
    return result.stdout.strip()


def require_tool(name: str) -> None:
    if shutil.which(name) is None:
        raise click.ClickException(f"'{name}' not found on PATH")


def container_build(container_bin: str, tag_ref: str, containerfile: Path) -> None:
    run(container_bin, "build", "-t", tag_ref, "-f", str(containerfile), str(SCRIPT_DIR))


def container_push(container_bin: str, tag_ref: str) -> None:
    # Podman verifies TLS by default and needs this to push to the local,
    # plain-HTTP registry used by this demo. Docker has no `push --tls-verify`
    # flag; insecure registries are configured at the daemon level instead.
    extra_args = ("--tls-verify=false",) if container_bin == "podman" else ()
    run(container_bin, "push", *extra_args, tag_ref)


def container_logs(container_bin: str, name: str) -> str:
    """Best-effort log fetch (container may already be gone/unhealthy)."""
    result = subprocess.run(
        [container_bin, "logs", name], capture_output=True, text=True
    )
    return (result.stdout + result.stderr).strip()


def generate_sbom(image_ref: str, out_path: Path) -> int:
    run("syft", f"registry:{image_ref}", "-o", f"cyclonedx-json={out_path}")
    components = json.loads(out_path.read_text()).get("components", [])
    return len(components)


def verify_chain(chain, *, label: str) -> bool:
    ok = True
    for stage in (chain.source_stage, chain.onnx_stage, chain.quant_stage):
        console.print(f"[bold]{label} / {stage.name}[/]")
        ok &= verify_stage(stage, pub=chain.pub)
    return ok


def wait_for_healthz(url: str, attempts: int = 60, delay: float = 0.5) -> bool:
    for _ in range(attempts):
        try:
            with urllib.request.urlopen(url, timeout=1) as resp:
                if resp.status == 200:
                    return True
        except (urllib.error.URLError, ConnectionError, TimeoutError, OSError):
            pass
        time.sleep(delay)
    return False


class PredictError(RuntimeError):
    """Raised when the deployed app's /predict endpoint returns an error response."""


def call_predict(url: str, text: str) -> dict:
    payload = json.dumps({"text": text}).encode()
    req = urllib.request.Request(
        url, data=payload, headers={"Content-Type": "application/json"}, method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        body = exc.read().decode(errors="replace")
        try:
            detail = json.loads(body)
        except json.JSONDecodeError:
            detail = body
        raise PredictError(f"/predict returned HTTP {exc.code}: {detail}") from exc


@click.command()
@click.option("--registry", default="localhost:5000", show_default=True)
@click.option("--model-id", default="prajjwal1/bert-tiny", show_default=True,
              help="Hugging Face model")
@click.option("--container-bin", default="podman", show_default=True)
@click.option("--text", default="Supply-chain integrity for edge AI.", show_default=True,
              help="Sample sentence sent to the deployed model's /predict endpoint.")
@click.option("--app-port", default=8091, show_default=True)
def demo(registry: str, model_id: str, container_bin: str, text: str, app_port: int,) -> None:
    for tool in (container_bin, "cosign", "oras", "syft"):
        require_tool(tool)
    if not signing.cosign_available():
        raise click.ClickException("cosign not found on PATH")

    model_dir = SCRIPT_DIR / "model"
    model_dir.mkdir(exist_ok=True)
    app_repo = "supply-chain-demo/sample-app"
    app_tag_ref = f"{registry}/{app_repo}:1.0.0"
    app_policy_path = SCRIPT_DIR / "policy.cue"
    container_name = "supply-chain-demo-app"
    predict_url = f"http://localhost:{app_port}/predict"
    health_url = f"http://localhost:{app_port}/healthz"

    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)

        console.rule("Step 1: Build & sign the AI-model chain (ai_model.demo)")
        chain = build_and_sign_chain(registry=registry, model_id=model_id, work=work)

        console.rule("Step 2: Verify the full chain before trusting the quantized model")
        if not verify_chain(chain, label="pre-deploy"):
            raise click.ClickException("AI-model chain failed verification, aborting deployment")
        console.print(
            "[green]ALLOW[/]: the provenance chain was verified end-to-end "
            "(source \u2192 onnx \u2192 quantized), it is safe to bake into the app image"
        )
        shutil.copy(chain.quant_stage.artifact_path, model_dir / "model.onnx")

        console.rule("Step 3: Build, sign, and verify the application image")
        container_build(container_bin, app_tag_ref, SCRIPT_DIR / "Containerfile")
        container_push(container_bin, app_tag_ref)
        digest = oras_resolve(app_tag_ref)
        app_ref = f"{registry}/{app_repo}@{digest}"
        console.print(f"Pushed {app_ref}")

        sbom_path = work / "app_sbom.cdx.json"
        component_count = generate_sbom(app_ref, sbom_path)
        console.print(f"Syft found {component_count} components (onnxruntime, transformers, flask, ...)")

        app_key, app_pub = signing.generate_key_pair(work / "app_keys")
        signing.sign(app_ref, key=app_key)
        signing.attest(app_ref, predicate_path=sbom_path, predicate_type="cyclonedx", key=app_key)
        verified = signing.verify(app_ref, key=app_pub) and signing.verify_attestation_policy(
            app_ref, key=app_pub, predicate_type="cyclonedx", policy_path=app_policy_path
        )
        console.print("[green]ALLOW: deploy[/]" if verified else "[red]ERROR: verification failed[/]")
        if not verified:
            raise click.ClickException("verification failed on a freshly signed app image")

        console.rule("Step 4: Run the deployed app and call real inference")
        subprocess.run(  # best-effort: nothing to remove on a first-ever run
            [container_bin, "rm", "-f", container_name], capture_output=True, text=True
        )
        run(
            container_bin, "run", "-d", "--rm", "--name", container_name,
            "-p", f"{app_port}:8080", app_tag_ref,
        )
        try:
            if not wait_for_healthz(health_url):
                console.print("[red]Application container logs:[/]")
                console.print(container_logs(container_bin, container_name))
                raise click.ClickException("application container did not become healthy in time")
            result = call_predict(predict_url, text)
            console.print(f"[bold]Live inference result[/] (model={chain.quant_stage.ref}):")
            console.print(result)

            console.rule("Tamper: modify the AI model in the registry, then try to redeploy")
            tampered_path = work / "model.quant.tampered.onnx"
            tampered_path.write_bytes(chain.quant_stage.artifact_path.read_bytes() + b"\x00")
            oras_push(
                chain.quant_stage.tag_ref, tampered_path,
                "application/vnd.onnx.model.quantized",
            )
            tampered_digest = oras_resolve(chain.quant_stage.tag_ref)
            tampered_ref = f"{chain.quant_stage.tag_ref.rsplit(':', 1)[0]}@{tampered_digest}"
            tampered_stage = Stage(
                name="quantized (tampered)",
                ref=tampered_ref,
                artifact_path=tampered_path,
                aibom_path=chain.quant_stage.aibom_path,
                recorded_hash=chain.quant_stage.recorded_hash,
                policy_path=chain.quant_stage.policy_path,
                tag_ref=chain.quant_stage.tag_ref,
            )
            if verify_stage(tampered_stage, pub=chain.pub):
                raise click.ClickException("tampered model verified (unexpected!)")
            console.print(
                "[green]REJECT[/]: tampered model failed signature verification. "
                "The update is refused before it ever reaches an application image - "
                "the already-running container keeps serving only the last verified model:"
            )
            result_after_tamper = call_predict(predict_url, text)
            console.print(result_after_tamper)
        except PredictError as exc:
            console.print(f"[red]{exc}[/]")
            console.print("[red]Application container logs:[/]")
            console.print(container_logs(container_bin, container_name))
            raise click.ClickException("inference call failed, see logs above") from exc
        finally:
            subprocess.run(
                [container_bin, "stop", container_name], capture_output=True, text=True
            )


if __name__ == "__main__":
    demo()
