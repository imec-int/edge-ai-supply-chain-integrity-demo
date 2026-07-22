"""OS-layer demo: build a Kairos image, generate an SBOM (Syft), sign + attest
(Cosign), push, verify (signature + CUE policy), tamper, and reject.

Usage:
    uv run python -m os_image.demo --registry localhost:5001
"""

import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import click
from rich.console import Console

from ai_model import signing

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


def oras_resolve(tag_ref: str) -> str:
    return run("oras", "resolve", "--plain-http", tag_ref)


def generate_sbom(image_ref: str, out_path: Path) -> int:
    run("syft", f"registry:{image_ref}", "-o", f"cyclonedx-json={out_path}")
    components = json.loads(out_path.read_text()).get("components", [])
    return len(components)


@click.command()
@click.option("--registry", default="localhost:5000", show_default=True)
@click.option("--container-bin", default="podman", show_default=True)
def demo(registry: str, container_bin: str) -> None:
    for tool in (container_bin, "cosign", "oras", "syft"):
        require_tool(tool)

    repo = "supply-chain-demo/kairos-os"
    tag_ref = f"{registry}/{repo}:1.0.0"
    policy_path = SCRIPT_DIR / "policy.cue"

    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)

        console.rule("Build the Kairos OS image")
        container_build(container_bin, tag_ref, SCRIPT_DIR / "Containerfile")

        console.rule("Push it to the registry")
        container_push(container_bin, tag_ref)
        digest = oras_resolve(tag_ref)
        ref = f"{registry}/{repo}@{digest}"
        console.print(f"Pushed {ref}")

        console.rule("Generate an SBOM with Syft")
        sbom_path = work / "sbom.cdx.json"
        component_count = generate_sbom(ref, sbom_path)
        console.print(f"Syft found {component_count} components")

        console.rule("Sign the image and attest the SBOM")
        key, pub = signing.generate_key_pair(work)
        signing.sign(ref, key=key)
        signing.attest(ref, predicate_path=sbom_path, predicate_type="cyclonedx", key=key)

        console.rule("Verify signature and CUE policy")
        verified = signing.verify(ref, key=pub) and signing.verify_attestation_policy(
            ref, key=pub, predicate_type="cyclonedx", policy_path=policy_path
        )
        console.print("[green]ALLOW: deploy[/]" if verified else "[red]ERROR: verification failed[/]")
        if not verified:
            raise click.ClickException("verification failed on a freshly signed image")

        console.rule("Tamper: rebuild FROM the signed image, push under the same tag")
        tamper_containerfile = work / "Containerfile.tamper"
        tamper_containerfile.write_text(
            f'FROM {tag_ref}\nRUN echo "attacker-was-here" > /etc/tamper-marker\n'
        )
        run(container_bin, "build", "-t", tag_ref, "-f", str(tamper_containerfile), str(work))
        container_push(container_bin, tag_ref)
        tampered_digest = oras_resolve(tag_ref)
        tampered_ref = f"{registry}/{repo}@{tampered_digest}"
        console.print(f"Tampered ref: {tampered_ref} (original: {ref})")

        console.rule("Verify the tampered image (expect rejection)")
        if signing.verify(tampered_ref, key=pub):
            raise click.ClickException("tampered image verified (unexpected!)")
        console.print("[green]REJECT: tampered image has no valid signature, as expected[/]")


if __name__ == "__main__":
    demo()


