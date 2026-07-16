"""
Thin ``cosign`` wrappers. Every actual cryptographic operation (signing, attestation,
verification, transparency-log interaction) is delegated to the `cosign` CLI -
this module just shells out to it so the demo scripts read cleanly.
"""

import os
import shutil
import subprocess
from pathlib import Path


class CosignError(RuntimeError):
    """Raised when a cosign invocation fails or cosign is unavailable."""


def cosign_available() -> bool:
    return shutil.which("cosign") is not None


def _run(args: list[str]) -> str:
    env = os.environ.copy()
    env.setdefault("COSIGN_PASSWORD", "")
    try:
        result = subprocess.run(
            ["cosign", *args], check=True, capture_output=True, text=True, env=env
        )
    except subprocess.CalledProcessError as exc:
        raise CosignError(
            f"cosign {args[0]} failed (exit {exc.returncode}): {exc.stderr.strip()}"
        ) from exc
    return result.stdout


def generate_key_pair(work_dir: Path) -> tuple[Path, Path]:
    """Generate a throwaway, passwordless demo keypair in ``work_dir``."""
    _run(["generate-key-pair", "--output-key-prefix", str(work_dir / "cosign")])
    return work_dir / "cosign.key", work_dir / "cosign.pub"


def sign(image_ref: str, *, key: Path, allow_http: bool = True) -> None:
    args = ["sign", "--yes", "--key", str(key)]
    if allow_http:
        args.append("--allow-http-registry")
    _run([*args, image_ref])


def attest(
    image_ref: str, *, predicate_path: Path, predicate_type: str, key: Path,
    allow_http: bool = True,
) -> None:
    args = [
        "attest",
        "--yes",
        "--key",
        str(key),
        "--predicate",
        str(predicate_path),
        "--type",
        predicate_type,
    ]
    if allow_http:
        args.append("--allow-http-registry")
    _run([*args, image_ref])


def verify(image_ref: str, *, key: Path, allow_http: bool = True) -> bool:
    args = ["verify", "--key", str(key)]
    if allow_http:
        args.append("--allow-http-registry")
    try:
        _run([*args, image_ref])
        return True
    except CosignError:
        return False


def verify_attestation_policy(
    image_ref: str, *, key: Path, predicate_type: str, policy_path: Path,
    allow_http: bool = True,
) -> bool:
    """Verify the attestation AND evaluate it against a CUE policy - both native cosign features."""
    args = [
        "verify-attestation",
        "--key",
        str(key),
        "--type",
        predicate_type,
        "--policy",
        str(policy_path),
    ]
    if allow_http:
        args.append("--allow-http-registry")
    try:
        _run([*args, image_ref])
        return True
    except CosignError:
        return False

