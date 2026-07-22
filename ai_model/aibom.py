"""Build a CycloneDX 1.6 AI Bill of Materials (AI-BOM) for a compiled model artifact."""

import datetime
import hashlib
import json
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from cyclonedx.model import HashAlgorithm, HashType, Property
from cyclonedx.model.bom import Bom
from cyclonedx.model.component import Component, ComponentType
from cyclonedx.output.json import JsonV1Dot6

_HASH_CHUNK = 64 * 1024


def sha256_file(path: str | Path) -> str:
    """Return the hex SHA-256 digest of a file, read in streaming chunks."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(_HASH_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass
class Artifact:
    """A single file artifact referenced by the AI-BOM (a model, dataset, ...)."""

    name: str
    path: str
    sha256: str

    @classmethod
    def from_file(cls, path: str | Path, name: str | None = None) -> "Artifact":
        p = Path(path)
        return cls(name=name or p.name, path=str(path), sha256=sha256_file(p))


@dataclass
class CompileProvenance:
    """Provenance for a single compilation step (ONNX -> device-specific artifact)."""

    compiler_name: str  # e.g. "tensorrt", "axelera-compiler"
    compiler_version: str  # e.g. "10.3.0"
    source_format: str  # e.g. "onnx"
    target_format: str  # e.g. "tensorrt-engine", "axelera-metis"
    target_device: str  # e.g. "nvidia-jetson-agx-orin"
    parameters: dict[str, str] = field(default_factory=dict)


def _properties(items: dict[str, Any]) -> list[Property]:
    """Render a dict as a sorted, stable list of CycloneDX ``Property`` models."""
    return [
        Property(
            name=name, value="Unknown" if items[name] is None else str(items[name])
        )
        for name in sorted(items)
    ]


def build_aibom(
    *,
    model_name: str,
    model_version: str,
    framework: str,
    output_artifact: Artifact,
    source_artifact: Artifact | None,
    provenance: CompileProvenance,
    registry_ref: str | None = None,
    source_aibom: Artifact | None = None,
    component_type: ComponentType = ComponentType.MACHINE_LEARNING_MODEL,
) -> dict[str, Any]:
    """Build a CycloneDX 1.6 AI-BOM document as a plain ``dict``.

    ``source_aibom``, when given, is the AI-BOM for the source model this artifact was
    compiled from. Recording its hash lets a verifier walk the chain back from the
    compiled artifact to the audited source model.

    ``component_type`` defaults to a machine-learning-model subject, but callers can
    override it (e.g. ``ComponentType.DATA``) for non-model artifacts, such as a
    tokenizer, that still need the same hash/provenance-chaining treatment.
    """
    model_ref = f"{model_name}@{model_version}"

    properties = {
        "Framework": framework,
        "Compiler": provenance.compiler_name,
        "Compiler Version": provenance.compiler_version,
        "Source Format": provenance.source_format,
        "Target Format": provenance.target_format,
        "Target Device": provenance.target_device,
        "Registry Reference": registry_ref,
        **{f"Compiler Param: {k}": v for k, v in provenance.parameters.items()},
    }
    if source_aibom is not None:
        properties["Source Model AI-BOM"] = source_aibom.name
        properties["Source Model AI-BOM SHA-256"] = source_aibom.sha256

    subject = Component(
        type=component_type,
        bom_ref=model_ref,
        name=model_name,
        version=model_version,
        hashes=[HashType(alg=HashAlgorithm.SHA_256, content=output_artifact.sha256)],
        properties=_properties(properties),
    )

    bom = Bom(serial_number=uuid.uuid4())
    bom.metadata.timestamp = datetime.datetime.now(datetime.timezone.utc)
    bom.metadata.component = subject
    bom.metadata.tools.components.add(
        Component(
            type=ComponentType.APPLICATION,
            name="supply-chain-integrity-demo",
            version="0.1.0",
            description="AI-BOM generator for compiled model artifacts",
        )
    )

    dependencies: list[Component] = []
    if source_artifact is not None:
        source_component = Component(
            type=ComponentType.DATA,
            bom_ref=f"source-model:{source_artifact.name}",
            name=source_artifact.name,
            description="Source model consumed by the compilation step",
            hashes=[
                HashType(alg=HashAlgorithm.SHA_256, content=source_artifact.sha256)
            ],
            properties=_properties({"Format": provenance.source_format}),
        )
        bom.components.add(source_component)
        dependencies.append(source_component)

    compiler_component = Component(
        type=ComponentType.APPLICATION,
        bom_ref=f"compiler:{provenance.compiler_name}@{provenance.compiler_version}",
        name=provenance.compiler_name,
        version=provenance.compiler_version,
        description=(
            f"Compiler that produced the {provenance.target_format} artifact "
            f"for {provenance.target_device}"
        ),
    )
    bom.components.add(compiler_component)
    dependencies.append(compiler_component)

    bom.register_dependency(subject, dependencies)

    return json.loads(JsonV1Dot6(bom).output_as_string())


def write_aibom(bom: dict[str, Any], path: str | Path) -> Path:
    """Serialize ``bom`` to ``path`` as stably formatted JSON (matters since it's signed)."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(bom, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out
