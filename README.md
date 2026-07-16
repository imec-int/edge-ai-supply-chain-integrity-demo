# Supply-Chain Integrity Demo

Runnable companion code for the CCS 2026 demo paper *"DEMO: Full-Stack Supply-Chain Integrity Pipeline for Edge AI on Immutable Bootable Containers"*.

The paper proposes a pipeline that signs and attests Bills of Materials at every layer of an edge AI stack (OS, container, AI model), then verifies and rejects tampered artifacts before deployment. This repo implements a local, hardware-free subset of that pipeline for two of its layers, using the same real, off-the-shelf tools (Kairos, Syft, Cosign, CycloneDX, aibomgen-cli) the paper describes.

## What the demo shows

| Claim                                                                                                                                      | Demonstrated by                                                                                                                                                                                                                  |
|--------------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **OS Layer:** OS built as a signed OCI image with an SBOM                                                                                  | `os_image/demo.py` builds a real Kairos image (`kairos-init`), pushes it, and generates a CycloneDX SBOM with Syft                                                                                                               |
| **AI Model Layer:** AI-BOM records model provenance, compiler/quantization parameters, and a reference back to the source model            | `ai_model/demo.py` builds a 3-stage chain (source checkpoint → ONNX → quantized ONNX), where each compiled stage's AI-BOM records the previous stage's AI-BOM hash                                                               |
| **Attestation & Verification:** every artifact is signed and attested and verification checks signature + attestation policy before deploy | Both demos sign (`cosign sign`) and attest (`cosign attest`) each artifact, then verify signature + a CUE policy (`cosign verify` / `verify-attestation --policy`)                                                               |
| **Threat Model:** tampering with an artifact in transit is detected and rejected                                                           | Both demos tamper with the final artifact after signing and show that verification fails and the (simulated) deployment is rejected                                                                                              |
| AI-BOM chains a compiled artifact back to **the original auditable source model**                                                          | `ai_model/demo.py` uses the third-party `aibomgen-cli` to generate the source model's AI-BOM (including its Hugging Face security scan result), then chains two more AI-BOMs on top of it for the compilation/optimization steps |

## Out of scope

This repo is a local, single-machine subset of the full pipeline for the EdgeLab initiative. It does **not** include:

- The EdgeLab cluster itself, K3s, GitOps delivery, or real edge hardware. Instead, the OS-image build uses `kairos-init`'s lightweight `init` stage only, and the AI-model demo quantizes on CPU with `onnxruntime` as a stand-in for a real TensorRT/Metis/... device compiler.
- A cluster admission controller (e.g. Sigstore `policy-controller`) enforcing policy at schedule time — policy is instead checked directly via `cosign verify-attestation --policy` in each demo script.

## Prerequisites

- Python 3.12+ and [`uv`](https://docs.astral.sh/uv/)
- A container CLI, for example `podman` or `docker`
- [`cosign`](https://docs.sigstore.dev/cosign/installation/) and [`oras`](https://oras.land/docs/installation) on `PATH`
- [`syft`](https://github.com/anchore/syft) on `PATH` (OS-image demo only)
- [`aibomgen-cli`](https://github.com/idlab-discover/aibomgen-cli) on `PATH` (AI-model demo only) — a prebuilt archive is included at `aibomgen-cli.tar.gz`:
  ```sh
  tar -xzf aibomgen-cli.tar.gz aibomgen-cli && sudo mv aibomgen-cli /usr/local/bin/
  ```
- Two local OCI registries for demo purposes:
  ```sh
  podman run -d -p 5001:5000 --name registry-os registry:2
  podman run -d -p 5002:5000 --name registry-ai registry:2
  ```

## Setup

```sh
uv sync                        # base dependencies (both demos)
uv sync --extra onnx-export    # + optimum-onnx/onnxruntime, needed by ai_model/demo.py
```

## Running the OS-image demo

```sh
uv run python -m os_image.demo --registry localhost:5001
```

Builds a Kairos image, pushes it, generates an SBOM with Syft, signs + attests it with Cosign, verifies the signature and CUE policy (`os_image/policy.cue`), then tampers with the image and shows verification rejecting it.

## Running the AI-model demo

```sh
uv run python -m ai_model.demo --registry localhost:5002 --model-id prajjwal1/bert-tiny
```

Fetches a small Hugging Face model, generates its source AI-BOM with `aibomgen-cli`, compiles it to ONNX (`optimum`) and then to quantized ONNX (`onnxruntime`), building and chaining an AI-BOM (`ai_model/aibom.py`) at each compiled stage. Every stage is pushed, signed, and attested as its own OCI artifact, then the whole chain is verified against `ai_model/source_policy.cue` / `ai_model/policy.cue` before the final artifact is tampered with to show rejection.
