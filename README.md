# Supply-Chain Integrity Demo

Runnable companion code for the ACM CCS 2026 demo paper *"DEMO: Full-Stack Supply-Chain Integrity Pipeline for Edge AI on Immutable Bootable Containers"*.

The paper proposes a pipeline that signs and attests Bills of Materials at every layer of an edge AI stack (OS, container, AI model), then verifies and rejects tampered artifacts before deployment. This repo implements a local, hardware-free subset of that pipeline, using the same real, off-the-shelf tools (Kairos, Syft, Cosign, CycloneDX, aibomgen-cli) the paper describes.

`container_app` is deliberately the *last* step, tying the other two layers together: it consumes the already-attested AI-model chain produced by `ai_model.demo`, bakes the verified model into a signed application image, actually runs that container, and calls its real `/predict` endpoint for a live inference result — then tampers with the model and shows the running app keeps serving only the last verified one.

## Contents

| Paper claim                                                                                                                                | Demonstrated by                                                                                                                                                                                                                  |
|--------------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **OS Layer:** OS built as a signed OCI image with an SBOM                                                                                  | `os_image/demo.py` builds a real Kairos image (`kairos-init`), pushes it, and generates a CycloneDX SBOM with Syft                                                                                                               |
| **Container & Application Layer:** app images carry a CycloneDX SBOM attached at build time, signed and pushed alongside the image        | `container_app/demo.py` builds an app image, generates a CycloneDX SBOM with Syft, signs + attests + verifies it via the *same* `signing.py` code path as the other two layers                                                  |
| **AI Model Layer:** AI-BOM records model provenance, compiler/quantization parameters, and a reference back to the source model            | `ai_model/demo.py` builds a 4-stage chain (source checkpoint + tokenizer → ONNX → quantized ONNX), where each compiled stage's AI-BOM records the previous stage's AI-BOM hash                                                               |
| **Attestation & Verification:** every artifact is signed and attested and verification checks signature + attestation policy before deploy | All three demos sign (`cosign sign`) and attest (`cosign attest`) each artifact, then verify signature + a CUE policy (`cosign verify` / `verify-attestation --policy`)                                                          |
| AI-BOM chains a compiled artifact back to **the original auditable source model**                                                          | `ai_model/demo.py` uses the third-party `aibomgen-cli` to generate the source model's AI-BOM (including its Hugging Face security scan result), then chains two more AI-BOMs on top of it for the compilation/optimization steps |
| End-to-end integrity gates **real application behaviour**, not just a standalone signature check                                            | `container_app/demo.py` re-verifies the full AI-model chain before baking the model into a running app, serves real inference from it, then shows a tampered model update is rejected while the deployed app keeps working    |

## Out of scope

This repo is a local, single-machine subset of the full pipeline for the EdgeLab initiative. It does **not** include:

- The EdgeLab cluster itself, K3s, GitOps delivery, or real edge hardware. Instead, the OS-image build uses `kairos-init`'s lightweight `init` stage only, and the AI-model demo quantizes on CPU with `onnxruntime` as a stand-in for a real TensorRT/Metis/... device compiler.
- A cluster admission controller (e.g. Sigstore `policy-controller`) enforcing policy at schedule time. The policy is instead checked directly via `cosign verify-attestation --policy` in each demo script.
- SLSA provenance attestations (in-toto build provenance covering builder identity, invocation, and materials). Each demo attaches a signed CycloneDX SBOM/AI-BOM as its attestation predicate, not a separate SLSA provenance statement.
- Runtime enforcement of the immutable, read-only `/usr` root filesystem (Kairos' own boot-time mechanism).

## Prerequisites

- Python 3.12+ and [`uv`](https://docs.astral.sh/uv/)
- A container CLI, for example `podman` or `docker`
- [`cosign`](https://docs.sigstore.dev/cosign/installation/) and [`oras`](https://oras.land/docs/installation) on `PATH`
- [`syft`](https://github.com/anchore/syft) on `PATH` (OS-image and container/app demos)
- [`aibomgen-cli`](https://github.com/idlab-discover/aibomgen-cli) on `PATH` (AI-model and container/app demos, since `container_app.demo` builds the AI-model chain itself)xw:
  ```sh
  tar -xzf aibomgen-cli.tar.gz aibomgen-cli && sudo mv aibomgen-cli /usr/local/bin/
  ```
- A single local OCI registry for demo purposes (note: on MacOS port 5000 might be occupied so pick a different port):
  ```sh
  podman run -d -p 5000:5000 --name registry registry:2
  ```

## Setup

```sh
uv sync                        # base dependencies (all three demos)
uv sync --extra onnx-export    # + optimum-onnx/onnxruntime, needed by ai_model/demo.py and container_app/demo.py
```

## Running the OS-image demo

```sh
uv run python -m os_image.demo --registry localhost:5000
```

Builds a Kairos image, pushes it, generates an SBOM with Syft, signs + attests it with Cosign, verifies the signature and CUE policy (`os_image/policy.cue`), then tampers with the image and shows verification rejecting it.

## Running the AI-model demo

```sh
uv run python -m ai_model.demo --registry localhost:5000 --model-id prajjwal1/bert-tiny
```

Fetches a small Hugging Face model, generates its source AI-BOM with `aibomgen-cli`, fetches and hashes its tokenizer (vocab/config files, chained to the same source AI-BOM), then compiles the model to ONNX (`optimum`) and to quantized ONNX (`onnxruntime`), building and chaining an AI-BOM (`ai_model/aibom.py`) at each compiled stage. Every stage - source, tokenizer, ONNX, and quantized - is pushed, signed, and attested as its own OCI artifact, then the whole chain is verified against `ai_model/source_policy.cue` / `ai_model/tokenizer_policy.cue` / `ai_model/policy.cue` before the final artifact is tampered with to show rejection.

## Running the container/app demo

```sh
uv run python -m container_app.demo --registry localhost:5000 --model-id prajjwal1/bert-tiny
```

This is the builds and signs the same 3-stage AI-model chain itself (i.e. the AI-model demo), then:

1. Re-verifies the full chain (source → onnx → quantized) before trusting the quantized model.
2. Bakes the verified `model.onnx` and the verified tokenizer into `container_app/`, builds the application image on top of them, generates its SBOM with Syft, and signs + attests + verifies the image (same `signing.py` path as the OS demo).
3. Actually runs the container (`podman run -d -p 8091:8080 ...`) and calls its real `POST /predict` endpoint — the printed result is a genuine ONNX inference computed inside the live, verified, deployed workload.
4. Tampers with the quantized model artifact in the registry and re-verifies it as if redeploying an update: verification fails, the update is rejected, and the already-running container is shown continuing to serve only the last verified model.

### Trying a different model
Note that `--model-id` accepts any Hugging Face model, but the whole chain — `optimum-cli` ONNX export, `onnxruntime` dynamic quantization, and `container_app`'s `/predict` was only built and tested against **plain text encoder models that export to a single ONNX graph**.