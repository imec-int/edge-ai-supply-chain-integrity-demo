# supply-chain-integrity-demo

Runnable, locally-reproducible reference implementation of a layered supply-chain
integrity pipeline for edge AI: a **Kairos OS image** and a **compiled AI model**, each
signed, attested, and verified with real, off-the-shelf tooling - no accelerator
hardware, cloud account, or proprietary SDK required.

This is a companion artifact to a research paper on full-stack supply-chain integrity
for edge AI (OS -> container -> AI model, signed and attested at every layer). It
deliberately does **not** include the full production pipeline (vendor accelerator
drivers, internal CI/registry, cluster topology) - only the generic, reusable
mechanism, so it can be reviewed and run independently.

## What this demonstrates

| Paper layer | This repo | Tooling (all off-the-shelf) |
|---|---|---|
| OS image | `os_image/` - a real, vendor-neutral Kairos image | `kairos-init`, `podman`/`docker`, `syft`, `cosign` |
| AI model | `ai_model/` - a real small Hugging Face model, compiled through a 3-stage chain | `aibomgen-cli`, `optimum`, `onnxruntime`, `cosign` |
| Attestation & verification | both layers | `cosign attest`/`verify-attestation --policy` (CUE) |

The AI-model layer is a genuine multi-stage provenance chain, not a single artifact:

```
stage 0: source checkpoint (real HF weights)     BOM: aibomgen-cli (3rd party, HF metadata)
    │                                              │
    ▼ optimum (HF -> ONNX)                         ▼ links back via BOM hash
stage 1: ONNX export                             BOM: ours (cyclonedx-python-lib)
    │                                              │
    ▼ onnxruntime.quantization (INT8, CPU-only)    ▼ links back via BOM hash
stage 2: quantized ONNX ("deployable" artifact)  BOM: ours (cyclonedx-python-lib)
```

Each stage is pushed, signed, and attested as **its own OCI artifact** - not just extra
attestations bolted onto one final digest - so the chain is a real hash-chain of
provenance rather than a pile of unrelated documents on one node. See
[Design notes](#design-notes) for why stage 2 uses quantization rather than requiring a
reviewer to own NVIDIA JetPack/TensorRT or an Axelera accelerator.

Every step below is a call to an existing, maintained tool - see [Design notes](#design-notes)
for why nothing here reinvents SBOM scanning, OCI registry clients, or policy evaluation.

## Prerequisites

- [`podman`](https://podman.io/) or `docker` (image build/push)
- [`cosign`](https://docs.sigstore.dev/system_config/installation/) (signing, attestation, CUE policy verification)
- [`oras`](https://oras.land/docs/installation) (generic OCI artifact push/resolve)
- [`syft`](https://github.com/anchore/syft) (real CycloneDX SBOM generation for the OS image)
- [`aibomgen-cli`](https://github.com/idlab-discover/aibomgen-cli/releases) (AI-BOM generation for the source Hugging Face model)
- [`uv`](https://docs.astral.sh/uv/) (Python env/dependency management)

macOS:

```bash
brew install podman cosign oras-project/tap/oras syft
# aibomgen-cli: download the release for your platform and put it on PATH, e.g.
curl -L -o aibomgen-cli.tar.gz \
  https://github.com/idlab-discover/aibomgen-cli/releases/download/v0.2.1/aibomgen-cli_0.2.1_darwin_arm64.tar.gz
tar -xzf aibomgen-cli.tar.gz && chmod +x aibomgen-cli && sudo mv aibomgen-cli /usr/local/bin/
```

> **Port note:** the local test registry uses port 5001, not 5000 - on macOS, port 5000
> is claimed by the AirPlay Receiver by default.

## Quick start

```bash
podman run -d --name demo-registry -p 5001:5000 registry:2

uv sync --extra onnx-export   # installs optimum-onnx + onnxruntime for the AI-model demo

./os_image/demo.sh localhost:5001
uv run python -m ai_model.demo --registry localhost:5001 --model-id prajjwal1/bert-tiny
```

Each demo prints the generated BOM(s), pushes the artifact(s), signs + attests them,
verifies signature and policy at every stage (ALLOW), then tampers with the final
pushed artifact and shows verification failing (REJECT) - the same build -> sign ->
verify -> tamper -> reject loop described in the paper's demonstration section,
runnable end-to-end in well under a minute on a laptop (once dependencies are
installed - the AI-model chain above took ~50s total on a base MacBook, including
downloading real Hugging Face weights).

## Layout

```
os_image/
  Containerfile   # real kairos-init build (init stage only - see file header for why)
  policy.cue      # CUE policy evaluated by `cosign verify-attestation --policy`
  demo.sh         # build -> syft SBOM -> sign -> attest -> verify -> tamper -> reject
ai_model/
  aibom.py            # CycloneDX AI-BOM builder, generic across compilation stages
                       # (the one piece of custom code - see Design notes)
  signing.py          # thin cosign CLI wrapper
  policy.cue          # policy for our own BOMs (ONNX + quantized stages): requires a
                       # "Source Model AI-BOM SHA-256" link back to the previous stage
  source_policy.cue   # policy for aibomgen-cli's source-model AI-BOM: requires its
                       # embedded Hugging Face security scan to say "safe"
  demo.py             # the full 3-stage chain: aibomgen-cli -> optimum -> onnxruntime
                       # -> sign/attest every stage -> verify chain -> tamper -> reject
```

## Design notes

- **OS image**: uses the real `kairos-init` tool (the same one production Kairos builds
  use), but only its lightweight `init` stage - not the ~100-package `install` stage,
  kernel, grub, or k0s provider. That's what makes production Kairos builds take
  several minutes and sensitive to the local apt mirror; skipping it keeps this demo to
  a few seconds while still producing a genuinely Kairos-branded, signable, scannable
  OCI image. See the comment block in `os_image/Containerfile` for how to opt into the
  full production-shaped build instead.
- **SBOM generation**: real `syft` output, not a hand-rolled scanner - the OS-image BOM
  reflects the packages actually in the image.
- **AI-BOM for the source model**: generated by
  [`aibomgen-cli`](https://github.com/idlab-discover/aibomgen-cli) (the tool behind
  Vandendriessche et al.'s automated, signed AI-BOM work cited in the paper's related
  work), which queries the Hugging Face Hub for license, model card, and security-scan
  metadata - not reimplemented here.
- **AI-BOM for the compiled stages** (`ai_model/aibom.py`): this is the one piece of
  genuinely custom code in this repo. There is no off-the-shelf tool that inspects a
  compiled model artifact and tells you which input it came from, which compiler
  produced it, or with what parameters - that provenance has to be recorded at compile
  time. It's built with the `cyclonedx-python-lib` object model (not hand-rolled JSON),
  and each stage's BOM records the *previous* stage's BOM hash, chaining the final
  artifact all the way back to the audited source model.
- **Why quantization instead of TensorRT/Axelera Metis**: the paper's production
  pipeline compiles ONNX to a device-specific engine (TensorRT for NVIDIA GPUs, Metis
  compiler output for Axelera NPUs). Neither is something a reviewer can be expected to
  set up - TensorRT needs an NVIDIA GPU + CUDA + JetPack, Metis needs an Axelera
  accelerator + proprietary SDK. `onnxruntime.quantization`'s dynamic INT8 quantization
  is a real, meaningful, CPU-only stand-in for that step: it measurably shrinks the
  model (bert-tiny's ONNX export goes from ~17.5MB to ~4.5MB) and captures exactly the
  "quantization choices" concern the paper's AI-BOM section calls out - without any
  special hardware. Swap in a real device compiler at this stage if you have the
  hardware; the BOM/sign/attest/verify mechanism is identical either way.
- **Compilation steps are real**: `optimum-cli export onnx` (HF checkpoint -> ONNX) and
  `onnxruntime.quantization.quantize_dynamic` (ONNX -> INT8) are actual conversions, not
  stand-ins with random bytes. Swap `--model-id` for any Hugging Face model to see the
  same chain with a different source.
- **Policy gate**: `cosign verify-attestation --policy <file>.cue` evaluates each
  attestation with a real CUE policy - this stands in for the paper's GitOps
  admission-time verification step, without needing a live Flux/Kubernetes cluster. The
  two policies check different, meaningful things (see `ai_model/*.cue` above), not
  just document shape. `cosign attest` also already wraps every predicate in a genuine
  `https://in-toto.io/Statement/v0.1` envelope by default.

## What this is *not*

- Not the full production pipeline: no accelerator vendor drivers (NVIDIA JetPack,
  Axelera SDK), no internal CI/registry, no cluster topology, no SLSA provenance
  predicate generation.
- Not a benchmark of model quality or inference performance - the compilation chain
  exists purely to produce derived artifacts whose provenance can be recorded and
  verified, not to optimize for accuracy or latency.

## License

Apache-2.0, see [LICENSE](LICENSE).


