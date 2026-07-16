#!/usr/bin/env bash
# OS-layer demo: build a real Kairos image -> generate an SBOM (Syft) -> sign + attest
# (Cosign) -> push -> verify (signature + CUE policy) -> tamper -> reject.
#
# Every step below is an existing, off-the-shelf tool - nothing here is custom code:
#   - kairos-init / k0s provider  -> the actual Kairos build tooling
#   - podman (or docker) build/push -> OCI image build/push
#   - syft                        -> real CycloneDX SBOM generation (scans actual packages)
#   - cosign                      -> signing, attestation, verification, CUE policy gate
#   - oras                        -> generic OCI digest resolution
#
# Usage: ./demo.sh [registry] (defaults to localhost:5000, see ../README.md to start one)
set -euo pipefail

REGISTRY="${1:-localhost:5000}"
REPO="supply-chain-demo/kairos-os"
TAG="1.0.0"
TAG_REF="${REGISTRY}/${REPO}:${TAG}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK_DIR="$(mktemp -d)"
trap 'rm -rf "$WORK_DIR"' EXIT

CONTAINER_BIN="${CONTAINER_BIN:-podman}"

for bin in "$CONTAINER_BIN" cosign oras syft; do
  command -v "$bin" >/dev/null || { echo "error: '$bin' not found on PATH" >&2; exit 1; }
done

echo "== 1. Build the vendor-neutral Kairos OS image =="
echo "   (real kairos-init tooling, lightweight 'init' stage only - a few seconds,"
echo "    see Containerfile for the full production-shaped build as an alternative)"
"$CONTAINER_BIN" build -t "$TAG_REF" -f "$SCRIPT_DIR/Containerfile" "$SCRIPT_DIR"

echo "== 2. Push it to the registry =="
"$CONTAINER_BIN" push --tls-verify=false "$TAG_REF"
DIGEST="$(oras resolve --plain-http "$TAG_REF")"
REF="${REGISTRY}/${REPO}@${DIGEST}"
echo "   pushed $REF"

echo "== 3. Generate a real SBOM with Syft =="
syft "registry:${REF}" -o "cyclonedx-json=${WORK_DIR}/sbom.cdx.json"
COMPONENT_COUNT=$(python3 -c "import json;print(len(json.load(open('${WORK_DIR}/sbom.cdx.json')).get('components',[])))")
echo "   syft found ${COMPONENT_COUNT} components"

echo "== 4. Sign the image + attest the SBOM (cosign) =="
COSIGN_PASSWORD="" cosign generate-key-pair --output-key-prefix "${WORK_DIR}/cosign" >/dev/null
COSIGN_PASSWORD="" cosign sign --yes --key "${WORK_DIR}/cosign.key" --allow-http-registry "$REF"
COSIGN_PASSWORD="" cosign attest --yes --key "${WORK_DIR}/cosign.key" \
  --predicate "${WORK_DIR}/sbom.cdx.json" --type cyclonedx --allow-http-registry "$REF"

echo "== 5. Verify: signature + CUE policy over the attestation (both native cosign features) =="
if cosign verify --key "${WORK_DIR}/cosign.pub" --allow-http-registry "$REF" >/dev/null 2>&1 \
  && cosign verify-attestation --key "${WORK_DIR}/cosign.pub" --type cyclonedx \
     --allow-http-registry --policy "$SCRIPT_DIR/policy.cue" "$REF" >/dev/null 2>&1; then
  echo "   ALLOW: deploy"
else
  echo "   ERROR: verification failed on a freshly signed image" >&2
  exit 1
fi

echo "== 6. Tamper: rebuild FROM the signed image, push under the same tag =="
cat > "${WORK_DIR}/Containerfile.tamper" <<EOF
FROM ${TAG_REF}
RUN echo "attacker-was-here" > /etc/tamper-marker
EOF
"$CONTAINER_BIN" build -t "$TAG_REF" -f "${WORK_DIR}/Containerfile.tamper" "${WORK_DIR}"
"$CONTAINER_BIN" push --tls-verify=false "$TAG_REF"
TAMPERED_DIGEST="$(oras resolve --plain-http "$TAG_REF")"
TAMPERED_REF="${REGISTRY}/${REPO}@${TAMPERED_DIGEST}"
echo "   tampered ref: $TAMPERED_REF (original: $REF)"

echo "== 7. Verify the tampered image (expect rejection) =="
if cosign verify --key "${WORK_DIR}/cosign.pub" --allow-http-registry "$TAMPERED_REF" >/dev/null 2>&1; then
  echo "   ERROR: tampered image verified (unexpected!)" >&2
  exit 1
else
  echo "   REJECT: tampered image has no valid signature, as expected"
fi


