// GitOps-style policy gate for the OS-image layer, evaluated natively by
// `cosign verify-attestation --policy` (CUE) against the Syft-generated SBOM.
//
// Kept minimal: asserts the attached document is a real CycloneDX SBOM describing a
// container/OS image with at least one discovered component (i.e. Syft actually found
// packages, rather than an empty/placeholder BOM being smuggled through).

import "list"

predicateType: "https://cyclonedx.org/bom"

predicate: {
	bomFormat: "CycloneDX"
	metadata: component: type: "container"
	components: list.MinItems(1)
}


