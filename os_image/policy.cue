// Policy gate for the OS-image layer, evaluated by
// `cosign verify-attestation --policy` against the Syft-generated SBOM.
//
// Asserts the attached document is a CycloneDX SBOM describing a container
// image with at least one discovered component.

import "list"

predicateType: "https://cyclonedx.org/bom"

predicate: {
	bomFormat: "CycloneDX"
	metadata: component: type: "container"
	components: list.MinItems(1)
}

