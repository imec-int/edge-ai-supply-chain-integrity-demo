// Policy gate for the container/application layer, evaluated by
// `cosign verify-attestation --policy` against the Syft-generated SBOM.
//
// Asserts the attached document is a CycloneDX SBOM describing a container
// image with at least one discovered component (e.g. the Flask dependency
// chain baked into the sample app).

import "list"

predicateType: "https://cyclonedx.org/bom"

predicate: {
	bomFormat: "CycloneDX"
	metadata: component: type: "container"
	components: list.MinItems(1)
}
