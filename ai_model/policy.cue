// Policy gate for the compiled-model AI-BOM, evaluated by cosign verify-attestation --policy`.
// Asserts the document is a well-formed CycloneDX AI-BOM describing a machine-learning-model component.

predicateType: "https://cyclonedx.org/bom"

predicate: {
	bomFormat:   "CycloneDX"
	specVersion: "1.6"
	metadata: component: type: "machine-learning-model"
}
