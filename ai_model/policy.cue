// GitOps-style policy gate for the AI-model layer, evaluated natively by
// `cosign verify-attestation --policy` (CUE), not custom application code.
//
// Kept intentionally minimal for demo purposes: it asserts the document is a
// well-formed CycloneDX AI-BOM describing a machine-learning-model component.
// A production policy would extend this with an allow-list of compilers/
// target devices, required license fields, etc. - see
// https://docs.sigstore.dev/policy-controller/overview/ for richer examples.

predicateType: "https://cyclonedx.org/bom"

predicate: {
	bomFormat:   "CycloneDX"
	specVersion: "1.6"
	metadata: component: type: "machine-learning-model"
}

