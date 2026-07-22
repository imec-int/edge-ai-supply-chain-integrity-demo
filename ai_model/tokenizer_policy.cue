// Policy gate for the tokenizer AI-BOM, evaluated by `cosign verify-attestation --policy`.
//
// The tokenizer (vocab/merges/config files) is fetched once from the same Hugging
// Face repo as the source checkpoint, hashed, and chained back to the source
// AI-BOM - see ai_model/demo.py's `fetch_tokenizer` / tokenizer stage. This asserts
// the attached document is a well-formed CycloneDX AI-BOM describing it as a
// "data" component (as opposed to the model-weight stages, which are
// "machine-learning-model").

predicateType: "https://cyclonedx.org/bom"

predicate: {
	bomFormat:   "CycloneDX"
	specVersion: "1.6"
	metadata: component: type: "data"
}

