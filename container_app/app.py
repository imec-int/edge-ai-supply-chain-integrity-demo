"""
Sample application workload for the 'Container & Application Layer'.
"""

import os
import sys
import traceback
from pathlib import Path
from typing import Any

from flask import Flask, jsonify, request

app = Flask(__name__)

MODEL_PATH = Path(os.environ.get("MODEL_PATH", "/app/model/model.onnx"))
TOKENIZER_PATH = Path(os.environ.get("TOKENIZER_PATH", "/app/model/tokenizer"))
MODEL_ID = os.environ.get("MODEL_ID", "prajjwal1/bert-tiny")  # metadata only, see _load()

_session: Any = None
_tokenizer: Any = None
_load_error: str | None = None


def _load() -> None:
    """
    Load the baked-in ONNX model + tokenizer once, before the server starts
    accepting requests.

    Both are verified, signed, and attested artifacts baked into this image at
    build time (see container_app/demo.py and ai_model/demo.py's tokenizer
    stage) - nothing is fetched from the network at runtime, so MODEL_ID above
    is informational only.

    Only basic text encoder models are supported.
    """
    global _session, _tokenizer, _load_error
    if _session is not None or _load_error is not None:
        return
    try:
        if not MODEL_PATH.is_file():
            raise RuntimeError(f"no verified model found at {MODEL_PATH}")
        if not TOKENIZER_PATH.is_dir():
            raise RuntimeError(f"no verified tokenizer found at {TOKENIZER_PATH}")
        import onnxruntime as ort
        from transformers import AutoTokenizer

        session = ort.InferenceSession(str(MODEL_PATH), providers=["CPUExecutionProvider"])
        tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER_PATH))
        _session, _tokenizer = session, tokenizer
        print(
            f"[app] model + tokenizer loaded OK from {MODEL_PATH} / {TOKENIZER_PATH}", flush=True
        )
    except Exception as exc:  # noqa: BLE001 - surfaced via /healthz and /predict
        _load_error = f"{exc}\n{traceback.format_exc()}"
        print(
            f"[app] FAILED to load model/tokenizer ({MODEL_ID}) from "
            f"{MODEL_PATH} / {TOKENIZER_PATH}: {exc}"
            f"\n(only simple text encoder models are supported)\n",
            file=sys.stderr, flush=True,
        )


@app.errorhandler(Exception)
def handle_uncaught(exc: Exception):
    return jsonify(error=str(exc), traceback=traceback.format_exc()), 500


@app.get("/healthz")
def healthz():
    _load()
    if _load_error is not None:
        return jsonify(
            status="degraded", model_path=str(MODEL_PATH),
            tokenizer_path=str(TOKENIZER_PATH), error=_load_error,
        ), 503
    return jsonify(
        status="ok", model_path=str(MODEL_PATH),
        tokenizer_path=str(TOKENIZER_PATH), model_id=MODEL_ID,
    )


@app.post("/predict")
def predict():
    _load()
    if _load_error is not None:
        return jsonify(error=f"model unavailable: {_load_error}"), 503
    if _session is None or _tokenizer is None:
        return jsonify(error="model not loaded yet"), 503

    body = request.get_json(silent=True) or {}
    text = body.get("text", "Supply-chain integrity for edge AI.")

    inputs = _tokenizer(text, return_tensors="np")
    input_names = {i.name for i in _session.get_inputs()}
    ort_inputs = {name: value for name, value in inputs.items() if name in input_names}
    missing = input_names - ort_inputs.keys()
    if missing:
        return jsonify(
            error=f"tokenizer did not produce required model input(s): {sorted(missing)}",
            tokenizer_outputs=sorted(inputs.keys()),
            model_inputs=sorted(input_names),
        ), 500

    outputs = _session.run(None, ort_inputs)
    last_hidden = outputs[0]

    return jsonify(
        input_text=text,
        model_path=str(MODEL_PATH),
        output_shape=list(last_hidden.shape),
        output_mean=float(last_hidden.mean()),
        output_l2_norm=float((last_hidden.astype("float64") ** 2).sum() ** 0.5),
    )


if __name__ == "__main__":
    _load()  # load once, synchronously, before accepting any connections
    app.run(host="0.0.0.0", port=8080)
