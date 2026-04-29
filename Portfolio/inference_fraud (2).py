"""
inference_fraud.py
──────────────────────────────────────────────────────────────────────────────
SageMaker inference script for the IEEE-CIS Fraud Detection pipeline.

Key changes:
  1. Loads 'finalized_fraud_model.joblib' (Random Forest fraud pipeline).
  2. input_fn supports 'application/json' via pd.read_json — preserves
     column names when the client uses JSONSerializer.
  3. predict_fn returns ONLY hard label strings: "Legitimate" or "Fraudulent".
  4. output_fn serialises the result as a JSON dict:
       {"prediction": "Fraudulent"}
  5. gensim/spacy are NOT imported at the top level — the lazy-import fix
     in Custom_Classes.py means the pipeline unpickles without them.

File layout expected inside the SageMaker model tarball
(mirrors your JupyterLab "Final Project RF" folder):
  .
  ├── finalized_fraud_model.joblib   ← saved pipeline
  ├── inference_fraud.py             ← this file  (entry_point)
  └── src/
      └── Custom_Classes.py          ← FeatureSelector, AutoPowerTransformer …
──────────────────────────────────────────────────────────────────────────────
"""

import joblib
import os
import sys
import json
import numpy as np
import pandas as pd
from io import BytesIO, StringIO


# ── Make sure the model directory is on the path so that
#    src/Custom_Classes.py can be imported when the pipeline is unpickled.
model_dir = os.environ.get('SM_MODEL_DIR', '.')

if model_dir not in sys.path:
    sys.path.insert(0, model_dir)

# src/ is bundled inside the tarball at the root level
from src.Custom_Classes import FeatureSelector, AutoPowerTransformer   # noqa: F401


# ─────────────────────────────────────────────────────────────────────────────
# model_fn
# ─────────────────────────────────────────────────────────────────────────────

def model_fn(model_dir: str):
    """Load the serialised imblearn/sklearn Pipeline from disk."""
    path = os.path.join(model_dir, 'finalized_fraud_model.joblib')
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Model artefact not found at '{path}'. "
            "Ensure 'finalized_fraud_model.joblib' is included in the model tarball."
        )
    pipeline = joblib.load(path)
    print(f"[model_fn] Pipeline loaded successfully from {path}")
    return pipeline


# ─────────────────────────────────────────────────────────────────────────────
# input_fn
# ─────────────────────────────────────────────────────────────────────────────

def input_fn(request_body, request_content_type: str) -> pd.DataFrame:
    """Deserialise the incoming request into a pandas DataFrame."""
    print(f"[input_fn] Received content-type: {request_content_type}")

    if request_content_type == 'application/json':
        if isinstance(request_body, bytes):
            request_body = request_body.decode('utf-8')
        df = pd.read_json(StringIO(request_body))
        print(f"[input_fn] Parsed JSON — shape: {df.shape}, columns: {list(df.columns[:5])}...")
        return df

    elif request_content_type == 'text/csv':
        if isinstance(request_body, bytes):
            request_body = request_body.decode('utf-8')
        df = pd.read_csv(StringIO(request_body))
        print(f"[input_fn] Parsed CSV  — shape: {df.shape}")
        return df

    elif request_content_type == 'application/x-npy':
        data = np.load(BytesIO(request_body), allow_pickle=True)
        df   = pd.DataFrame(data)
        print(f"[input_fn] Parsed NPY  — shape: {df.shape}")
        return df

    else:
        raise ValueError(
            f"Unsupported content type: '{request_content_type}'. "
            "Use 'application/json' (recommended), 'text/csv', or 'application/x-npy'."
        )


# ─────────────────────────────────────────────────────────────────────────────
# predict_fn
# ─────────────────────────────────────────────────────────────────────────────

def predict_fn(input_df: pd.DataFrame, model):
    """
    Run the full imblearn/sklearn Pipeline on the input DataFrame.
    Returns a list of human-readable label strings per row.
      0  →  "Legitimate"
      1  →  "Fraudulent"
    """
    print(f"[predict_fn] Running pipeline on {len(input_df)} row(s)...")

    raw_preds = model.predict(input_df)

    label_map = {0: "Legitimate", 1: "Fraudulent"}
    labels = [label_map[int(p)] for p in raw_preds]

    print(f"[predict_fn] Labels: {labels}")
    return {"prediction": labels}


# ─────────────────────────────────────────────────────────────────────────────
# output_fn
# ─────────────────────────────────────────────────────────────────────────────

def output_fn(prediction, content_type: str):
    """
    Serialise predictions back to the client as a JSON dict.

    Example response body (single row):
        {"prediction": ["Legitimate"]}

    Example response body (batch):
        {"prediction": ["Fraudulent", "Legitimate", "Legitimate"]}
    """
    print(f"[output_fn] Formatting output for content-type: {content_type}")
    return json.dumps(prediction), "application/json"
