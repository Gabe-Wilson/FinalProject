"""
streamlit_appFINAL.py
────────────────────────────────────────────────────────────────────────────
IEEE-CIS Fraud Detection — Streamlit front-end.

KEY FIX: The pipeline's first step is FeatureSelector, which was fitted on
the full raw training dataset (~90+ columns).  Sending only 3 columns to
the endpoint causes a KeyError inside FeatureSelector.transform().

Solution: build a full-width raw row with median/mode defaults for every
column the pipeline was trained on, then overwrite the 3 user-controlled
features before sending.  The pipeline's own FeatureSelector + downstream
steps reduce those ~90 columns down to the final features automatically.

GitHub repo layout:
  Portfolio/
    streamlit_appFINAL.py   <- this file
    X_train.csv             <- background dataset (post-pipeline columns for SHAP)
    finalized_fraud_model.joblib
  src/
    Custom_Classes.py
"""

import os
import sys
import traceback
import types
import importlib
import streamlit as st
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from joblib import load

# ── Path setup so src/ classes are importable ──────────────────────────────
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR  = os.path.join(ROOT_DIR, "src")
for _p in [ROOT_DIR, SRC_DIR]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Make 'src' importable as a package so the pickle's stored reference
# to 'src.Custom_Classes' resolves correctly on Streamlit Cloud.
if "src" not in sys.modules:
    src_mod = types.ModuleType("src")
    src_mod.__path__ = [SRC_DIR]
    src_mod.__package__ = "src"
    sys.modules["src"] = src_mod
if "src.Custom_Classes" not in sys.modules:
    sys.modules["src.Custom_Classes"] = importlib.import_module("Custom_Classes")

from Custom_Classes import FeatureSelector, AutoPowerTransformer  # noqa: F401

# ── Config ─────────────────────────────────────────────────────────────────
ENDPOINT_NAME  = st.secrets["aws_credentials"]["AWS_ENDPOINT"]
PORTFOLIO_DIR  = os.path.dirname(os.path.abspath(__file__))
PIPELINE_PATH  = os.path.join(PORTFOLIO_DIR, "finalized_fraud_model.joblib")
X_TRAIN_PATH   = os.path.join(PORTFOLIO_DIR, "X_train.csv")

USER_FEATURES = ["card6", "V317", "V312"]

RAW_FEATURE_DEFAULTS = {
    "Unnamed: 0": 0,
    "card1": 9500, "card3": 150.0, "card5": 226.0,
    "addr1": 299.0, "addr2": 87.0,
    "dist1": 0.0,   "dist2": 0.0,
    "C5": 0.0, "C7": 0.0, "C8": 1.0, "C9": 1.0,
    "C10": 0.0, "C12": 0.0, "C13": 1.0,
    "D1": 0.0, "D10": 0.0,
    "V14": 0.0, "V15": 0.0, "V16": 0.0, "V17": 0.0, "V18": 0.0,
    "V21": 0.0, "V22": 0.0, "V23": 0.0, "V24": 0.0, "V25": 0.0,
    "V26": 0.0, "V29": 0.0, "V30": 0.0, "V31": 0.0, "V32": 0.0,
    "V33": 0.0, "V34": 0.0,
    "V55": 0.0, "V56": 0.0, "V57": 0.0, "V58": 0.0, "V59": 0.0,
    "V60": 0.0, "V63": 0.0, "V64": 0.0,
    "V69": 0.0, "V70": 0.0, "V71": 0.0, "V72": 0.0, "V73": 0.0,
    "V74": 0.0,
    "V95": 0.0, "V101": 0.0,
    "V108": 0.0, "V109": 0.0, "V110": 0.0, "V111": 0.0, "V112": 0.0,
    "V113": 0.0, "V114": 0.0, "V115": 0.0, "V116": 0.0, "V117": 0.0,
    "V118": 0.0, "V119": 0.0, "V120": 0.0, "V121": 0.0, "V122": 0.0,
    "V123": 0.0, "V124": 0.0, "V125": 0.0, "V126": 0.0, "V129": 0.0,
    "V132": 0.0,
    "V279": 0.0, "V280": 0.0, "V281": 0.0, "V282": 0.0, "V284": 0.0,
    "V287": 0.0, "V290": 0.0, "V291": 0.0, "V292": 0.0, "V293": 0.0,
    "V294": 0.0, "V295": 0.0, "V296": 0.0, "V298": 0.0, "V299": 0.0,
    "V302": 0.0, "V303": 0.0, "V304": 0.0, "V307": 0.0, "V308": 0.0,
    "V309": 0.0,
    "V312": 0.0, "V317": 0.0,
    "V318": 0.0, "V320": 0.0,
    "ProductCD": "W",
    "card4": "visa",
    "card6": "debit",
    "M6": "T",
}

# ───────────────────────────────────────────────────────────────────────────
# Page config
# ───────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Fraud Detector", page_icon="🔍", layout="wide")
st.title("🔍 IEEE-CIS Fraud Detection — Real-Time Inference")
st.markdown(
    "Adjust the three most influential transaction features in the sidebar, "
    "then click **Predict**. All other features are held at their training-set medians."
)

# ───────────────────────────────────────────────────────────────────────────
# Sidebar
# ───────────────────────────────────────────────────────────────────────────
st.sidebar.header("Transaction Features")

card6 = st.sidebar.selectbox(
    "card6  (card type)",
    ["debit", "credit"],
    help="Most influential feature. Credit cards have a higher fraud rate in this dataset."
)
v317 = st.sidebar.number_input(
    "V317  (Vesta engineered feature)",
    value=0.0, format="%.4f",
    help="Second most important feature. Typical legitimate range: 0 - 50."
)
v312 = st.sidebar.number_input(
    "V312  (Vesta engineered feature)",
    value=0.0, format="%.4f",
    help="Third most important feature. Typical legitimate range: 0 - 20."
)

raw_row = dict(RAW_FEATURE_DEFAULTS)
raw_row["card6"] = card6
raw_row["V317"]  = v317
raw_row["V312"]  = v312

input_df = pd.DataFrame([raw_row])

st.write("### User-Controlled Input Features")
st.dataframe(input_df[USER_FEATURES])

with st.expander("Show full raw payload sent to the endpoint"):
    st.dataframe(input_df)

# ───────────────────────────────────────────────────────────────────────────
# Prediction
# ───────────────────────────────────────────────────────────────────────────
if st.button("🔎 Predict"):

    try:
        import boto3, json

        runtime = boto3.client(
            "sagemaker-runtime",
            region_name=st.secrets["aws_credentials"]["AWS_DEFAULT_REGION"],
            aws_access_key_id=st.secrets["aws_credentials"]["AWS_ACCESS_KEY_ID"],
            aws_secret_access_key=st.secrets["aws_credentials"]["AWS_SECRET_ACCESS_KEY"],
            aws_session_token=st.secrets["aws_credentials"]["AWS_SESSION_TOKEN"],
        )

        payload  = input_df.to_json(orient="columns")
        response = runtime.invoke_endpoint(
            EndpointName=ENDPOINT_NAME,
            ContentType="application/json",
            Body=payload,
        )
        result = json.loads(response["Body"].read().decode("utf-8"))
        label  = (
            result["prediction"][0]
            if isinstance(result["prediction"], list)
            else result["prediction"]
        )

    except Exception as ex:
        st.error(f"Endpoint call failed: {ex}")
        st.stop()

    if label == "Fraudulent":
        st.error("🚨 **FRAUDULENT** — This transaction is predicted as fraudulent.")
    else:
        st.success("✅ **LEGITIMATE** — This transaction appears legitimate.")
    st.markdown(f"**Prediction:** `{label}`")

    # ── SHAP waterfall plot ──────────────────────────────────────────────
    st.write("### Local SHAP Explanation")
    try:
        pipeline = load(PIPELINE_PATH)
        rf_model = pipeline.named_steps["model"]
        n_features = rf_model.n_features_in_

        # Transform raw input through all pre-model steps (skip sampler + model)
        from sklearn.pipeline import Pipeline as SKPipeline
        pre_steps = [
            (name, pipeline.named_steps[name])
            for name, _ in pipeline.steps
            if name not in ("sampler", "model")
        ]
        pre_pipe      = SKPipeline(pre_steps)
        X_transformed = pre_pipe.transform(input_df)
        X_arr         = np.array(X_transformed, dtype=np.float64)

        # Load background data
        X_train_bg = pd.read_csv(X_TRAIN_PATH)
        bg_arr     = np.array(X_train_bg, dtype=np.float64)

        # Guarantee both arrays have exactly n_features columns
        def _align(arr, n):
            if arr.shape[1] == n:
                return arr
            if arr.shape[1] > n:
                return arr[:, :n]
            pad = np.zeros((arr.shape[0], n - arr.shape[1]))
            return np.hstack([arr, pad])

        X_arr  = _align(X_arr,  n_features)
        bg_arr = _align(bg_arr, n_features)

        # Feature names: use X_train.csv column names when count matches
        feat_names = (
            X_train_bg.columns.tolist()
            if X_train_bg.shape[1] == n_features
            else [f"f{i}" for i in range(n_features)]
        )

        explainer   = shap.TreeExplainer(rf_model, data=bg_arr)
        shap_values = explainer(X_arr)

        # RF returns shape (1, n_features, 2) — slice to class 1 (Fraud)
        if shap_values.values.ndim == 3:
            sv_fraud = shap.Explanation(
                values        = shap_values.values[0, :, 1],
                base_values   = shap_values.base_values[0, 1],
                data          = shap_values.data[0],
                feature_names = feat_names,
            )
        else:
            sv_fraud = shap_values[0]

        fig, ax = plt.subplots()
        shap.waterfall_plot(sv_fraud, max_display=10, show=False)
        st.pyplot(fig)
        plt.close(fig)

    except Exception as ex:
        st.warning(f"⚠️ SHAP explanation unavailable: {ex}")
        st.code(traceback.format_exc())   # full stack trace so you can see exactly what failed

st.caption(
    "Powered by AWS SageMaker + Random Forest Pipeline | IEEE-CIS Fraud Detection"
)
