"""
streamlit_app.py
────────────────────────────────────────────────────────────────────────────
IEEE-CIS Fraud Detection — Streamlit front-end.

GitHub repo layout assumed:
  Portfolio/
    streamlit_app.py        ← this file
    X_train.csv             ← background dataset for SHAP TreeExplainer
  src/
    Custom_Classes.py
    feature_utils.py

The app:
  • Accepts the 4 post-pipeline features as user inputs.
  • Sends them to the SageMaker endpoint (returns "Legitimate"/"Fraudulent").
  • Builds a local SHAP waterfall plot using X_train.csv as the background.
────────────────────────────────────────────────────────────────────────────
"""

import os
import sys
import streamlit as st
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from joblib import load

# ── Path setup so src/ classes are importable ──────────────────────────────
# Works both locally (Portfolio/ is cwd) and on Streamlit Cloud
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root
SRC_DIR  = os.path.join(ROOT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from Custom_Classes import FeatureSelector, AutoPowerTransformer  # noqa: F401

# ── SageMaker endpoint (update name if yours differs) ──────────────────────
ENDPOINT_NAME = st.secrets["aws_credentials"]["AWS_ENDPOINT"]

# ── Local artefact paths (relative to this file's directory = Portfolio/) ──
PORTFOLIO_DIR  = os.path.dirname(os.path.abspath(__file__))
PIPELINE_PATH  = os.path.join(PORTFOLIO_DIR, "finalized_fraud_model.joblib")
X_TRAIN_PATH   = os.path.join(PORTFOLIO_DIR, "X_train.csv")

# ── The 4 post-pipeline feature columns expected by the endpoint ────────────
FEATURE_COLS = [
    "cat__card6_credit",
    "remainder__V317",
    "cat__card6_debit",
    "remainder__V312",
]

# ───────────────────────────────────────────────────────────────────────────
# Page config
# ───────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Fraud Detector", page_icon="🔍", layout="wide")
st.title("🔍 IEEE-CIS Fraud Detection — Real-Time Inference")
st.markdown("Adjust the feature values in the sidebar, then click **Predict**.")

# ───────────────────────────────────────────────────────────────────────────
# Sidebar — user inputs for the 4 features
# ───────────────────────────────────────────────────────────────────────────
st.sidebar.header("Transaction Features")

card6_credit = st.sidebar.slider(
    "cat__card6_credit  (one-hot: 1 = credit card)",
    min_value=0.0, max_value=1.0, value=1.0, step=1.0
)
v317 = st.sidebar.number_input(
    "remainder__V317",
    value=0.0, format="%.4f"
)
card6_debit = st.sidebar.slider(
    "cat__card6_debit  (one-hot: 1 = debit card)",
    min_value=0.0, max_value=1.0, value=0.0, step=1.0
)
v312 = st.sidebar.number_input(
    "remainder__V312",
    value=0.0, format="%.4f"
)

# Build a single-row DataFrame with the correct column names
input_df = pd.DataFrame([{
    "cat__card6_credit": card6_credit,
    "remainder__V317":   v317,
    "cat__card6_debit":  card6_debit,
    "remainder__V312":   v312,
}])

st.write("### Input Features")
st.dataframe(input_df)

# ───────────────────────────────────────────────────────────────────────────
# Prediction
# ───────────────────────────────────────────────────────────────────────────
if st.button("🔎 Predict"):

    # ── Call the SageMaker endpoint ──────────────────────────────────────
    try:
        import boto3, json

        runtime = boto3.client(
            "sagemaker-runtime",
            region_name=st.secrets["aws_credentials"]["AWS_DEFAULT_REGION"],
            aws_access_key_id=st.secrets["aws_credentials"]["AWS_ACCESS_KEY_ID"],
            aws_secret_access_key=st.secrets["aws_credentials"]["AWS_SECRET_ACCESS_KEY"],
            aws_session_token=st.secrets["aws_credentials"]["AWS_SESSION_TOKEN"],
        )
        payload = input_df.to_json(orient="columns")   # same as JSONSerializer

        response = runtime.invoke_endpoint(
            EndpointName=ENDPOINT_NAME,
            ContentType="application/json",
            Body=payload,
        )
        result = json.loads(response["Body"].read().decode("utf-8"))

        # Endpoint returns {"prediction": ["Legitimate"]} or ["Fraudulent"]
        label = result["prediction"][0] if isinstance(result["prediction"], list) \
                else result["prediction"]

    except Exception as ex:
        st.error(f"❌ Endpoint call failed: {ex}")
        st.stop()

    # ── Display result ───────────────────────────────────────────────────
    if label == "Fraudulent":
        st.error("🚨 **FRAUDULENT** — This transaction is predicted as fraudulent.")
    else:
        st.success("✅ **LEGITIMATE** — This transaction appears legitimate.")

    st.markdown(f"**Prediction:** `{label}`")

    # ── SHAP waterfall plot ──────────────────────────────────────────────
    st.write("### Local SHAP Explanation")
    try:
        # Load the full pipeline to reach the RandomForest step
        pipeline = load(PIPELINE_PATH)

        # The pipeline's final estimator must be named 'model' in your Pipeline steps
        rf_model = pipeline.named_steps["model"]

        # Load X_train background data (already transformed — post-pipeline columns)
        X_train_bg = pd.read_csv(X_TRAIN_PATH, usecols=FEATURE_COLS)

        # Build TreeExplainer with background data for a more stable baseline
        explainer = shap.TreeExplainer(rf_model, data=X_train_bg)

        # SHAP values for the single input row
        shap_values = explainer(input_df)

        # For RandomForestClassifier shap_values shape is (1, n_features, 2)
        # Slice to class 1 (fraud)
        if shap_values.values.ndim == 3:
            sv_fraud = shap.Explanation(
                values          = shap_values.values[0, :, 1],
                base_values     = shap_values.base_values[0, 1],
                data            = shap_values.data[0],
                feature_names   = FEATURE_COLS,
            )
        else:
            # Binary output — single set of SHAP values
            sv_fraud = shap_values[0]

        fig, ax = plt.subplots()
        shap.waterfall_plot(sv_fraud, max_display=10, show=False)
        st.pyplot(fig)
        plt.close(fig)

    except Exception as ex:
        st.warning(f"⚠️ SHAP explanation unavailable: {ex}")

st.caption(
    "Powered by AWS SageMaker + Random Forest Pipeline | IEEE-CIS Fraud Detection"
)
