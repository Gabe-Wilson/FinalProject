import streamlit as st
import pandas as pd
import numpy as np
import json, boto3, shap, matplotlib.pyplot as plt
from joblib import load
from sagemaker.predictor import Predictor
from sagemaker.serializers import JSONSerializer
from sagemaker.deserializers import JSONDeserializer
from sklearn.pipeline import Pipeline as SKPipeline

# ── Config ────────────────────────────────────────────────────
ENDPOINT_NAME  = "fraud-rf-json-endpoint"
PIPELINE_PATH  = "finalized_fraud_model.joblib"

st.set_page_config(page_title="Fraud Detector", page_icon="🔍", layout="wide")
st.title("🔍 IEEE-CIS Fraud Detection — Real-Time Inference")
st.markdown("Enter transaction details in the sidebar, then click **Predict**.")

# ── Sidebar inputs ────────────────────────────────────────────
st.sidebar.header("Transaction Features")
transaction_amt = st.sidebar.number_input("TransactionAmt ($)", min_value=0.0, value=50.0, step=1.0)
product_cd      = st.sidebar.selectbox("ProductCD", ["W", "H", "C", "S", "R"])
card4           = st.sidebar.selectbox("card4 (network)", ["visa", "mastercard", "discover", "american express"])
addr1           = st.sidebar.number_input("addr1", min_value=0, value=200, step=1)
dist1           = st.sidebar.number_input("dist1", min_value=0.0, value=0.0)

input_df = pd.DataFrame([{
    "TransactionAmt": transaction_amt,
    "ProductCD"     : product_cd,
    "card4"         : card4,
    "addr1"         : addr1,
    "dist1"         : dist1,
}])

st.write("### Input Data")
st.dataframe(input_df)

# ── Prediction ────────────────────────────────────────────────
if st.button("🔎 Predict"):
    try:
        pred_client = Predictor(
            endpoint_name=ENDPOINT_NAME,
            serializer=JSONSerializer(),
            deserializer=JSONDeserializer()
        )

        # Convert DataFrame → dict so JSONSerializer can serialise it
        result = pred_client.predict(input_df.to_dict(orient='list'))

        # Endpoint now returns {"predictions": [...], "probabilities": [...]}
        if isinstance(result, dict):
            label       = int(result["predictions"][0])
            probability = float(result["probabilities"][0])
        else:
            # Fallback for plain list response
            label       = int(result[0])
            probability = None

        if label == 1:
            st.error("🚨 **FRAUD DETECTED** — This transaction is predicted as fraudulent.")
        else:
            st.success("✅ **LEGITIMATE** — This transaction appears normal.")

        if probability is not None:
            st.metric("Fraud Probability", f"{probability:.2%}")

        # ── SHAP local explanation ────────────────────────────
        st.write("### Local SHAP Explanation")
        try:
            pipeline = load(PIPELINE_PATH)

            # Build pre-processing pipeline (no sampler or model)
            pre_steps = [
                (n, s) for n, s in pipeline.steps
                if n not in ("model", "sampler")
            ]
            pre_pipe = SKPipeline(pre_steps)
            X_tr     = pre_pipe.transform(input_df)

            # TreeExplainer is correct for Random Forest
            explainer = shap.TreeExplainer(pipeline.named_steps['model'])
            shap_vals = explainer(X_tr)

            # shap_vals shape is (n_samples, n_features, n_classes) for RF
            # Slice to class 1 (fraud) for the single input row
            fig, _ = plt.subplots()
            shap.waterfall_plot(shap_vals[0, :, 1], max_display=10, show=False)
            st.pyplot(fig)

        except Exception as ex:
            st.warning(f"SHAP explanation unavailable: {ex}")

    except Exception as ex:
        st.error(f"Prediction error: {ex}")

st.caption("Powered by AWS SageMaker + Random Forest Pipeline | IEEE-CIS Fraud Detection")
