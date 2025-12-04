import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import lightgbm as lgb
from tensorflow.keras.models import load_model

# -----------------------------------------------------------
# Load Artifacts (V6)
# -----------------------------------------------------------

MODEL_DIR = "/mnt/data"

preprocessor = joblib.load(f"{MODEL_DIR}/preprocessor.joblib")
train_columns = joblib.load(f"{MODEL_DIR}/train_columns.joblib")

# LightGBM model
lgb_model = joblib.load(f"{MODEL_DIR}/lgb_model.joblib")

# NeuralNet model
nn_model = load_model(f"{MODEL_DIR}/nn_best.h5")

# Ensemble metadata (best combination found)
with open(f"{MODEL_DIR}/ensemble_meta.json") as f:
    ensemble_meta = json.load(f)

LGB_WEIGHT = ensemble_meta["lgb_weight"]
THRESHOLD = ensemble_meta["threshold"]


# -----------------------------------------------------------
# Prediction helper
# -----------------------------------------------------------
def predict_single_row(input_df):
    """Takes a dataframe with ONE row and returns ensemble probability."""

    # Ensure all missing columns exist
    for col in train_columns:
        if col not in input_df.columns:
            input_df[col] = np.nan

    input_df = input_df[train_columns]

    # Preprocess
    X_proc = preprocessor.transform(input_df)

    # Model predictions
    lgb_prob = lgb_model.predict_proba(X_proc)[:, 1]
    nn_prob  = nn_model.predict(X_proc).ravel()

    # Weighted ensemble
    final_prob = LGB_WEIGHT * lgb_prob + (1 - LGB_WEIGHT) * nn_prob

    final_label = int(final_prob >= THRESHOLD)

    return {
        "lgb_prob": float(lgb_prob[0]),
        "nn_prob": float(nn_prob[0]),
        "ensemble_prob": float(final_prob[0]),
        "ensemble_label": final_label
    }


# -----------------------------------------------------------
# STREAMLIT UI
# -----------------------------------------------------------

st.title("📈 Customer Churn Prediction — V6 Model")
st.write("This app uses a **LightGBM + Neural Network ensemble (V6)** to predict churn probability.")

st.subheader("🔧 Enter Customer Features")

# Manual entry form
with st.form("manual_input"):
    col1, col2 = st.columns(2)

    with col1:
        Age = st.number_input("Age", 18, 100, 30)
        Tenure = st.number_input("Tenure (months)", 0, 240, 12)
        Usage = st.number_input("Usage Frequency", 0, 100, 10)
        TotalSpend = st.number_input("Total Spend", 0.0, 50000.0, 200.0)

    with col2:
        Gender = st.selectbox("Gender", ["Male", "Female"])
        Subscription = st.selectbox("Subscription Type", ["Basic", "Standard", "Premium"])
        Contract = st.selectbox("Contract Length", ["1 month", "6 months", "12 months", "24 months"])

    submitted = st.form_submit_button("Predict")

if submitted:
    row = {
        "Age": Age,
        "Tenure": Tenure,
        "Usage Frequency": Usage,
        "Total Spend": TotalSpend,
        "Gender": Gender,
        "Subscription Type": Subscription,
        "Contract Length": Contract,

        # Engineered features
        "spend_per_month": TotalSpend / max(Tenure, 1),
        "usage_per_tenure": Usage / max(Tenure, 1),
        "age_times_spend": Age * TotalSpend,
    }

    result = predict_single_row(pd.DataFrame([row]))

    st.subheader("📊 Prediction Results (Ensemble)")
    st.write(f"**Final Churn Probability:** `{result['ensemble_prob']:.4f}`")
    st.write(f"**Predicted Label:** `{result['ensemble_label']}`")

    with st.expander("🔍 Model Internal Probabilities"):
        st.write(f"LightGBM: `{result['lgb_prob']:.4f}`")
        st.write(f"NeuralNet: `{result['nn_prob']:.4f}`")

# CSV Upload
st.subheader("📤 Predict from CSV")

uploaded_file = st.file_uploader("Upload CSV with customer records", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    preds = []

    for _, row in df.iterrows():
        r = row.to_dict()

        # fill engineered features
        r["spend_per_month"] = r["Total Spend"] / max(r["Tenure"], 1)
        r["usage_per_tenure"] = r["Usage Frequency"] / max(r["Tenure"], 1)
        r["age_times_spend"] = r["Age"] * r["Total Spend"]

        preds.append(predict_single_row(pd.DataFrame([r])))

    df["lgb_prob"] = [p["lgb_prob"] for p in preds]
    df["nn_prob"] = [p["nn_prob"] for p in preds]
    df["ensemble_prob"] = [p["ensemble_prob"] for p in preds]
    df["prediction"] = [p["ensemble_label"] for p in preds]

    st.write("### 📄 Predictions")
    st.dataframe(df)

    st.download_button("Download Results", df.to_csv(index=False), "predictions.csv")

