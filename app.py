# ================== CREDIT CARD FRAUD DETECTION - STREAMLIT APP ==================

import streamlit as st
import numpy as np
import pandas as pd
import joblib
import json
import os
import matplotlib.pyplot as plt


# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------- LOAD ARTIFACTS ----------------
@st.cache_resource
def load_model():
    return joblib.load("model.pkl")

@st.cache_resource
def load_scaler():
    loaded = joblib.load("scaler.pkl")
    # Support new format (dict of separate scalers) and legacy format (single scaler)
    if isinstance(loaded, dict):
        return loaded
    return {"amount": loaded, "time": loaded}

@st.cache_data
def load_feature_names():
    with open("feature_names.json", "r") as f:
        return json.load(f)

@st.cache_data
def load_results():
    with open("outputs/results.json", "r") as f:
        return json.load(f)

@st.cache_data
def load_data():
    return pd.read_csv("creditcard.csv")

# Load everything
try:
    model = load_model()
    scalers = load_scaler()
    feature_names = load_feature_names()
    results = load_results()
    data = load_data()
    artifacts_loaded = True
except FileNotFoundError as e:
    st.error(f"Missing file: {e}. Please run `python train_model.py` first to train the model.")
    st.stop()

# ---------------- SIDEBAR ----------------
st.sidebar.title("💳 Fraud Detection")
page = st.sidebar.radio("Navigate", ["Dashboard", "Single Prediction", "Batch Prediction", "Model Comparison"])

st.sidebar.markdown("---")
st.sidebar.subheader("About")
st.sidebar.info(
    "This app detects credit card fraud using machine learning. "
    "Trained on the Kaggle Credit Card Fraud Detection dataset with "
    "SMOTE oversampling and multiple model comparison."
)

# ---------------- DASHBOARD ----------------
if page == "Dashboard":
    st.title("💳 Credit Card Fraud Detection Dashboard")

    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    fraud_count = int(data["Class"].sum())
    normal_count = len(data) - fraud_count

    with col1:
        st.metric("Total Transactions", f"{len(data):,}")
    with col2:
        st.metric("Normal Transactions", f"{normal_count:,}")
    with col3:
        st.metric("Fraud Transactions", f"{fraud_count:,}")
    with col4:
        st.metric("Fraud Ratio", f"{fraud_count/len(data):.4%}")

    st.markdown("---")

    # Best model info
    best_name = max(results, key=lambda k: results[k]["f1"])
    best = results[best_name]

    col1, col2 = st.columns(2)
    with col1:
        st.subheader(f"Best Model: {best_name}")
        # Primary metrics in a spacious 3-column layout so nothing gets truncated
        r1_col1, r1_col2, r1_col3 = st.columns(3)
        r1_col1.metric("F1-Score", f"~{best['f1']:.1%}")
        r1_col2.metric("Precision", f"~{best['precision']:.1%}")
        r1_col3.metric("Recall", f"~{best['recall']:.1%}")

        st.write("")  # small spacer
        r2_col1, r2_col2, _ = st.columns(3)
        r2_col1.metric("ROC-AUC", f"~{best['roc_auc']:.1%}")
        r2_col2.metric("PR-AUC", f"~{best['pr_auc']:.1%}")

    with col2:
        # Class distribution
        st.subheader("Class Distribution")
        fig, ax = plt.subplots()
        categories = ["Normal", "Fraud"]
        counts = [normal_count, fraud_count]
        bars = ax.bar(categories, counts, color=["#2ecc71", "#e74c3c"])
        ax.set_ylabel("Count")
        ax.set_yscale("log")
        for bar, count in zip(bars, counts):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{count:,}", ha="center", va="bottom")
        st.pyplot(fig)
        plt.close()

    st.markdown("---")

    # Output visualizations
    st.subheader("Visualizations")
    viz_col1, viz_col2 = st.columns(2)

    viz_files = {
        "ROC Curves": "roc_curves.png",
        "Precision-Recall Curves": "pr_curves.png",
        "Confusion Matrices": "confusion_matrices.png",
        "Model Comparison": "model_comparison.png",
        "Feature Importance": "feature_importance.png",
    }

    for i, (title, filename) in enumerate(viz_files.items()):
        path = os.path.join("outputs", filename)
        if os.path.exists(path):
            col = viz_col1 if i % 2 == 0 else viz_col2
            col.image(path, caption=title, width="stretch")

# ---------------- SINGLE PREDICTION ----------------
elif page == "Single Prediction":
    st.title("🔍 Single Transaction Prediction")
    st.markdown("Enter transaction features below to check if it's fraudulent.")

    with st.form("prediction_form"):
        col1, col2 = st.columns(2)

        # Amount and Time (will be scaled)
        with col1:
            amount = st.number_input("Transaction Amount ($)", min_value=0.0, value=100.0, step=10.0)
        with col2:
            time = st.number_input("Time (seconds since first transaction)", min_value=0.0, value=0.0, step=100.0)

        st.markdown("### PCA Features (V1-V28)")
        st.markdown("Enter values for the PCA-transformed features. Default is 0.0 (typical for normal transactions).")

        v_features = {}
        v_cols = st.columns(4)
        for i in range(1, 29):
            with v_cols[(i - 1) % 4]:
                v_features[f"V{i}"] = st.number_input(f"V{i}", value=0.0, step=0.1, format="%.4f")

        submitted = st.form_submit_button("Predict", type="primary")

        if submitted:
            # Build feature vector in the correct order
            scaled_amount = scalers["amount"].transform([[amount]])[0][0]
            scaled_time = scalers["time"].transform([[time]])[0][0]

            feature_dict = {"scaled_Amount": scaled_amount, "scaled_Time": scaled_time}
            for k, v in v_features.items():
                feature_dict[k] = v

            # Create DataFrame with correct column order
            input_df = pd.DataFrame([[feature_dict.get(f, 0.0) for f in feature_names]], columns=feature_names)

            # Predict
            prediction = model.predict(input_df)[0]

            # Handle Isolation Forest output: 1=normal(inlier), -1=fraud(outlier)
            if hasattr(model, "decision_function") and not hasattr(model, "predict_proba"):
                prediction = 1 if prediction == -1 else 0

            # Get probability / anomaly score
            if hasattr(model, "predict_proba"):
                prob = model.predict_proba(input_df)[0][1]
            elif hasattr(model, "decision_function"):
                score = model.decision_function(input_df)[0]
                prob = 1 / (1 + np.exp(score))  # Lower score = more anomalous = higher fraud prob
            else:
                prob = None

            # Display result
            if prediction == 1:
                st.error(f"🚨 FRAUD DETECTED! (Confidence: {prob:.2%})" if prob is not None else "🚨 FRAUD DETECTED!")
            else:
                st.success(f"✅ NORMAL TRANSACTION (Confidence: {1 - prob:.2%})" if prob is not None else "✅ NORMAL TRANSACTION")

            # Show feature values
            with st.expander("View Input Features"):
                st.dataframe(input_df.T.rename(columns={0: "Value"}))

# ---------------- BATCH PREDICTION ----------------
elif page == "Batch Prediction":
    st.title("📊 Batch Prediction")
    st.markdown("Upload a CSV file with transaction data to get predictions for multiple transactions at once.")

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file is not None:
        try:
            batch_data = pd.read_csv(uploaded_file)
            st.write(f"**Loaded {len(batch_data):,} transactions**")
            st.dataframe(batch_data.head())

            if st.button("Run Batch Prediction", type="primary"):
                # Prepare features (use copy to avoid mutating original data)
                batch_features = batch_data.copy()
                if "Amount" in batch_features.columns and "Time" in batch_features.columns:
                    batch_features["scaled_Amount"] = scalers["amount"].transform(batch_features["Amount"].values.reshape(-1, 1))
                    batch_features["scaled_Time"] = scalers["time"].transform(batch_features["Time"].values.reshape(-1, 1))
                    batch_features.drop(["Amount", "Time"], axis=1, inplace=True, errors="ignore")

                # Reorder columns and validate
                available_cols = [c for c in feature_names if c in batch_features.columns]
                if len(available_cols) != len(feature_names):
                    missing = set(feature_names) - set(available_cols)
                    st.error(f"Missing required features in uploaded CSV: {missing}")
                    st.stop()
                input_df = batch_features[available_cols]

                # Predict
                predictions = model.predict(input_df)
                # Handle Isolation Forest: 1=normal(inlier), -1=fraud(outlier)
                if hasattr(model, "decision_function") and not hasattr(model, "predict_proba"):
                    predictions = np.where(predictions == -1, 1, 0)

                # Probabilities
                if hasattr(model, "predict_proba"):
                    probs = model.predict_proba(input_df)[:, 1]
                elif hasattr(model, "decision_function"):
                    scores = model.decision_function(input_df)
                    probs = 1 / (1 + np.exp(scores))
                else:
                    probs = [None] * len(predictions)

                # Results
                result_df = batch_data.copy()
                result_df["Prediction"] = predictions
                result_df["Prediction_Label"] = np.where(predictions == 1, "FRAUD", "NORMAL")
                if probs[0] is not None:
                    result_df["Fraud_Probability"] = probs

                # Summary
                fraud_detected = predictions.sum()
                col1, col2, col3 = st.columns(3)
                col1.metric("Total Transactions", f"{len(predictions):,}")
                col2.metric("Fraud Detected", f"{int(fraud_detected):,}")
                col3.metric("Normal Transactions", f"{int(len(predictions) - fraud_detected):,}")

                # Show results
                st.dataframe(result_df, width="stretch")

                # Download button
                csv = result_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download Predictions as CSV",
                    data=csv,
                    file_name="fraud_predictions.csv",
                    mime="text/csv"
                )

        except Exception as e:
            st.error(f"Error processing file: {e}")

# ---------------- MODEL COMPARISON ----------------
elif page == "Model Comparison":
    st.title("📈 Model Comparison")

    # Results table
    st.subheader("Performance Metrics")
    metrics_df = pd.DataFrame(results).T
    metrics_display = metrics_df[["f1", "precision", "recall", "roc_auc", "pr_auc"]]
    metrics_display.columns = ["F1", "Precision", "Recall", "ROC-AUC", "PR-AUC"]
    metrics_display = metrics_display.round(4)

    # Highlight best
    st.dataframe(
        metrics_display.style.highlight_max(axis=0, color="lightgreen"),
        width="stretch"
    )

    # Visualizations
    st.subheader("Visualizations")
    tab1, tab2, tab3, tab4 = st.tabs(["ROC Curves", "PR Curves", "Confusion Matrices", "Model Comparison"])

    with tab1:
        if os.path.exists("outputs/roc_curves.png"):
            st.image("outputs/roc_curves.png", width="stretch")
    with tab2:
        if os.path.exists("outputs/pr_curves.png"):
            st.image("outputs/pr_curves.png", width="stretch")
    with tab3:
        if os.path.exists("outputs/confusion_matrices.png"):
            st.image("outputs/confusion_matrices.png", width="stretch")
    with tab4:
        if os.path.exists("outputs/model_comparison.png"):
            st.image("outputs/model_comparison.png", width="stretch")

    # Detailed classification report for best model
    st.subheader("Best Model Details")
    best_name = max(results, key=lambda k: results[k]["f1"])
    st.info(f"Best performing model: **{best_name}** (F1: {results[best_name]['f1']:.4f})")

    # Feature importance
    if os.path.exists("outputs/feature_importance.png"):
        st.subheader("Feature Importance")
        st.image("outputs/feature_importance.png", width="stretch")
