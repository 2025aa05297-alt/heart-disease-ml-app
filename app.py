import streamlit as st
import pandas as pd
import joblib
import os
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)
from models.common_preprocessing import load_scaler

# --------------------------------------------------
# Page Configuration
# --------------------------------------------------
st.set_page_config(
    page_title="Heart Disease Prediction - ML Models",
    layout="centered"
)

st.title("Heart Disease Prediction System")
st.write(
    """
    This Streamlit application allows users to evaluate different machine learning
    classification models trained on the Heart Disease dataset.
    Upload a **test CSV file**, select a model, and view performance metrics.
    """
)

# --------------------------------------------------
# Load Scaler
# --------------------------------------------------
scaler = load_scaler("saved_models/scaler.pkl")

# --------------------------------------------------
# Model Dictionary
# --------------------------------------------------
MODEL_PATHS = {
    "Logistic Regression": "saved_models/logistic.pkl",
    "Decision Tree": "saved_models/decision_tree.pkl",
    "KNN": "saved_models/knn.pkl",
    "Naive Bayes": "saved_models/naive_bayes.pkl",
    "Random Forest": "saved_models/random_forest.pkl",
    "XGBoost": "saved_models/xgboost.pkl"
}

# --------------------------------------------------
# File Upload
# --------------------------------------------------
st.header("Upload Test Dataset")
uploaded_file = st.file_uploader(
    "Upload test CSV file (must include target column)",
    type=["csv"]
)

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    if "target" not in df.columns:
        st.error("Uploaded CSV must contain a 'target' column.")
        st.stop()

    X = df.drop(columns=["target"])
    y_true = df["target"]

    # Scale features
    X_scaled = scaler.transform(X)

    # --------------------------------------------------
    # Model Selection
    # --------------------------------------------------
    st.header("Select Model")
    selected_model_name = st.selectbox(
        "Choose a classification model",
        list(MODEL_PATHS.keys())
    )

    model_path = MODEL_PATHS[selected_model_name]

    if not os.path.exists(model_path):
        st.error(f"Model file not found: {model_path}")
        st.stop()

    model = joblib.load(model_path)

    # --------------------------------------------------
    # Predictions
    # --------------------------------------------------
    y_pred = model.predict(X_scaled)

    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_scaled)[:, 1]
        auc = roc_auc_score(y_true, y_prob)
    else:
        auc = "N/A"

    # --------------------------------------------------
    # Metrics Calculation
    # --------------------------------------------------
    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)

    # --------------------------------------------------
    # Display Metrics
    # --------------------------------------------------
    st.header("Evaluation Metrics")

    col1, col2, col3 = st.columns(3)
    col1.metric("Accuracy", f"{acc:.3f}")
    col2.metric("Precision", f"{precision:.3f}")
    col3.metric("Recall", f"{recall:.3f}")

    col4, col5, col6 = st.columns(3)
    col4.metric("F1 Score", f"{f1:.3f}")
    col5.metric("MCC", f"{mcc:.3f}")
    col6.metric("AUC", f"{auc:.3f}" if auc != "N/A" else "N/A")

    # --------------------------------------------------
    # Confusion Matrix
    # --------------------------------------------------
    st.header("Confusion Matrix")
    cm = confusion_matrix(y_true, y_pred)
    st.write(pd.DataFrame(
        cm,
        columns=["Predicted 0", "Predicted 1"],
        index=["Actual 0", "Actual 1"]
    ))

    # --------------------------------------------------
    # Classification Report
    # --------------------------------------------------
    st.header("Classification Report")
    report = classification_report(y_true, y_pred, output_dict=True)
    st.dataframe(pd.DataFrame(report).transpose())

else:
    st.info("Please upload a test CSV file to continue.")
