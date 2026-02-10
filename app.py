import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)
import seaborn as sns
import matplotlib.pyplot as plt

# -------------------------------
# Page configuration
# -------------------------------
st.set_page_config(page_title="Credit Card Churn Prediction", layout="wide")

st.title("Credit Card Customer Churn Prediction")
st.write("Upload test data and evaluate different machine learning models.")

# -------------------------------
# Load models and scaler
# -------------------------------
@st.cache_resource
def load_models():
    models = {
        "Logistic Regression": joblib.load("model/logistic.pkl"),
        "Decision Tree": joblib.load("model/decision_tree.pkl"),
        "KNN": joblib.load("model/knn.pkl"),
        "Naive Bayes": joblib.load("model/naive_bayes.pkl"),
        "Random Forest": joblib.load("model/random_forest.pkl"),
        "XGBoost": joblib.load("model/xgboost.pkl"),
    }
    scaler = joblib.load("model/scaler.pkl")
    return models, scaler

models, scaler = load_models()

# -------------------------------
# Dataset upload
# -------------------------------
uploaded_file = st.file_uploader("Upload test dataset (CSV)", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

    st.subheader("Uploaded Dataset Preview")
    st.dataframe(data.head())

    # -------------------------------
    # Target column handling
    # -------------------------------
    if "Attrition_Flag" not in data.columns:
        st.error("Target column 'Attrition_Flag' not found in dataset.")
        st.stop()

    y = data["Attrition_Flag"].map({
        "Existing Customer": 0,
        "Attrited Customer": 1
    })

    X = data.drop(columns=["Attrition_Flag", "CLIENTNUM"], errors="ignore")

    # One-hot encoding
    X = pd.get_dummies(X)

    # Align columns (important!)
    scaler_features = scaler.feature_names_in_
    X = X.reindex(columns=scaler_features, fill_value=0)

    # Scaling
    X_scaled = scaler.transform(X)

    # -------------------------------
    # Model selection
    # -------------------------------
    st.subheader("Model Selection")
    model_name = st.selectbox("Choose a model", list(models.keys()))
    model = models[model_name]

    # -------------------------------
    # Prediction
    # -------------------------------
    if model_name in ["Logistic Regression", "KNN", "Naive Bayes"]:
        y_pred = model.predict(X_scaled)
        y_prob = model.predict_proba(X_scaled)[:, 1]
    else:
        y_pred = model.predict(X)
        y_prob = model.predict_proba(X)[:, 1]

    # -------------------------------
    # Metrics display
    # -------------------------------
    st.subheader("Evaluation Metrics")

    col1, col2, col3 = st.columns(3)

    col1.metric("Accuracy", round(accuracy_score(y, y_pred), 4))
    col2.metric("Precision", round(precision_score(y, y_pred), 4))
    col3.metric("Recall", round(recall_score(y, y_pred), 4))

    col4, col5, col6 = st.columns(3)

    col4.metric("F1 Score", round(f1_score(y, y_pred), 4))
    col5.metric("AUC", round(roc_auc_score(y, y_prob), 4))

    # -------------------------------
    # Confusion Matrix
    # -------------------------------
    st.subheader("Confusion Matrix")

    cm = confusion_matrix(y, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

    st.pyplot(fig)
