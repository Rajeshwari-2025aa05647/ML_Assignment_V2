import streamlit as st
import pandas as pd
import joblib
import numpy as np

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

st.set_page_config(page_title="ML Assignment 2", layout="centered")
st.title("Machine Learning Classification Models")

# Add download link for test data
st.markdown("""
### Instructions
1. Download the test data CSV file from the GitHub repository or use the sample test data
2. Upload the CSV file below
3. Select a model to see evaluation metrics
""")

# File uploader
uploaded_file = st.file_uploader("Upload CSV test file", type=["csv"])

# Model selection
model_name = st.selectbox(
    "Select Model",
    [
        "Logistic Regression",
        "Decision Tree",
        "KNN",
        "Naive Bayes",
        "Random Forest",
        "XGBoost"
    ]
)

if uploaded_file:
    # Load the test data
    df = pd.read_csv(uploaded_file)

    # Check if target column exists
    if "num" not in df.columns:
        st.error("Target column 'num' not found in uploaded file.")
        st.stop()

    # Separate features and target
    y = df["num"]
    X = df.drop("num", axis=1)

    # CRITICAL FIX: The test data is already preprocessed (encoded, scaled, imputed)
    # from the notebook. We should NOT re-preprocess it.
    # Just convert to numpy array if needed
    if isinstance(X, pd.DataFrame):
        X_test = X.values
    else:
        X_test = X

    # Load the selected model
    try:
        model = joblib.load(f"model/{model_name.replace(' ', '_')}.pkl")
    except FileNotFoundError:
        st.error(f"Model file 'model/{model_name.replace(' ', '_')}.pkl' not found. Please ensure model files are in the 'model/' directory.")
        st.stop()

    # Make predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # Calculate metrics
    st.subheader("Evaluation Metrics")
    
    metrics = {
        "Accuracy": accuracy_score(y, y_pred),
        "AUC": roc_auc_score(y, y_prob),
        "Precision": precision_score(y, y_pred),
        "Recall": recall_score(y, y_pred),
        "F1": f1_score(y, y_pred),
        "MCC": matthews_corrcoef(y, y_pred)
    }
    
    # Display metrics in a nice format
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Accuracy", f"{metrics['Accuracy']:.4f}")
        st.metric("Precision", f"{metrics['Precision']:.4f}")
    
    with col2:
        st.metric("AUC", f"{metrics['AUC']:.4f}")
        st.metric("Recall", f"{metrics['Recall']:.4f}")
    
    with col3:
        st.metric("F1 Score", f"{metrics['F1']:.4f}")
        st.metric("MCC", f"{metrics['MCC']:.4f}")

    # Confusion Matrix
    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y, y_pred)
    
    # Create a DataFrame for better display
    cm_df = pd.DataFrame(
        cm,
        index=['Actual Negative', 'Actual Positive'],
        columns=['Predicted Negative', 'Predicted Positive']
    )
    st.dataframe(cm_df)

    # Classification Report
    st.subheader("Classification Report")
    st.text(classification_report(y, y_pred, target_names=['No Disease', 'Disease']))
    
    # Additional information
    st.info(f"Total predictions: {len(y)} | Model: {model_name}")

else:
    st.info("Please upload a CSV file to begin evaluation.")
    st.markdown("""
    **Expected CSV format:**
    - Must contain preprocessed features (already encoded, scaled, and imputed)
    - Must contain 'num' column as target variable (0 or 1)
    - Download test_data.csv from the GitHub repository
    """)
