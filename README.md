# ML Assignment 2 – Classification Models Deployment

## a. Problem Statement
The objective of this assignment is to design, implement, and evaluate multiple machine learning classification models on a single dataset. The models are compared using standard performance metrics, and the complete solution is deployed through an interactive Streamlit web application. This assignment demonstrates an end-to-end machine learning workflow including data preprocessing, model training, evaluation, and deployment.

---

## b. Dataset Description
The dataset used for this assignment is the **Heart Disease UCI dataset**, obtained from a public repository. The task is formulated as a **binary classification problem** to predict the presence of heart disease.

**Dataset Summary:**
- Type: Binary Classification  
- Number of instances: 920  
- Number of features: 15  
- Target variable: `num`  
  - `0` → No heart disease  
  - `1` → Presence of heart disease  

The dataset contains a mix of numerical and categorical health indicators. The target variable was converted into a binary format to enable consistent evaluation across all models.

---

## c. Models Used and Evaluation Metrics
Six different classification models were implemented and evaluated using the same preprocessing pipeline and test dataset to ensure a fair comparison.

**Evaluation Metrics Used:**
- Accuracy  
- Area Under ROC Curve (AUC)  
- Precision  
- Recall  
- F1 Score  
- Matthews Correlation Coefficient (MCC)

### Performance Comparison

| ML Model | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|---------|----------|-----|-----------|--------|----------|-----|
| Logistic Regression | 0.8478 | 0.9189 | 0.8426 | 0.8922 | 0.8667 | 0.6913 |
| Decision Tree | 0.8370 | 0.8338 | 0.8462 | 0.8627 | 0.8544 | 0.6694 |
| KNN | 0.8315 | 0.8949 | 0.8447 | 0.8529 | 0.8488 | 0.6586 |
| Naive Bayes | 0.8261 | 0.8775 | 0.8365 | 0.8529 | 0.8447 | 0.6473 |
| Random Forest | 0.8967 | 0.9500 | 0.8807 | 0.9412 | 0.9100 | 0.7916 |
| XGBoost | 0.8804 | 0.9399 | 0.8704 | 0.9216 | 0.8952 | 0.7579 |

---

## d. Observations on Model Performance

| ML Model | Observation |
|--------|-------------|
| Logistic Regression | Logistic Regression showed stable and interpretable performance with good accuracy and AUC values. It performed well when the relationship between features and the target variable was approximately linear. However, it was limited in capturing complex non-linear patterns present in the dataset. |
| Decision Tree | The Decision Tree model effectively captured non-linear relationships but exhibited signs of overfitting. Small changes in the data led to noticeable variations in predictions. This reduced its generalization ability compared to ensemble-based methods. |
| KNN | K-Nearest Neighbors produced competitive results after proper feature scaling was applied. Its performance was highly sensitive to the choice of distance metric and the number of neighbors. Additionally, it can be computationally expensive and sensitive to noisy data points. |
| Naive Bayes | Naive Bayes provided fast and computationally efficient predictions. Its assumption of feature independence simplifies model training but may not fully represent real-world relationships. Despite this limitation, it achieved reasonable performance on the heart disease dataset. |
| Random Forest | Random Forest achieved the strongest and most consistent performance across all evaluation metrics. Its ensemble learning approach reduced overfitting by aggregating multiple decision trees. This made it highly robust and well-suited for complex medical classification tasks. |
| XGBoost | XGBoost delivered high predictive accuracy and strong AUC values, closely matching Random Forest. Its gradient boosting framework effectively captured complex feature interactions. However, careful hyperparameter tuning is required to balance performance and computational cost. |

---

## e. Deployment Overview
All trained models were deployed using **Streamlit Community Cloud**. The web application allows users to:
- Select a classification model dynamically  
- View evaluation metrics in a structured and visual format  
- Analyze confusion matrices and classification reports  
- Download and evaluate results using a fixed test dataset for consistency  

This deployment completes the end-to-end machine learning pipeline and ensures reproducibility between training results and deployed model evaluation.

---

## f. Conclusion
All six classification models demonstrated strong performance on the given dataset. Ensemble-based methods such as **Random Forest** and **XGBoost** achieved superior results due to their ability to model complex, non-linear feature interactions. The deployed Streamlit application successfully integrates model evaluation and visualization, providing a practical demonstration of machine learning model deployment.
