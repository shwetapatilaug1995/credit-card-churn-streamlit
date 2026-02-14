# Credit Card Customer Churn Prediction



## Problem Statement



The objective of this project is to build and compare multiple machine learning classification models to predict customer attrition in a credit card business. The goal is to identify customers who are likely to leave the service, enabling proactive retention strategies.



---



## Dataset Description



The dataset is sourced from Kaggle and contains 10,127 records and 23 attributes with customer demographic and transaction-related features.



* Dataset Name: Credit Card Customers Dataset

* Source: Kaggle

* Total Records: ~10,000

* Target Column: Attrition\_Flag

* Problem Type: Binary Classification (Existing vs Attrited Customers)



---



## Models Implemented



The following machine learning models were trained and evaluated:



* Logistic Regression

* Decision Tree

* K-Nearest Neighbors (KNN)

* Naive Bayes

* Random Forest

* XGBoost



---




## Model Performance Comparison

| Model | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|-------|----------|-----|-----------|--------|----------|-----|
| Logistic Regression | 0.8998 | 0.9166 | 0.7675 | 0.5385 | 0.6329 | 0.5891 |
| Decision Tree | 0.9334 | 0.8707 | 0.8006 | 0.7785 | 0.7894 | 0.7499 |
| KNN | 0.8712 | 0.8003 | 0.7623 | 0.2862 | 0.4161 | 0.4151 |
| Naive Bayes | 0.8702 | 0.8378 | 0.6006 | 0.5692 | 0.5845 | 0.5079 |
| Random Forest | 0.9551 | 0.9841 | 0.9366 | 0.7723 | 0.8465 | 0.8258 |
| XGBoost | 0.9664 | 0.9920 | 0.9269 | 0.8585 | 0.8914 | 0.8725 |




---



## Performance Observations

| ML Model Name | Observation about Model Performance |
|---------------|-------------------------------------|
| Logistic Regression | Logistic Regression provided a reasonable baseline performance. However, it showed lower recall compared to tree-based models, indicating difficulty in capturing complex nonlinear relationships present in the data. |
| Decision Tree | The Decision Tree model achieved good recall and balanced performance, but its generalization capability was slightly lower compared to ensemble methods, which may indicate mild overfitting. |
| KNN | The K-Nearest Neighbors (KNN) model showed relatively low recall despite moderate accuracy. This suggests sensitivity to feature scaling and class imbalance, which affected its ability to correctly identify churn cases. |
| Naive Bayes | Naive Bayes demonstrated moderate performance across evaluation metrics. Its assumptions of feature independence may have limited its effectiveness for this dataset. |
| Random Forest (Ensemble) | Random Forest performed very well with high accuracy and strong predictive capability. The ensemble approach helped improve robustness and reduced overfitting compared to single decision trees. |
| XGBoost (Ensemble) | XGBoost achieved the best overall performance with the highest accuracy, AUC, and MCC scores. This indicates strong learning capability and excellent generalization due to gradient boosting optimization. |





---



## Project Structure



```

ML Assignment 2

│── model/

│   ├── model.pkl

│   ├── scaler.pkl

│   └── features.pkl

│

│── train\_models.py

│── app.py

│── BankChurners.csv

│── requirements.txt

│── README.md

```



---



## Conclusion



This project demonstrates the application of multiple machine learning algorithms for customer churn prediction. Ensemble methods, particularly XGBoost and Random Forest, provided the most reliable results and can be effectively used for real-world churn prediction systems.



---



## Author



Shweta Patil

2024DC04177

