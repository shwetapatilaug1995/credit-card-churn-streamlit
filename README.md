\# Credit Card Customer Churn Prediction



\## Problem Statement

The objective of this project is to build and compare multiple machine learning classification models to predict customer attrition in a credit card business. The goal is to identify customers who are likely to leave the service, enabling proactive retention strategies.



\## Dataset Description

The dataset is sourced from Kaggle and contains 10,127 records with customer demographic and transaction-related features. 



Dataset name: Credit Card Customers Dataset



Source: Kaggle



Total records (~10,000)



Target column: Attrition\_Flag



Binary classification (Existing vs Attrited)



Minimum features satisfied 







\## Models Used

Logistic Regression, Decision Tree, KNN, Naive Bayes, Random Forest, XGBoost.





| ML Model                  | Accuracy | AUC    | Precision | Recall | F1 Score | MCC    |

| ------------------------- | -------- | ------ | --------- | ------ | -------- | ------ |

| Logistic Regression       | 1.0000   | 1.0000 | 1.0000    | 1.0000 | 1.0000   | 1.0000 |

| Decision Tree             | 1.0000   | 1.0000 | 1.0000    | 1.0000 | 1.0000   | 1.0000 |

| K-Nearest Neighbors (KNN) | 0.9980   | 0.9999 | 1.0000    | 0.9877 | 0.9938   | 0.9927 |

| Naive Bayes               | 1.0000   | 1.0000 | 1.0000    | 1.0000 | 1.0000   | 1.0000 |

| Random Forest (Ensemble)  | 1.0000   | 1.0000 | 1.0000    | 1.0000 | 1.0000   | 1.0000 |

| XGBoost (Ensemble)        | 0.9990   | 1.0000 | 1.0000    | 0.9938 | 0.9969   | 0.9963 |





\##Model Performance Observations

| ML Model                  | Observation about model performance                                                                                                                                                             | Logistic Regression       | Achieved perfect performance across all metrics, indicating the dataset is highly linearly separable after preprocessing. The model is simple, interpretable, and performed exceptionally well. |

| Decision Tree             | Also achieved perfect scores, suggesting that the tree was able to capture all decision boundaries. However, such perfect performance may indicate potential overfitting.                       |

| K-Nearest Neighbors (KNN) | Delivered very high performance with slightly lower recall, showing sensitivity to neighborhood structure and dependence on feature scaling.                                                    |

| Naive Bayes               | Achieved perfect metrics, indicating strong conditional independence among features for this dataset. Performs well despite its simplicity.                                                     |

| Random Forest (Ensemble)  | Delivered perfect performance, demonstrating the strength of ensemble learning in reducing variance and improving generalization.                                                               |

| XGBoost (Ensemble)        | Achieved near-perfect performance with marginally lower recall. This confirms XGBoost’s robustness and effectiveness in handling complex patterns.                                              |



