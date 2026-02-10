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





| ML Model Name             | Accuracy | AUC    | Precision | Recall | F1     | MCC    |

| ------------------------- | -------- | ------ | --------- | ------ | ------ | ------ |

| Logistic Regression       | 1.0000   | 1.0000 | 1.0000    | 1.0000 | 1.0000 | 1.0000 |

| Decision Tree             | 1.0000   | 1.0000 | 1.0000    | 1.0000 | 1.0000 | 1.0000 |

| k-Nearest Neighbors (kNN) | 0.9980   | 0.9999 | 1.0000    | 0.9877 | 0.9938 | 0.9927 |

| Naive Bayes               | 1.0000   | 1.0000 | 1.0000    | 1.0000 | 1.0000 | 1.0000 |

| Random Forest (Ensemble)  | 1.0000   | 1.0000 | 1.0000    | 1.0000 | 1.0000 | 1.0000 |

| XGBoost (Ensemble)        | 0.9990   | 1.0000 | 1.0000    | 0.9938 | 0.9969 | 0.9963 |





\##Model Performance Observations

| ML Model Name             | Observation about model performance                                                                                                                                                                        |

| ------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |

| Logistic Regression       | Achieved perfect performance on the test dataset, indicating strong linear separability of the features. However, such perfect scores may also indicate possible data simplicity or potential overfitting. |

| Decision Tree             | Delivered perfect classification results but is prone to overfitting, especially when depth is unrestricted, which may affect generalization on unseen data.                                               |

| k-Nearest Neighbors (kNN) | Performed very well with high accuracy and MCC, though slightly lower recall indicates sensitivity to class boundaries and dependence on distance-based feature scaling.                                   |

| Naive Bayes               | Achieved perfect metrics, showing strong conditional independence among features for this dataset; however, this assumption may not always hold in real-world data.                                        |

| Random Forest (Ensemble)  | Demonstrated excellent and stable performance by aggregating multiple decision trees, reducing overfitting and improving robustness.                                                                       |

| XGBoost (Ensemble)        | Provided near-perfect results with strong generalization due to gradient boosting, making it the most reliable and scalable model for this problem.                                                        |

&nbsp;                                           |

