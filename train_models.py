

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score,matthews_corrcoef
)

df=pd.read_csv("BankChurners.csv")
target = "Attrition_Flag"
#Target_column

df[target] = df[target].map({
    "Existing Customer": 0,
    "Attrited Customer": 1
})
#Drop unnecessary columns
df = df.drop([
    "CLIENTNUM"
], axis=1)
#categorical columns handling
cat_cols = df.select_dtypes(include="object").columns
df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
#Split features & label
X=df.drop(target,axis=1)
y=df[target]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    return {
        "Accuracy": accuracy_score(y_test, y_pred),
        "AUC": roc_auc_score(y_test, y_prob),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1": f1_score(y_test, y_pred),
        "MCC": matthews_corrcoef(y_test, y_pred)
    }
#Logistic Regression
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(max_iter=1000)
lr.fit(X_train_scaled, y_train)

lr_metrics = evaluate_model(lr, X_test_scaled, y_test)
#Decision Tree Classifier
from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)

dt_metrics = evaluate_model(dt, X_test, y_test)
#K-Nearest Neighbors (KNN)
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)

knn_metrics = evaluate_model(knn, X_test_scaled, y_test)
#naive_bayes
from sklearn.naive_bayes import GaussianNB

nb = GaussianNB()
nb.fit(X_train_scaled, y_train)

nb_metrics = evaluate_model(nb, X_test_scaled, y_test)
#RandomForest
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

rf.fit(X_train, y_train)

rf_metrics = evaluate_model(rf, X_test, y_test)
#xgboost - Ensemble
from xgboost import XGBClassifier

xgb = XGBClassifier(
    use_label_encoder=False,
    eval_metric="logloss",
    random_state=42
)

xgb.fit(X_train, y_train)

xgb_metrics = evaluate_model(xgb, X_test, y_test)
import joblib

joblib.dump(lr, "model/logistic.pkl")
joblib.dump(dt, "model/decision_tree.pkl")
joblib.dump(knn, "model/knn.pkl")
joblib.dump(nb, "model/naive_bayes.pkl")
joblib.dump(rf, "model/random_forest.pkl")
joblib.dump(xgb, "model/xgboost.pkl")
joblib.dump(scaler, "model/scaler.pkl")


print("Logistic:", lr_metrics)
print("Decision Tree:", dt_metrics)
print("KNN:", knn_metrics)
print("Naive Bayes:", nb_metrics)
print("Random Forest:", rf_metrics)
print("XGBoost:", xgb_metrics)

