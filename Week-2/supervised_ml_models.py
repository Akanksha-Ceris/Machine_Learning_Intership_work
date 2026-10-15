"""
Week 2 Internship Task: Supervised Machine Learning Models
Run with: python supervised_ml_models.py

Dependencies:
    pip install scikit-learn pandas numpy

This project demonstrates:
- Classification: Logistic Regression, Decision Tree, Random Forest, K-Nearest Neighbors
- Regression: Linear Regression
- Evaluation using accuracy, precision, recall, F1-score, confusion matrix, MAE, RMSE, and R²
Datasets are built into scikit-learn, so no separate download is needed.
"""

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris, load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, mean_absolute_error, mean_squared_error, r2_score
)

RANDOM_STATE = 42

def run_classification():
    data = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.2,
        random_state=RANDOM_STATE, stratify=data.target
    )

    models = {
        "Logistic Regression": make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
        ),
        "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE),
        "K-Nearest Neighbors": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5)),
    }

    rows = []
    print("\\n=== CLASSIFICATION: IRIS DATASET ===")
    for name, model in models.items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        metrics = {
            "Model": name,
            "Accuracy": accuracy_score(y_test, predictions),
            "Precision (macro)": precision_score(y_test, predictions, average="macro", zero_division=0),
            "Recall (macro)": recall_score(y_test, predictions, average="macro", zero_division=0),
            "F1-score (macro)": f1_score(y_test, predictions, average="macro", zero_division=0),
        }
        rows.append(metrics)
        print("\\n" + name)
        for key, value in metrics.items():
            if key != "Model":
                print(f"  {key}: {value:.4f}")
        print("  Confusion matrix:")
        print(confusion_matrix(y_test, predictions))
    result = pd.DataFrame(rows).sort_values("Accuracy", ascending=False)
    print("\\nClassification comparison:")
    print(result.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    return result

def run_regression():
    data = load_diabetes()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.2, random_state=RANDOM_STATE
    )
    model = LinearRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    print("\\n=== REGRESSION: DIABETES DATASET ===")
    print("Model: Linear Regression")
    print(f"  MAE: {mae:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  R² score: {r2:.4f}")
    return {"Model": "Linear Regression", "MAE": mae, "RMSE": rmse, "R2": r2}

if __name__ == "__main__":
    classification_results = run_classification()
    regression_results = run_regression()
    print("\\nDone. Use these measured results in your report.")
