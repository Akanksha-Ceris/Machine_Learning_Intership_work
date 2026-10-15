"""Train, evaluate and serialize a breast-cancer classifier."""
import json
import joblib
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# 1. Load data
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target, name="target")  # 0 = malignant, 1 = benign
print("Shape:", X.shape, "| Missing values:", int(X.isnull().sum().sum()))

# 2. Preprocessing: stratified split (scaling happens inside the pipeline -> no data leakage)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# 3. Train & compare models
models = {
    "Logistic Regression": Pipeline([("scaler", StandardScaler()),
                                     ("clf", LogisticRegression(max_iter=1000))]),
    "Random Forest": Pipeline([("scaler", StandardScaler()),
                               ("clf", RandomForestClassifier(n_estimators=200, random_state=42))]),
}
results, best_name, best_model, best_f1 = {}, None, None, -1
for name, pipe in models.items():
    cv = cross_val_score(pipe, X_train, y_train, cv=5, scoring="accuracy").mean()
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    r = {"cv_accuracy": round(float(cv), 4),
         "test_accuracy": round(accuracy_score(y_test, pred), 4),
         "precision": round(precision_score(y_test, pred), 4),
         "recall": round(recall_score(y_test, pred), 4),
         "f1": round(f1_score(y_test, pred), 4)}
    results[name] = r
    print(name, r)
    if r["f1"] > best_f1:
        best_name, best_model, best_f1 = name, pipe, r["f1"]

# 4. Confusion matrix + chart
cm = confusion_matrix(y_test, best_model.predict(X_test))
results["best_model"] = best_name
results["confusion_matrix"] = cm.tolist()
fig, ax = plt.subplots(figsize=(4, 3.5))
ax.imshow(cm, cmap="Blues")
for i in range(2):
    for j in range(2):
        ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=14)
ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
ax.set_xticklabels(["Malignant", "Benign"]); ax.set_yticklabels(["Malignant", "Benign"])
ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title(f"Confusion Matrix - {best_name}")
plt.tight_layout(); plt.savefig("confusion_matrix.png", dpi=150)

# 5. Serialize with joblib
joblib.dump({"model": best_model, "features": [str(f) for f in data.feature_names],
             "means": X.mean().round(4).to_dict()}, "model/model.joblib")
json.dump(results, open("model/metrics.json", "w"), indent=2)
print("Best model:", best_name, "-> saved to model/model.joblib")
