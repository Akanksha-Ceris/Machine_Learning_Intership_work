"""Flask API + web UI that serves the trained model."""
import json
import joblib
import pandas as pd
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)
bundle = joblib.load("model/model.joblib")
model, FEATURES, MEANS = bundle["model"], [str(f) for f in bundle["features"]], bundle["means"]
METRICS = json.load(open("model/metrics.json"))
LABELS = {0: "Malignant", 1: "Benign"}


def predict_from(values: dict):
    # Missing features fall back to dataset mean, so a partial form still works
    row = [float(values.get(f, MEANS[f]) or MEANS[f]) for f in FEATURES]
    x = pd.DataFrame([row], columns=FEATURES)
    cls = int(model.predict(x)[0])
    prob = float(model.predict_proba(x)[0][cls])
    return {"prediction": LABELS[cls], "confidence": round(prob * 100, 2)}


@app.route("/")
def home():
    shown = FEATURES[:10]  # the 10 "mean" features keep the form short
    return render_template("index.html", features=shown, means=MEANS,
                           metrics=METRICS[METRICS["best_model"]], name=METRICS["best_model"])


@app.route("/health")
def health():
    return jsonify(status="ok")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True) or request.form.to_dict()
    try:
        return jsonify(predict_from(data))
    except (ValueError, TypeError) as e:
        return jsonify(error=f"Invalid input: {e}"), 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
