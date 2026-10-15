# Breast Cancer Prediction - ML Model Deployment (Flask)

End-to-end machine learning project: data preprocessing, model training, evaluation,
serialization with Joblib, and deployment as a Flask web app + REST API.

## Project structure
```
app.py                 Flask app (web UI + /predict API)
train_model.py         Preprocessing, training, evaluation, saves model
model/model.joblib     Serialized scikit-learn pipeline
model/metrics.json     Evaluation results
templates/index.html   Web form
confusion_matrix.png   Evaluation chart
requirements.txt       Dependencies
Procfile               For Render/Heroku deployment (gunicorn)
```

## Run locally
```bash
pip install -r requirements.txt
python train_model.py     # trains and saves model/model.joblib
python app.py             # open http://127.0.0.1:5000
```

## API
`POST /predict` with JSON (any missing feature defaults to the dataset mean)
```bash
curl -X POST http://127.0.0.1:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"mean radius": 20.5, "mean texture": 25, "mean perimeter": 135, "mean area": 1300}'
```
Response: `{"prediction": "Malignant", "confidence": 98.52}`

`GET /health` -> `{"status": "ok"}`

## Results
| Model | CV Acc. | Test Acc. | F1 |
|---|---|---|---|
| Logistic Regression (selected) | 98.02% | 98.25% | 98.61% |
| Random Forest | 95.82% | 95.61% | 96.55% |

## Deploy on Render (free)
New Web Service -> connect this repo -> Build: `pip install -r requirements.txt && python train_model.py` -> Start: `gunicorn app:app`.

> Educational project only - not a medical diagnosis tool.
