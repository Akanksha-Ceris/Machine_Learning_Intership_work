# Week 2: Supervised Machine Learning Models

## Overview
This project demonstrates supervised machine learning with Scikit-learn:
- Classification: Logistic Regression, Decision Tree, Random Forest, and K-Nearest Neighbors (KNN), using the Iris dataset.
- Regression: Linear Regression, using the built-in Diabetes dataset.

## Requirements
- Python 3.9 or newer recommended
- scikit-learn
- pandas
- numpy

Install dependencies:
```bash
pip install scikit-learn pandas numpy
```

## Run
```bash
python supervised_ml_models.py
```

The script prints classification accuracy, macro precision, macro recall, macro F1-score, confusion matrices, and regression MAE, RMSE, and R². Results are calculated when you run the script; do not report unrun or assumed values.

## Method
A reproducible 80/20 train-test split is used. The random state is fixed at 42. StandardScaler is applied through a pipeline to Logistic Regression and KNN to avoid fitting the scaler on test data. The Iris dataset is stratified during the split.

## Notes
The datasets are educational examples from scikit-learn. For a real application, select a dataset relevant to the problem and validate the approach appropriately.
