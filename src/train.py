"""
train.py

This script trains multiple ML models for heart disease prediction
and logs experiments using MLflow.

Models trained:
1. Logistic Regression
2. Random Forest

Metrics logged:
- Accuracy
- ROC-AUC
"""

import os
import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

from preprocess import load_and_preprocess_data


def train_and_log_model(model, model_name, X_train, X_test, y_train, y_test):
    """
    Trains a model and logs parameters, metrics, and artifacts to MLflow.
    """

    with mlflow.start_run(run_name=model_name):

        # Train model
        model.fit(X_train, y_train)

        # Predictions
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_proba)

        # Log parameters & metrics
        mlflow.log_param("model_name", model_name)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("roc_auc", roc_auc)

        # Log model artifact
        mlflow.sklearn.log_model(model, artifact_path="model")

        print(f"{model_name} -> Accuracy: {accuracy:.4f}, ROC-AUC: {roc_auc:.4f}")


if __name__ == "__main__":

    # Set MLflow experiment
    mlflow.set_experiment("Heart-Disease-MLOps")

    # Dataset path
    DATA_PATH = "data/heart.csv"

    # Load & preprocess data
    X_train, X_test, y_train, y_test, scaler = load_and_preprocess_data(DATA_PATH)

    # Define models
    models = {
        "Logistic_Regression": LogisticRegression(max_iter=1000),
        "Random_Forest": RandomForestClassifier(
            n_estimators=100, random_state=42
        )
    }

    # Train and log each model
    for model_name, model in models.items():
        train_and_log_model(
            model, model_name, X_train, X_test, y_train, y_test
        )
