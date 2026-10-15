"""
evaluate_models.py
------------------
Evaluates trained classifiers and returns a metrics summary DataFrame.
"""

import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, classification_report
)


def evaluate_model(model, X_test, y_test, model_name: str = "Model") -> dict:
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "Model": model_name,
        "Accuracy": round(accuracy_score(y_test, y_pred), 4),
        "Precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "Recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
        "F1-Score": round(f1_score(y_test, y_pred, zero_division=0), 4),
        "ROC-AUC": round(roc_auc_score(y_test, y_prob), 4),
    }
    return metrics


def evaluate_all_models(models: dict, X_test, y_test) -> pd.DataFrame:
    """
    Evaluate all models in the dict and return a summary DataFrame.
    models: {'model_name': sklearn_model, ...}
    """
    records = []
    for name, model in models.items():
        metrics = evaluate_model(model, X_test, y_test, model_name=name)
        records.append(metrics)
        print(f"\n{name}:")
        print(f"  Accuracy : {metrics['Accuracy']:.4f}")
        print(f"  Precision: {metrics['Precision']:.4f}")
        print(f"  Recall   : {metrics['Recall']:.4f}")
        print(f"  F1-Score : {metrics['F1-Score']:.4f}")
        print(f"  ROC-AUC  : {metrics['ROC-AUC']:.4f}")

    return pd.DataFrame(records).sort_values("ROC-AUC", ascending=False).reset_index(drop=True)


def get_best_model(models: dict, X_test, y_test) -> tuple:
    """Returns (best_model_name, best_model) based on ROC-AUC."""
    summary = evaluate_all_models(models, X_test, y_test)
    best_name = summary.iloc[0]["Model"]
    return best_name, models[best_name], summary
