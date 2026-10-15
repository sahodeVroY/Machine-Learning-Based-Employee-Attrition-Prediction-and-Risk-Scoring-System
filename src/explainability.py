"""
explainability.py
-----------------
Feature importance and SHAP-based explainability helpers.
"""

import os
import numpy as np
import pandas as pd
import shap
import joblib


def get_feature_importance(model, feature_cols: list, top_n: int = 20) -> pd.DataFrame:
    """
    Extract feature importance from a tree-based model or logistic regression.
    Returns a DataFrame sorted by importance descending.
    """
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_[0])
    else:
        raise ValueError("Model does not expose feature_importances_ or coef_")

    df = pd.DataFrame({
        "Feature": feature_cols,
        "Importance": importances,
    }).sort_values("Importance", ascending=False).head(top_n).reset_index(drop=True)
    return df


def compute_shap_values(model, X: pd.DataFrame, max_samples: int = 200):
    """
    Compute SHAP values for a sample of the dataset.
    Returns (shap_values, explainer, X_sample).
    """
    X_sample = X.sample(min(max_samples, len(X)), random_state=42).reset_index(drop=True)

    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)
        # For classifiers, shap_values may be a list; take class 1
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
    except Exception:
        # Fall back to KernelExplainer for non-tree models
        background = shap.sample(X, 50)
        explainer = shap.KernelExplainer(model.predict_proba, background)
        shap_values = explainer.shap_values(X_sample)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]

    return shap_values, explainer, X_sample


def explain_single_employee(model, employee_row: pd.DataFrame, feature_cols: list, top_n: int = 8) -> pd.DataFrame:
    """
    Return top contributing features for a single employee.
    """
    try:
        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(employee_row)
        if isinstance(sv, list):
            sv = sv[1]
        contribs = sv[0]
    except Exception:
        # Fallback: use raw feature importance
        fi = get_feature_importance(model, feature_cols, top_n=top_n)
        return fi

    df = pd.DataFrame({
        "Feature": feature_cols,
        "Contribution": contribs,
        "AbsContrib": np.abs(contribs),
    }).sort_values("AbsContrib", ascending=False).head(top_n).reset_index(drop=True)
    df["Direction"] = df["Contribution"].apply(lambda x: "Increases Risk ↑" if x > 0 else "Decreases Risk ↓")
    return df
