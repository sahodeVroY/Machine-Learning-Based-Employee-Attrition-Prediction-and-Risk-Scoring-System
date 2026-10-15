"""
risk_scoring.py
---------------
Converts attrition probabilities into Low / Medium / High risk categories.
"""

import pandas as pd
import numpy as np


def assign_risk_category(prob: float, low_threshold: float = 0.30, high_threshold: float = 0.60) -> str:
    """Map a single probability to a risk label."""
    if prob < low_threshold:
        return "Low Risk"
    elif prob <= high_threshold:
        return "Medium Risk"
    else:
        return "High Risk"


def score_employees(
    model,
    X: pd.DataFrame,
    employee_ids=None,
    low_threshold: float = 0.30,
    high_threshold: float = 0.60,
) -> pd.DataFrame:
    """
    Generate attrition probability and risk category for each employee.

    Parameters
    ----------
    model        : trained sklearn-compatible classifier
    X            : feature DataFrame (already scaled)
    employee_ids : optional index / IDs to attach to the result
    low_threshold  : probability below which employee is Low Risk
    high_threshold : probability above which employee is High Risk

    Returns
    -------
    pd.DataFrame with columns: EmployeeIndex, AttritionProbability, RiskCategory
    """
    probs = model.predict_proba(X)[:, 1]
    categories = [assign_risk_category(p, low_threshold, high_threshold) for p in probs]

    result = pd.DataFrame({
        "AttritionProbability": probs,
        "RiskCategory": categories,
    })

    if employee_ids is not None:
        result.insert(0, "EmployeeIndex", employee_ids)
    else:
        result.insert(0, "EmployeeIndex", range(len(probs)))

    return result


RISK_COLOR_MAP = {
    "Low Risk": "#22c55e",     # green
    "Medium Risk": "#f59e0b",  # amber
    "High Risk": "#ef4444",    # red
}
