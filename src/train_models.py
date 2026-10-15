"""
train_models.py
---------------
Trains Logistic Regression, Random Forest, and XGBoost classifiers.
Saves models to the models/ directory.
"""

import os
import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")


def train_logistic_regression(X_train, y_train, random_state: int = 42):
    model = LogisticRegression(max_iter=1000, random_state=random_state, class_weight="balanced")
    model.fit(X_train, y_train)
    return model


def train_random_forest(X_train, y_train, random_state: int = 42):
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def train_xgboost(X_train, y_train, random_state: int = 42):
    scale_pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    model = XGBClassifier(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_train, y_train, verbose=False)
    return model


def train_all_models(X_train, y_train, save: bool = True) -> dict:
    """
    Train all three classifiers and optionally save them.
    Returns a dict: {'logistic_regression': model, 'random_forest': model, 'xgboost': model}
    """
    print("Training Logistic Regression...")
    lr = train_logistic_regression(X_train, y_train)

    print("Training Random Forest...")
    rf = train_random_forest(X_train, y_train)

    print("Training XGBoost...")
    xgb = train_xgboost(X_train, y_train)

    models = {
        "logistic_regression": lr,
        "random_forest": rf,
        "xgboost": xgb,
    }

    if save:
        os.makedirs(MODELS_DIR, exist_ok=True)
        for name, model in models.items():
            joblib.dump(model, os.path.join(MODELS_DIR, f"{name}.pkl"))
            print(f"  Saved {name}.pkl")

    return models
