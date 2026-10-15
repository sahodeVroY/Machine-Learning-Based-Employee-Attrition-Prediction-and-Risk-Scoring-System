"""
run_pipeline.py
---------------
End-to-end pipeline: preprocess → train → evaluate → risk score → save.
Run this once to generate all model artifacts before launching the dashboard.

Usage:
    python run_pipeline.py
"""

import os
import sys
import joblib
import pandas as pd

# Make sure src is importable
sys.path.insert(0, os.path.dirname(__file__))

from src.data_preprocessing import preprocess, MODELS_DIR
from src.train_models import train_all_models
from src.evaluate_models import get_best_model
from src.risk_scoring import score_employees
from src.explainability import get_feature_importance


def main():
    print("=" * 60)
    print("  Employee Attrition Prediction — Pipeline")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Step 1 — Preprocess
    # ------------------------------------------------------------------
    print("\n[1/4] Preprocessing data...")
    X_train, X_test, y_train, y_test, scaler, encoders, feature_cols = preprocess(save=True)
    print(f"      Train size (after SMOTE): {len(X_train):,}  |  Test size: {len(X_test):,}")
    print(f"      Features: {len(feature_cols)}")

    # ------------------------------------------------------------------
    # Step 2 — Train
    # ------------------------------------------------------------------
    print("\n[2/4] Training models...")
    models = train_all_models(X_train, y_train, save=True)

    # ------------------------------------------------------------------
    # Step 3 — Evaluate
    # ------------------------------------------------------------------
    print("\n[3/4] Evaluating models...")
    best_name, best_model, summary = get_best_model(models, X_test, y_test)
    print(f"\n  Best model by ROC-AUC: {best_name}")
    print(summary.to_string(index=False))

    # Save summary
    summary_path = os.path.join(os.path.dirname(__file__), "models", "model_metrics.csv")
    summary.to_csv(summary_path, index=False)
    print(f"\n  Metrics saved to {summary_path}")

    # Save best model name so the app can load it
    joblib.dump(best_name, os.path.join(MODELS_DIR, "best_model_name.pkl"))

    # ------------------------------------------------------------------
    # Step 4 — Risk Score entire test set & save
    # ------------------------------------------------------------------
    print("\n[4/4] Generating risk scores for test set...")
    risk_df = score_employees(best_model, X_test)

    # Merge with original test features
    test_with_risk = pd.concat([
        X_test.reset_index(drop=True),
        pd.Series(y_test.values, name="Attrition"),
        risk_df.drop(columns=["EmployeeIndex"]),
    ], axis=1)

    output_path = os.path.join(os.path.dirname(__file__), "data", "processed", "risk_scored_employees.csv")
    test_with_risk.to_csv(output_path, index=False)
    print(f"  Risk scores saved to {output_path}")

    # Feature importance
    fi = get_feature_importance(best_model, feature_cols)
    fi_path = os.path.join(os.path.dirname(__file__), "models", "feature_importance.csv")
    fi.to_csv(fi_path, index=False)
    print(f"  Feature importance saved to {fi_path}")

    print("\n" + "=" * 60)
    print("  Pipeline complete.")
    print(f"  Launch dashboard: streamlit run app/app.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
