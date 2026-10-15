"""
data_preprocessing.py
---------------------
Loads, cleans, encodes, scales, and splits the employee dataset.
Handles class imbalance via SMOTE.
"""

import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.over_sampling import SMOTE

DATA_RAW = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "employee_data.csv")
DATA_PROCESSED = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

# Columns that are constant / irrelevant
DROP_COLS = ["EmployeeCount", "EmployeeNumber", "Over18", "StandardHours"]

# Categorical columns for label encoding
ORDINAL_COLS = ["BusinessTravel", "Department", "EducationField", "Gender",
                "JobRole", "MaritalStatus", "OverTime"]


def load_raw_data(path: str = DATA_RAW) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    # Drop constant / identifier columns if present
    cols_to_drop = [c for c in DROP_COLS if c in df.columns]
    df = df.drop(columns=cols_to_drop)

    # Convert Attrition to binary if it is string
    if df["Attrition"].dtype == object:
        df["Attrition"] = df["Attrition"].map({"Yes": 1, "No": 0})

    # Convert OverTime to binary if it is string
    if "OverTime" in df.columns and df["OverTime"].dtype == object:
        df["OverTime"] = df["OverTime"].map({"Yes": 1, "No": 0})

    # Drop rows where target is null
    df = df.dropna(subset=["Attrition"])
    return df.reset_index(drop=True)


def encode_categoricals(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Label-encode remaining string columns. Returns df and encoder dict."""
    encoders: dict = {}
    str_cols = df.select_dtypes(include="object").columns.tolist()
    for col in str_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le
    return df, encoders


def scale_features(X_train: pd.DataFrame, X_test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)
    return X_train_scaled, X_test_scaled, scaler


def preprocess(path: str = DATA_RAW, test_size: float = 0.2, random_state: int = 42,
               apply_smote: bool = True, save: bool = True):
    """
    Full preprocessing pipeline.
    Returns X_train, X_test, y_train, y_test, scaler, encoders, feature_cols
    """
    df = load_raw_data(path)
    df = clean_data(df)
    df, encoders = encode_categoricals(df)

    # Feature engineering (imported lazily to avoid circular)
    from src.feature_engineering import add_engineered_features
    df = add_engineered_features(df)

    target = "Attrition"
    feature_cols = [c for c in df.columns if c != target]

    X = df[feature_cols]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    X_train, X_test, scaler = scale_features(X_train, X_test)

    if apply_smote:
        smote = SMOTE(random_state=random_state)
        X_train, y_train = smote.fit_resample(X_train, y_train)
        X_train = pd.DataFrame(X_train, columns=feature_cols)
        y_train = pd.Series(y_train, name=target)

    if save:
        os.makedirs(DATA_PROCESSED, exist_ok=True)
        os.makedirs(MODELS_DIR, exist_ok=True)
        processed_df = pd.concat([X_test.assign(Attrition=y_test.values)], axis=1)
        processed_df.to_csv(os.path.join(DATA_PROCESSED, "processed_employee_data.csv"), index=False)
        joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.pkl"))
        joblib.dump(encoders, os.path.join(MODELS_DIR, "encoders.pkl"))
        joblib.dump(feature_cols, os.path.join(MODELS_DIR, "feature_cols.pkl"))

    return X_train, X_test, y_train, y_test, scaler, encoders, feature_cols
