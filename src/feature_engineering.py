"""
feature_engineering.py
-----------------------
Adds engineered features to the cleaned, numerically encoded DataFrame.
"""

import pandas as pd
import numpy as np


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds:
    - IncomeExperienceRatio: MonthlyIncome / (TotalWorkingYears + 1)
    - PromotionDelayFlag: 1 if YearsSinceLastPromotion >= 4
    - EngagementScore: average of JobSatisfaction, EnvironmentSatisfaction,
                       RelationshipSatisfaction, WorkLifeBalance, JobInvolvement
    - WorkloadStressFlag: 1 if OverTime == 1
    """
    df = df.copy()

    # Income-to-Experience Ratio
    if "MonthlyIncome" in df.columns and "TotalWorkingYears" in df.columns:
        df["IncomeExperienceRatio"] = df["MonthlyIncome"] / (df["TotalWorkingYears"] + 1)

    # Promotion Delay Indicator
    if "YearsSinceLastPromotion" in df.columns:
        df["PromotionDelayFlag"] = (df["YearsSinceLastPromotion"] >= 4).astype(int)

    # Engagement Composite Score
    engagement_cols = ["JobSatisfaction", "EnvironmentSatisfaction",
                       "RelationshipSatisfaction", "WorkLifeBalance", "JobInvolvement"]
    present_eng = [c for c in engagement_cols if c in df.columns]
    if present_eng:
        df["EngagementScore"] = df[present_eng].mean(axis=1)

    # Workload Stress Flag (OverTime already binary)
    if "OverTime" in df.columns:
        df["WorkloadStressFlag"] = df["OverTime"].astype(int)

    return df
