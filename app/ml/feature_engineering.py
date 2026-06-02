"""
app/ml/feature_engineering.py — Derived ratio features for credit risk modelling.

Creates domain-specific features with safe divide-by-zero handling.
Implemented as a scikit-learn compatible transformer (stateless).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from app.ml.settings import DAYS_EMPLOYED_SENTINEL, ENGINEERED_FEATURES
from app.utils.logger import get_logger

logger = get_logger(__name__)


def _safe_ratio(
    numerator: pd.Series,
    denominator: pd.Series,
) -> pd.Series:
    """
    Compute ``numerator / denominator`` with divide-by-zero protection.

    Returns NaN where the denominator is zero, null, or non-finite.
    """
    denom = denominator.astype(float)
    num = numerator.astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        result = num / denom
    invalid = denom.isna() | (denom == 0) | ~np.isfinite(denom)
    result = result.where(~invalid, np.nan)
    return result


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Create engineered ratio features from raw Home Credit columns.

    Features created:
      - ``debt_to_income_ratio``:   (AMT_CREDIT + AMT_ANNUITY) / AMT_INCOME_TOTAL
      - ``credit_to_income_ratio``: AMT_CREDIT / AMT_INCOME_TOTAL
      - ``annuity_to_income_ratio``: AMT_ANNUITY / AMT_INCOME_TOTAL
      - ``employment_age_ratio``:   employment_years / age_years

    Missing source columns produce NaN in the derived feature rather than
    raising an error.

    Example::

        engineer = FeatureEngineer()
        X_new = engineer.fit_transform(X)
    """

    def __init__(self) -> None:
        self.created_features_: List[str] = []
        self.skipped_features_: Dict[str, str] = {}

    def fit(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
    ) -> "FeatureEngineer":
        """Record which features can be created from available columns."""
        self.created_features_ = []
        self.skipped_features_ = {}

        availability = {
            "debt_to_income_ratio": {"AMT_CREDIT", "AMT_ANNUITY", "AMT_INCOME_TOTAL"},
            "credit_to_income_ratio": {"AMT_CREDIT", "AMT_INCOME_TOTAL"},
            "annuity_to_income_ratio": {"AMT_ANNUITY", "AMT_INCOME_TOTAL"},
            "employment_age_ratio": {"DAYS_EMPLOYED", "DAYS_BIRTH"},
        }

        cols = set(X.columns)
        for feat in ENGINEERED_FEATURES:
            required = availability[feat]
            if required.issubset(cols):
                self.created_features_.append(feat)
            else:
                missing = required - cols
                self.skipped_features_[feat] = f"missing columns: {sorted(missing)}"

        logger.info(
            f"FeatureEngineer fit — {len(self.created_features_)} features will be created"
        )
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Add engineered features to a copy of *X*."""
        X_out = X.copy()

        if "credit_to_income_ratio" in self.created_features_:
            X_out["credit_to_income_ratio"] = _safe_ratio(
                X_out["AMT_CREDIT"], X_out["AMT_INCOME_TOTAL"]
            )

        if "annuity_to_income_ratio" in self.created_features_:
            X_out["annuity_to_income_ratio"] = _safe_ratio(
                X_out["AMT_ANNUITY"], X_out["AMT_INCOME_TOTAL"]
            )

        if "debt_to_income_ratio" in self.created_features_:
            total_debt = X_out["AMT_CREDIT"].astype(float) + X_out["AMT_ANNUITY"].astype(float)
            X_out["debt_to_income_ratio"] = _safe_ratio(
                total_debt, X_out["AMT_INCOME_TOTAL"]
            )

        if "employment_age_ratio" in self.created_features_:
            emp = X_out["DAYS_EMPLOYED"].astype(float).copy()
            emp = emp.replace(DAYS_EMPLOYED_SENTINEL, np.nan)
            emp_years = emp.abs() / 365.25
            age_years = X_out["DAYS_BIRTH"].astype(float).abs() / 365.25
            X_out["employment_age_ratio"] = _safe_ratio(emp_years, age_years)

        return X_out

    def get_feature_summary(self) -> Dict[str, Any]:
        """Return summary of created and skipped features."""
        return {
            "created_features": list(self.created_features_),
            "skipped_features": dict(self.skipped_features_),
        }
