"""
app/ml/feature_selector.py — Remove high-missing and zero-variance columns.

Fits on training data only to prevent leakage.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from app.ml.settings import HIGH_MISSING_THRESHOLD, ID_COLUMNS, TARGET_COLUMN
from app.utils.logger import get_logger

logger = get_logger(__name__)


class FeatureSelector(BaseEstimator, TransformerMixin):
    """
    Drop columns with excessive missing values or zero variance.

    Columns in ``ID_COLUMNS`` and ``TARGET_COLUMN`` are always preserved
    during selection (they are excluded from the feature matrix later).

    Args:
        missing_threshold: Fraction in (0, 1] above which a column is dropped.

    Example::

        selector = FeatureSelector(missing_threshold=0.80)
        X_selected = selector.fit_transform(X_train)
    """

    def __init__(self, missing_threshold: float = HIGH_MISSING_THRESHOLD) -> None:
        self.missing_threshold = missing_threshold
        self.dropped_high_missing_: List[str] = []
        self.dropped_zero_variance_: List[str] = []
        self.selected_features_: List[str] = []
        self.selection_report_: Dict[str, Any] = {}

    def fit(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
    ) -> "FeatureSelector":
        """
        Identify columns to drop based on missing rate and variance.

        Args:
            X: Training feature DataFrame (may include ID/TARGET columns).
            y: Unused; present for sklearn Pipeline compatibility.

        Returns:
            Fitted selector.
        """
        protected: Set[str] = set(ID_COLUMNS) | {TARGET_COLUMN}
        candidate_cols = [c for c in X.columns if c not in protected]

        missing_rates = X[candidate_cols].isnull().mean()
        self.dropped_high_missing_ = [
            col
            for col in candidate_cols
            if missing_rates[col] > self.missing_threshold
        ]

        remaining = [
            c for c in candidate_cols if c not in self.dropped_high_missing_
        ]
        self.dropped_zero_variance_ = []

        for col in remaining:
            series = X[col].dropna()
            if len(series) == 0:
                self.dropped_zero_variance_.append(col)
                continue
            if pd.api.types.is_numeric_dtype(X[col]):
                if series.nunique() <= 1 or float(series.std()) == 0.0:
                    self.dropped_zero_variance_.append(col)
            else:
                if series.nunique() <= 1:
                    self.dropped_zero_variance_.append(col)

        drop_set = set(self.dropped_high_missing_) | set(self.dropped_zero_variance_)
        self.selected_features_ = [c for c in X.columns if c not in drop_set]

        self.selection_report_ = {
            "missing_threshold": self.missing_threshold,
            "dropped_high_missing": [
                {
                    "column": col,
                    "missing_pct": round(float(missing_rates[col] * 100), 2),
                }
                for col in self.dropped_high_missing_
            ],
            "dropped_zero_variance": list(self.dropped_zero_variance_),
            "input_feature_count": len(candidate_cols),
            "output_feature_count": len(self.selected_features_) - len(
                protected & set(self.selected_features_)
            ),
            "total_dropped": len(drop_set),
        }

        logger.info(
            f"FeatureSelector — dropped {len(self.dropped_high_missing_)} high-missing, "
            f"{len(self.dropped_zero_variance_)} zero-variance columns"
        )
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Return DataFrame restricted to selected columns."""
        missing_in_transform = set(self.selected_features_) - set(X.columns)
        # TARGET is optional at inference time (unlabelled scoring data)
        required_missing = missing_in_transform - {TARGET_COLUMN}
        if required_missing:
            raise ValueError(
                f"Transform input missing expected columns: {sorted(required_missing)}"
            )
        available = [c for c in self.selected_features_ if c in X.columns]
        return X[available].copy()

    def get_selection_report(self) -> Dict[str, Any]:
        """Return the feature selection report from the last ``fit()``."""
        return dict(self.selection_report_)
