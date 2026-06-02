"""
tests/test_feature_selector.py — Unit tests for app.ml.feature_selector.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.ml.feature_selector import FeatureSelector


@pytest.fixture()
def sample_df() -> pd.DataFrame:
    n = 100
    return pd.DataFrame(
        {
            "SK_ID_CURR": range(n),
            "TARGET": [0, 1] * (n // 2),
            "GOOD_FEATURE": np.random.uniform(0, 1, n),
            "HIGH_MISSING": [np.nan if i < 85 else 1.0 for i in range(n)],
            "ZERO_VARIANCE": [5.0] * n,
            "LOW_VARIANCE": [1.0, 1.0, 1.0, 2.0] + [1.0] * (n - 4),
        }
    )


class TestFeatureSelector:
    def test_drops_high_missing_columns(self, sample_df):
        selector = FeatureSelector(missing_threshold=0.80)
        result = selector.fit_transform(sample_df)
        assert "HIGH_MISSING" not in result.columns

    def test_drops_zero_variance_columns(self, sample_df):
        selector = FeatureSelector(missing_threshold=0.80)
        result = selector.fit_transform(sample_df)
        assert "ZERO_VARIANCE" not in result.columns

    def test_preserves_id_and_target(self, sample_df):
        selector = FeatureSelector(missing_threshold=0.80)
        result = selector.fit_transform(sample_df)
        assert "SK_ID_CURR" in result.columns
        assert "TARGET" in result.columns

    def test_preserves_informative_features(self, sample_df):
        selector = FeatureSelector(missing_threshold=0.80)
        result = selector.fit_transform(sample_df)
        assert "GOOD_FEATURE" in result.columns

    def test_selection_report_populated(self, sample_df):
        selector = FeatureSelector(missing_threshold=0.80)
        selector.fit(sample_df)
        report = selector.get_selection_report()
        assert report["total_dropped"] >= 2
        assert len(report["dropped_high_missing"]) >= 1
        assert "ZERO_VARIANCE" in report["dropped_zero_variance"]

    def test_transform_raises_on_missing_columns(self, sample_df):
        selector = FeatureSelector(missing_threshold=0.80)
        selector.fit(sample_df)
        bad_df = sample_df.drop(columns=["GOOD_FEATURE"])
        with pytest.raises(ValueError, match="missing expected columns"):
            selector.transform(bad_df)

    def test_same_columns_on_train_and_test(self, sample_df):
        train = sample_df.iloc[:80].copy()
        test = sample_df.iloc[80:].copy()
        selector = FeatureSelector(missing_threshold=0.80)
        selector.fit(train)
        train_out = selector.transform(train)
        test_out = selector.transform(test)
        assert list(train_out.columns) == list(test_out.columns)
