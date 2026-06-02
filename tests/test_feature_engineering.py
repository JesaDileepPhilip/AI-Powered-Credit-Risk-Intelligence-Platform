"""
tests/test_feature_engineering.py — Unit tests for app.ml.feature_engineering.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.ml.feature_engineering import FeatureEngineer, _safe_ratio


@pytest.fixture()
def sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "SK_ID_CURR": [1, 2, 3, 4],
            "TARGET": [0, 1, 0, 1],
            "AMT_INCOME_TOTAL": [100_000.0, 200_000.0, 0.0, 150_000.0],
            "AMT_CREDIT": [50_000.0, 80_000.0, 30_000.0, np.nan],
            "AMT_ANNUITY": [5_000.0, 10_000.0, 3_000.0, 7_000.0],
            "DAYS_BIRTH": [-10_000, -15_000, -8_000, -12_000],
            "DAYS_EMPLOYED": [-1_000, 365_243, -500, -2_000],
        }
    )


class TestSafeRatio:
    def test_normal_division(self):
        num = pd.Series([10.0, 20.0])
        denom = pd.Series([2.0, 4.0])
        result = _safe_ratio(num, denom)
        assert list(result) == [5.0, 5.0]

    def test_zero_denominator_returns_nan(self):
        num = pd.Series([10.0])
        denom = pd.Series([0.0])
        result = _safe_ratio(num, denom)
        assert np.isnan(result.iloc[0])

    def test_null_denominator_returns_nan(self):
        num = pd.Series([10.0])
        denom = pd.Series([np.nan])
        result = _safe_ratio(num, denom)
        assert np.isnan(result.iloc[0])


class TestFeatureEngineer:
    def test_fit_creates_all_features_when_columns_present(self, sample_df):
        engineer = FeatureEngineer()
        engineer.fit(sample_df)
        assert len(engineer.created_features_) == 4

    def test_transform_adds_engineered_columns(self, sample_df):
        engineer = FeatureEngineer()
        result = engineer.fit_transform(sample_df)
        for col in (
            "debt_to_income_ratio",
            "credit_to_income_ratio",
            "annuity_to_income_ratio",
            "employment_age_ratio",
        ):
            assert col in result.columns

    def test_credit_to_income_ratio_correct(self, sample_df):
        engineer = FeatureEngineer()
        result = engineer.fit_transform(sample_df)
        expected = 50_000.0 / 100_000.0
        assert result.loc[0, "credit_to_income_ratio"] == pytest.approx(expected)

    def test_zero_income_produces_nan_ratio(self, sample_df):
        engineer = FeatureEngineer()
        result = engineer.fit_transform(sample_df)
        assert np.isnan(result.loc[2, "credit_to_income_ratio"])

    def test_employment_sentinel_produces_nan_ratio(self, sample_df):
        engineer = FeatureEngineer()
        result = engineer.fit_transform(sample_df)
        assert np.isnan(result.loc[1, "employment_age_ratio"])

    def test_skips_features_when_source_columns_missing(self):
        df = pd.DataFrame({"SK_ID_CURR": [1], "TARGET": [0], "AMT_CREDIT": [100.0]})
        engineer = FeatureEngineer()
        engineer.fit(df)
        result = engineer.transform(df)
        assert "credit_to_income_ratio" not in result.columns
        assert len(engineer.skipped_features_) > 0

    def test_get_feature_summary(self, sample_df):
        engineer = FeatureEngineer()
        engineer.fit(sample_df)
        summary = engineer.get_feature_summary()
        assert "created_features" in summary
        assert len(summary["created_features"]) == 4
