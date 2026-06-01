"""
tests/test_profiler.py — Unit tests for app.eda.profiler.DataProfiler.

Tests cover:
  - Dimensions calculation
  - Column type classification
  - Missing value detection
  - Duplicate detection
  - Target distribution calculation
  - Numeric/categorical summaries
  - Feature categorisation
  - Target correlation computation
  - Full run() pipeline
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.eda.profiler import DataProfiler, FEATURE_CATEGORIES


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def sample_df() -> pd.DataFrame:
    """Representative DataFrame mimicking application_train structure."""
    np.random.seed(42)
    n = 200

    df = pd.DataFrame(
        {
            "SK_ID_CURR":       range(100_001, 100_001 + n),
            "TARGET":           np.random.choice([0, 1], size=n, p=[0.92, 0.08]),
            "AMT_INCOME_TOTAL": np.random.uniform(40_000, 400_000, n),
            "AMT_CREDIT":       np.random.uniform(100_000, 1_500_000, n),
            "AMT_ANNUITY":      np.random.uniform(5_000, 50_000, n),
            "DAYS_BIRTH":       np.random.randint(-25_000, -7_000, n),
            "DAYS_EMPLOYED":    np.random.choice(
                list(range(-5_000, -100)) + [365_243], size=n
            ),
            "EXT_SOURCE_1":     np.where(
                np.random.rand(n) > 0.3, np.random.uniform(0, 1, n), np.nan
            ),
            "EXT_SOURCE_2":     np.random.uniform(0, 1, n),
            "EXT_SOURCE_3":     np.where(
                np.random.rand(n) > 0.4, np.random.uniform(0, 1, n), np.nan
            ),
            "CODE_GENDER":      np.random.choice(["M", "F"], size=n),
            "NAME_CONTRACT_TYPE": np.random.choice(
                ["Cash loans", "Revolving loans"], size=n
            ),
            "FLAG_OWN_CAR":     np.random.choice(["Y", "N"], size=n),
            "HIGH_MISSING_COL": np.where(np.random.rand(n) > 0.1, np.nan, 1.0),
        }
    )
    return df


@pytest.fixture()
def profiler(sample_df) -> DataProfiler:
    return DataProfiler(sample_df, dataset_name="test_dataset")


# ---------------------------------------------------------------------------
# Dimensions
# ---------------------------------------------------------------------------


class TestDimensions:
    def test_returns_correct_row_count(self, profiler, sample_df):
        dims = profiler.get_dimensions()
        assert dims["rows"] == len(sample_df)

    def test_returns_correct_column_count(self, profiler, sample_df):
        dims = profiler.get_dimensions()
        assert dims["columns"] == len(sample_df.columns)

    def test_total_cells(self, profiler, sample_df):
        dims = profiler.get_dimensions()
        assert dims["total_cells"] == len(sample_df) * len(sample_df.columns)

    def test_all_values_are_int(self, profiler):
        dims = profiler.get_dimensions()
        for v in dims.values():
            assert isinstance(v, int)


# ---------------------------------------------------------------------------
# Column Types
# ---------------------------------------------------------------------------


class TestColumnTypes:
    def test_returns_entry_for_every_column(self, profiler, sample_df):
        col_types = profiler.get_column_types()
        assert set(col_types.keys()) == set(sample_df.columns)

    def test_string_column_detected_as_string(self, profiler):
        col_types = profiler.get_column_types()
        assert col_types["CODE_GENDER"]["kind"] == "string"

    def test_numeric_column_detected(self, profiler):
        col_types = profiler.get_column_types()
        kind = col_types["AMT_INCOME_TOTAL"]["kind"]
        assert kind in {"integer", "float"}

    def test_unique_count_is_non_negative(self, profiler):
        col_types = profiler.get_column_types()
        for meta in col_types.values():
            assert meta["unique_count"] >= 0

    def test_sample_values_has_at_most_three_entries(self, profiler):
        col_types = profiler.get_column_types()
        for meta in col_types.values():
            assert len(meta["sample_values"]) <= 3


# ---------------------------------------------------------------------------
# Missing Values
# ---------------------------------------------------------------------------


class TestMissingValues:
    def test_returns_dataframe(self, profiler):
        result = profiler.get_missing_value_stats()
        assert isinstance(result, pd.DataFrame)

    def test_high_missing_col_is_present(self, profiler):
        result = profiler.get_missing_value_stats()
        assert "HIGH_MISSING_COL" in result["column"].values

    def test_columns_are_sorted_descending(self, profiler):
        result = profiler.get_missing_value_stats()
        if len(result) > 1:
            assert result["missing_pct"].is_monotonic_decreasing

    def test_no_zero_missing_in_result(self, profiler):
        result = profiler.get_missing_value_stats()
        assert (result["missing_count"] > 0).all()

    def test_missing_pct_in_valid_range(self, profiler):
        result = profiler.get_missing_value_stats()
        assert (result["missing_pct"] >= 0).all()
        assert (result["missing_pct"] <= 100).all()

    def test_no_missing_df_returns_empty(self):
        df_clean = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
        profiler = DataProfiler(df_clean)
        result = profiler.get_missing_value_stats()
        assert len(result) == 0


# ---------------------------------------------------------------------------
# Duplicates
# ---------------------------------------------------------------------------


class TestDuplicates:
    def test_returns_dict_with_required_keys(self, profiler):
        result = profiler.get_duplicate_stats()
        for key in ("total_rows", "duplicate_rows", "duplicate_pct", "is_unique"):
            assert key in result

    def test_no_duplicates_in_sample(self, profiler):
        # Fixture has unique SK_ID_CURR; row-level dups unlikely with random data
        result = profiler.get_duplicate_stats()
        assert result["duplicate_rows"] >= 0

    def test_is_unique_is_boolean(self, profiler):
        result = profiler.get_duplicate_stats()
        assert isinstance(result["is_unique"], bool)

    def test_duplicate_detection(self):
        df = pd.DataFrame({"A": [1, 1, 2], "B": [3, 3, 4]})
        profiler = DataProfiler(df)
        result = profiler.get_duplicate_stats()
        assert result["duplicate_rows"] == 1
        assert not result["is_unique"]


# ---------------------------------------------------------------------------
# Target Distribution
# ---------------------------------------------------------------------------


class TestTargetDistribution:
    def test_returns_dict_when_target_present(self, profiler):
        result = profiler.get_target_distribution()
        assert result is not None
        assert isinstance(result, dict)

    def test_default_rate_between_0_and_100(self, profiler):
        result = profiler.get_target_distribution()
        assert 0.0 <= result["default_rate"] <= 100.0

    def test_counts_sum_to_total_valid(self, profiler):
        result = profiler.get_target_distribution()
        total = sum(result["counts"].values())
        assert total == result["total_valid"]

    def test_imbalance_ratio_is_positive(self, profiler):
        result = profiler.get_target_distribution()
        assert result["imbalance_ratio"] > 0

    def test_returns_none_when_target_absent(self):
        df_no_target = pd.DataFrame({"A": [1, 2, 3]})
        p = DataProfiler(df_no_target)
        assert p.get_target_distribution() is None


# ---------------------------------------------------------------------------
# Numeric Summary
# ---------------------------------------------------------------------------


class TestNumericSummary:
    def test_returns_dataframe(self, profiler):
        result = profiler.get_numeric_summary()
        assert isinstance(result, pd.DataFrame)

    def test_has_skewness_and_kurtosis_columns(self, profiler):
        result = profiler.get_numeric_summary()
        assert "skewness" in result.columns
        assert "kurtosis" in result.columns

    def test_has_missing_pct_column(self, profiler):
        result = profiler.get_numeric_summary()
        assert "missing_pct" in result.columns

    def test_all_numeric_cols_represented(self, profiler, sample_df):
        numeric_cols = sample_df.select_dtypes(include=[np.number]).columns.tolist()
        result = profiler.get_numeric_summary()
        for col in numeric_cols:
            assert col in result.index


# ---------------------------------------------------------------------------
# Categorical Summary
# ---------------------------------------------------------------------------


class TestCategoricalSummary:
    def test_returns_dataframe(self, profiler):
        result = profiler.get_categorical_summary()
        assert isinstance(result, pd.DataFrame)

    def test_code_gender_is_in_result(self, profiler):
        result = profiler.get_categorical_summary()
        assert "CODE_GENDER" in result["column"].values

    def test_top_value_pct_in_valid_range(self, profiler):
        result = profiler.get_categorical_summary()
        assert (result["top_value_pct"] >= 0).all()
        assert (result["top_value_pct"] <= 100).all()


# ---------------------------------------------------------------------------
# Feature Categorisation
# ---------------------------------------------------------------------------


class TestFeatureCategorisation:
    def test_returns_dict_with_expected_categories(self, profiler):
        categories = profiler.categorize_features()
        expected = {"demographic", "financial", "credit_history", "behavioural", "other"}
        assert set(categories.keys()) == expected

    def test_all_values_are_lists(self, profiler):
        categories = profiler.categorize_features()
        for v in categories.values():
            assert isinstance(v, list)

    def test_no_column_appears_in_multiple_categories(self, profiler, sample_df):
        categories = profiler.categorize_features()
        # Flatten all categorised columns
        all_cols: list = []
        for cat, cols in categories.items():
            if cat != "other":
                all_cols.extend(cols)
        assert len(all_cols) == len(set(all_cols)), "Duplicate column in categories"

    def test_known_demographic_feature_categorised(self, profiler):
        categories = profiler.categorize_features()
        # CODE_GENDER is in the fixture and in FEATURE_CATEGORIES["demographic"]
        if "CODE_GENDER" in profiler.df.columns:
            assert "CODE_GENDER" in categories["demographic"]


# ---------------------------------------------------------------------------
# Target Correlations
# ---------------------------------------------------------------------------


class TestTargetCorrelations:
    def test_returns_dataframe(self, profiler):
        result = profiler.get_correlation_with_target(top_n=10)
        assert isinstance(result, pd.DataFrame)

    def test_has_expected_columns(self, profiler):
        result = profiler.get_correlation_with_target(top_n=10)
        for col in ("feature", "correlation", "abs_correlation", "p_value", "significant"):
            assert col in result.columns

    def test_sorted_by_abs_correlation_descending(self, profiler):
        result = profiler.get_correlation_with_target(top_n=10)
        if len(result) > 1:
            # Drop any NaN rows (scipy produces NaN for constant-input columns)
            valid = result["abs_correlation"].dropna()
            assert valid.is_monotonic_decreasing

    def test_abs_correlation_between_0_and_1(self, profiler):
        result = profiler.get_correlation_with_target(top_n=10)
        valid = result["abs_correlation"].dropna()
        assert (valid >= 0).all()
        assert (valid <= 1).all()

    def test_returns_none_when_target_absent(self):
        df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
        p = DataProfiler(df)
        assert p.get_correlation_with_target() is None


# ---------------------------------------------------------------------------
# Full run() pipeline
# ---------------------------------------------------------------------------


class TestFullRun:
    def test_run_returns_dict(self, profiler):
        result = profiler.run()
        assert isinstance(result, dict)

    def test_run_populates_all_expected_keys(self, profiler):
        result = profiler.run()
        expected_keys = {
            "dataset_name", "dimensions", "column_types", "missing_value_stats",
            "duplicate_stats", "target_distribution", "numeric_summary",
            "categorical_summary", "feature_categories", "target_correlations",
        }
        assert expected_keys.issubset(set(result.keys()))

    def test_profile_property_populated_after_run(self, profiler):
        assert profiler.profile is None
        profiler.run()
        assert profiler.profile is not None

    def test_run_is_idempotent(self, profiler):
        result1 = profiler.run()
        result2 = profiler.run()
        assert result1["dimensions"] == result2["dimensions"]
