"""
tests/test_insights.py — Unit tests for app.eda.insights.BusinessInsightGenerator.

Tests cover:
  - Each individual analyze_* method
  - Graceful handling of missing columns
  - Correct severity levels
  - run() pipeline completeness
  - to_dataframe() output shape
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import pytest

from app.eda.insights import BusinessInsightGenerator


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def full_df() -> pd.DataFrame:
    """Full-featured DataFrame with all key columns present."""
    np.random.seed(0)
    n = 500

    age_days   = np.random.randint(-25_000, -7_000, n)
    emp_days   = np.where(
        np.random.rand(n) > 0.1,
        np.random.randint(-4_000, -100, n),
        365_243,  # sentinel
    )

    return pd.DataFrame(
        {
            "SK_ID_CURR":       range(100_001, 100_001 + n),
            "TARGET":           np.random.choice([0, 1], size=n, p=[0.92, 0.08]),
            "AMT_INCOME_TOTAL": np.random.uniform(40_000, 400_000, n),
            "AMT_CREDIT":       np.random.uniform(100_000, 1_500_000, n),
            "AMT_ANNUITY":      np.random.uniform(5_000, 50_000, n),
            "DAYS_BIRTH":       age_days,
            "DAYS_EMPLOYED":    emp_days,
            "EXT_SOURCE_1":     np.where(np.random.rand(n) > 0.25, np.random.uniform(0, 1, n), np.nan),
            "EXT_SOURCE_2":     np.random.uniform(0, 1, n),
            "EXT_SOURCE_3":     np.where(np.random.rand(n) > 0.4, np.random.uniform(0, 1, n), np.nan),
            "CODE_GENDER":      np.random.choice(["M", "F"], size=n),
            "FLAG_DOCUMENT_3":  np.random.choice([0, 1], size=n),
            "FLAG_DOCUMENT_6":  np.random.choice([0, 1], size=n),
            "HIGH_MISSING_A":   np.where(np.random.rand(n) > 0.1, np.nan, 1.0),
            "HIGH_MISSING_B":   np.where(np.random.rand(n) > 0.15, np.nan, 2.0),
        }
    )


@pytest.fixture()
def bureau_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "SK_ID_CURR":         [100_001, 100_001, 100_002, 100_003],
            "SK_ID_BUREAU":       [5_000_001, 5_000_002, 5_000_003, 5_000_004],
            "CREDIT_ACTIVE":      ["Active", "Closed", "Active", "Active"],
            "CREDIT_DAY_OVERDUE": [0, 0, 7, 0],
            "AMT_CREDIT_SUM":     [135_000.0, 45_000.0, 270_000.0, 90_000.0],
            "CREDIT_TYPE":        ["Consumer credit", "Car loan", "Consumer credit", "Mortgage"],
        }
    )


@pytest.fixture()
def gen(full_df, bureau_df) -> BusinessInsightGenerator:
    return BusinessInsightGenerator(full_df, bureau_df=bureau_df)


@pytest.fixture()
def gen_no_bureau(full_df) -> BusinessInsightGenerator:
    return BusinessInsightGenerator(full_df, bureau_df=None)


# ---------------------------------------------------------------------------
# analyze_class_imbalance
# ---------------------------------------------------------------------------


class TestClassImbalance:
    def test_returns_dict_with_default_rate(self, gen):
        result = gen.analyze_class_imbalance()
        assert "default_rate" in result
        assert 0 <= result["default_rate"] <= 100

    def test_returns_imbalance_ratio(self, gen):
        result = gen.analyze_class_imbalance()
        assert "imbalance_ratio" in result
        assert result["imbalance_ratio"] > 0

    def test_insight_appended(self, gen):
        initial = len(gen.insights)
        gen.analyze_class_imbalance()
        assert len(gen.insights) == initial + 1

    def test_critical_severity_for_high_imbalance(self):
        # Force 1:100 imbalance
        df = pd.DataFrame(
            {
                "TARGET":           [0] * 990 + [1] * 10,
                "SK_ID_CURR":       range(1000),
                "AMT_INCOME_TOTAL": [100_000] * 1000,
            }
        )
        g = BusinessInsightGenerator(df)
        g.analyze_class_imbalance()
        assert g.insights[-1]["severity"] == "critical"

    def test_empty_result_when_target_absent(self):
        df = pd.DataFrame({"AMT_CREDIT": [1, 2, 3]})
        g = BusinessInsightGenerator(df)
        result = g.analyze_class_imbalance()
        assert result == {}
        assert len(g.insights) == 0


# ---------------------------------------------------------------------------
# analyze_income_patterns
# ---------------------------------------------------------------------------


class TestIncomePatterns:
    def test_returns_median_income_keys(self, gen):
        result = gen.analyze_income_patterns()
        assert "median_income_default" in result
        assert "median_income_non_default" in result

    def test_income_gap_pct_is_float(self, gen):
        result = gen.analyze_income_patterns()
        assert isinstance(result["income_gap_pct"], float)

    def test_skips_when_columns_absent(self):
        df = pd.DataFrame({"TARGET": [0, 1]})
        g = BusinessInsightGenerator(df)
        result = g.analyze_income_patterns()
        assert result == {}


# ---------------------------------------------------------------------------
# analyze_credit_patterns
# ---------------------------------------------------------------------------


class TestCreditPatterns:
    def test_returns_mean_credit_keys(self, gen):
        result = gen.analyze_credit_patterns()
        assert "mean_credit_default" in result
        assert "mean_credit_non_default" in result

    def test_skips_when_credit_col_absent(self):
        df = pd.DataFrame({"TARGET": [0, 1], "AMT_INCOME_TOTAL": [100_000, 90_000]})
        g = BusinessInsightGenerator(df)
        result = g.analyze_credit_patterns()
        assert result == {}


# ---------------------------------------------------------------------------
# analyze_age_patterns
# ---------------------------------------------------------------------------


class TestAgePatterns:
    def test_returns_rate_keys(self, gen):
        result = gen.analyze_age_patterns()
        assert "young_default_rate" in result
        assert "senior_default_rate" in result
        assert "overall_default_rate" in result

    def test_rates_in_valid_range(self, gen):
        result = gen.analyze_age_patterns()
        for key in ("young_default_rate", "senior_default_rate", "overall_default_rate"):
            assert 0 <= result[key] <= 100

    def test_skips_when_days_birth_absent(self):
        df = pd.DataFrame({"TARGET": [0, 1], "AMT_CREDIT": [100_000, 90_000]})
        g = BusinessInsightGenerator(df)
        result = g.analyze_age_patterns()
        assert result == {}


# ---------------------------------------------------------------------------
# analyze_employment_patterns
# ---------------------------------------------------------------------------


class TestEmploymentPatterns:
    def test_detects_sentinel_values(self, gen, full_df):
        result = gen.analyze_employment_patterns()
        expected_anomaly = int((full_df["DAYS_EMPLOYED"] == 365_243).sum())
        assert result["anomaly_count"] == expected_anomaly

    def test_anomaly_pct_in_range(self, gen):
        result = gen.analyze_employment_patterns()
        assert 0 <= result["anomaly_pct"] <= 100

    def test_severity_is_warning(self, gen):
        initial = len(gen.insights)
        gen.analyze_employment_patterns()
        assert gen.insights[initial]["severity"] == "warning"

    def test_skips_when_days_employed_absent(self):
        df = pd.DataFrame({"TARGET": [0, 1]})
        g = BusinessInsightGenerator(df)
        result = g.analyze_employment_patterns()
        assert result == {}


# ---------------------------------------------------------------------------
# analyze_external_sources
# ---------------------------------------------------------------------------


class TestExternalSources:
    def test_returns_correlations_for_each_ext_source(self, gen):
        result = gen.analyze_external_sources()
        for col in ("EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"):
            assert col in result

    def test_correlations_in_valid_range(self, gen):
        result = gen.analyze_external_sources()
        for v in result.values():
            assert -1 <= v <= 1

    def test_skips_when_no_ext_sources(self):
        df = pd.DataFrame({"TARGET": [0, 1], "AMT_CREDIT": [100, 200]})
        g = BusinessInsightGenerator(df)
        result = g.analyze_external_sources()
        assert result == {}


# ---------------------------------------------------------------------------
# analyze_missing_values
# ---------------------------------------------------------------------------


class TestMissingValues:
    def test_returns_count_keys(self, gen):
        result = gen.analyze_missing_values()
        assert "high_missing_count" in result
        assert "moderate_missing_count" in result

    def test_high_missing_detected(self, gen):
        result = gen.analyze_missing_values()
        # Fixture has HIGH_MISSING_A and HIGH_MISSING_B with ~90% missing
        assert result["high_missing_count"] >= 2


# ---------------------------------------------------------------------------
# analyze_bureau_integration
# ---------------------------------------------------------------------------


class TestBureauIntegration:
    def test_returns_metrics_when_bureau_present(self, gen):
        result = gen.analyze_bureau_integration()
        assert "total_bureau_records" in result
        assert result["total_bureau_records"] == 4

    def test_returns_empty_when_bureau_absent(self, gen_no_bureau):
        result = gen_no_bureau.analyze_bureau_integration()
        assert result == {}

    def test_insight_not_appended_when_bureau_absent(self, gen_no_bureau):
        # Run all others first, then check bureau doesn't add
        initial = len(gen_no_bureau.insights)
        gen_no_bureau.analyze_bureau_integration()
        assert len(gen_no_bureau.insights) == initial


# ---------------------------------------------------------------------------
# analyze_document_flags
# ---------------------------------------------------------------------------


class TestDocumentFlags:
    def test_returns_dict_with_flag_rates(self, gen):
        result = gen.analyze_document_flags()
        assert isinstance(result, dict)
        # Fixture has FLAG_DOCUMENT_3 and FLAG_DOCUMENT_6
        assert "FLAG_DOCUMENT_3" in result or "FLAG_DOCUMENT_6" in result

    def test_rates_are_floats_in_valid_range(self, gen):
        result = gen.analyze_document_flags()
        for rate in result.values():
            assert 0.0 <= rate <= 100.0

    def test_skips_when_no_flag_columns(self):
        df = pd.DataFrame({"TARGET": [0, 1], "AMT_CREDIT": [100, 200]})
        g = BusinessInsightGenerator(df)
        result = g.analyze_document_flags()
        assert result == {}


# ---------------------------------------------------------------------------
# Full run() pipeline
# ---------------------------------------------------------------------------


class TestRunPipeline:
    def test_run_returns_list(self, gen):
        result = gen.run()
        assert isinstance(result, list)

    def test_run_produces_insights(self, gen):
        result = gen.run()
        assert len(result) > 0

    def test_all_insights_have_required_keys(self, gen):
        result = gen.run()
        required = {"category", "title", "finding", "recommendation", "severity"}
        for item in result:
            assert required.issubset(set(item.keys()))

    def test_all_severities_are_valid(self, gen):
        result = gen.run()
        valid_severities = {"info", "warning", "critical"}
        for item in result:
            assert item["severity"] in valid_severities

    def test_insights_property_matches_run_output(self, gen):
        result = gen.run()
        assert result == gen.insights

    def test_to_dataframe_shape(self, gen):
        gen.run()
        df = gen.to_dataframe()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == len(gen.insights)
        assert "severity" in df.columns

    def test_run_survives_column_gaps(self):
        """Pipeline should not crash when most columns are absent."""
        minimal_df = pd.DataFrame(
            {
                "SK_ID_CURR": [1, 2, 3],
                "TARGET":     [0, 1, 0],
            }
        )
        g = BusinessInsightGenerator(minimal_df)
        result = g.run()
        # At minimum, class imbalance and missing value insights should still fire
        assert len(result) >= 1
