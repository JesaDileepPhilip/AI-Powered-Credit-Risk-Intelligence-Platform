"""
tests/test_shap_explainer.py — Unit tests for app.explainability.shap_explainer.

Tests:
  - SHAPExplainer.fit() initialises the explainer correctly
  - explain_global() returns required keys and correct shapes
  - explain_local() returns required keys and valid probability
  - Feature name handling (single-row, multi-row)
  - Error on explain before fit
  - Background sub-sampling
  - expected_value is within [0, 1] for probability output
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import List
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from app.explainability.shap_explainer import SHAPExplainer, TOP_N_FEATURES


# ─────────────────────────────────────────────────────────────────────────────
# Helpers / fixtures
# ─────────────────────────────────────────────────────────────────────────────

def _make_lgbm_model(n_features: int = 8, n_samples: int = 200) -> any:
    """Train a tiny real LightGBM model for integration tests."""
    from lightgbm import LGBMClassifier

    rng = np.random.default_rng(42)
    X = rng.standard_normal((n_samples, n_features)).astype(np.float32)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)

    feature_names = [f"f_{i}" for i in range(n_features)]
    model = LGBMClassifier(
        n_estimators=50,
        num_leaves=15,
        random_state=42,
        verbose=-1,
        n_jobs=1,
    )
    model.fit(X, y, feature_name=feature_names)
    return model, feature_names, X, y


@pytest.fixture(scope="module")
def lgbm_setup():
    """Shared LightGBM model + data fixture (created once per module)."""
    model, feature_names, X, y = _make_lgbm_model(n_features=8, n_samples=300)
    return model, feature_names, X, y


@pytest.fixture(scope="module")
def fitted_explainer(lgbm_setup):
    """Fitted SHAPExplainer fixture shared across tests."""
    model, feature_names, X, _ = lgbm_setup
    explainer = SHAPExplainer(model=model, feature_names=feature_names)
    explainer.fit(X)
    return explainer, feature_names, X


# ─────────────────────────────────────────────────────────────────────────────
# Fit tests
# ─────────────────────────────────────────────────────────────────────────────

class TestSHAPExplainerFit:
    def test_fit_returns_self(self, lgbm_setup) -> None:
        model, feature_names, X, _ = lgbm_setup
        exp = SHAPExplainer(model=model, feature_names=feature_names)
        result = exp.fit(X)
        assert result is exp

    def test_is_fitted_after_fit(self, lgbm_setup) -> None:
        model, feature_names, X, _ = lgbm_setup
        exp = SHAPExplainer(model=model, feature_names=feature_names)
        exp.fit(X)
        assert exp._is_fitted is True

    def test_expected_value_is_float(self, fitted_explainer) -> None:
        exp, _, _ = fitted_explainer
        assert isinstance(exp.expected_value, float)

    def test_expected_value_in_unit_interval(self, fitted_explainer) -> None:
        """For probability output, expected_value should be in [0, 1]."""
        exp, _, _ = fitted_explainer
        assert 0.0 <= exp.expected_value <= 1.0

    def test_explainer_object_is_set(self, fitted_explainer) -> None:
        exp, _, _ = fitted_explainer
        assert exp._explainer is not None

    def test_raises_before_fit(self, lgbm_setup) -> None:
        model, feature_names, X, _ = lgbm_setup
        exp = SHAPExplainer(model=model, feature_names=feature_names)
        with pytest.raises(RuntimeError, match="not fitted"):
            exp.explain_global(X)

    def test_background_subsampling(self, lgbm_setup) -> None:
        """When n > background_samples, a sub-sample is used."""
        model, feature_names, X, _ = lgbm_setup
        exp = SHAPExplainer(model=model, feature_names=feature_names, background_samples=20)
        exp.fit(X)  # X has 300 rows, limit is 20
        assert exp._is_fitted is True

    def test_fit_with_small_background(self, lgbm_setup) -> None:
        """All rows used when n <= background_samples."""
        model, feature_names, X, _ = lgbm_setup
        X_small = X[:10]
        exp = SHAPExplainer(model=model, feature_names=feature_names, background_samples=100)
        exp.fit(X_small)
        assert exp._is_fitted is True


# ─────────────────────────────────────────────────────────────────────────────
# Global explanation tests
# ─────────────────────────────────────────────────────────────────────────────

class TestExplainGlobal:
    def test_returns_required_keys(self, fitted_explainer) -> None:
        exp, _, X = fitted_explainer
        result = exp.explain_global(X[:50])
        for key in ("shap_values", "mean_abs_shap", "feature_names",
                    "expected_value", "top_features"):
            assert key in result, f"Missing key: {key}"

    def test_shap_values_shape(self, fitted_explainer) -> None:
        exp, feature_names, X = fitted_explainer
        result = exp.explain_global(X[:30])
        assert result["shap_values"].shape == (30, len(feature_names))

    def test_mean_abs_shap_shape(self, fitted_explainer) -> None:
        exp, feature_names, X = fitted_explainer
        result = exp.explain_global(X[:20])
        assert result["mean_abs_shap"].shape == (len(feature_names),)

    def test_mean_abs_shap_non_negative(self, fitted_explainer) -> None:
        exp, _, X = fitted_explainer
        result = exp.explain_global(X[:20])
        assert (result["mean_abs_shap"] >= 0.0).all()

    def test_top_features_length(self, fitted_explainer) -> None:
        exp, feature_names, X = fitted_explainer
        result = exp.explain_global(X[:20])
        # TOP_N_FEATURES or fewer (if we have fewer total features)
        assert len(result["top_features"]) <= min(TOP_N_FEATURES, len(feature_names))

    def test_top_features_have_correct_keys(self, fitted_explainer) -> None:
        exp, _, X = fitted_explainer
        result = exp.explain_global(X[:20])
        for feat in result["top_features"]:
            assert "feature" in feat
            assert "importance" in feat

    def test_top_features_sorted_descending(self, fitted_explainer) -> None:
        exp, _, X = fitted_explainer
        result = exp.explain_global(X[:20])
        importances = [f["importance"] for f in result["top_features"]]
        assert importances == sorted(importances, reverse=True)

    def test_feature_names_preserved(self, fitted_explainer) -> None:
        exp, feature_names, X = fitted_explainer
        result = exp.explain_global(X[:20])
        assert result["feature_names"] == feature_names


# ─────────────────────────────────────────────────────────────────────────────
# Local explanation tests
# ─────────────────────────────────────────────────────────────────────────────

class TestExplainLocal:
    def test_returns_required_keys(self, fitted_explainer) -> None:
        exp, _, X = fitted_explainer
        result = exp.explain_local(X[0])
        for key in ("shap_values", "feature_names", "feature_values",
                    "expected_value", "predicted_probability",
                    "positive_drivers", "negative_drivers"):
            assert key in result, f"Missing key: {key}"

    def test_predicted_probability_in_unit_interval(self, fitted_explainer) -> None:
        exp, _, X = fitted_explainer
        for row in X[:10]:
            result = exp.explain_local(row)
            assert 0.0 <= result["predicted_probability"] <= 1.0

    def test_shap_values_is_1d(self, fitted_explainer) -> None:
        exp, feature_names, X = fitted_explainer
        result = exp.explain_local(X[0])
        assert len(result["shap_values"]) == len(feature_names)

    def test_positive_drivers_have_positive_shap(self, fitted_explainer) -> None:
        exp, _, X = fitted_explainer
        # Run over several rows to increase coverage
        for row in X[:20]:
            result = exp.explain_local(row)
            for d in result["positive_drivers"]:
                assert d["shap_value"] > 0, (
                    f"Positive driver {d['feature']} has non-positive SHAP: {d['shap_value']}"
                )

    def test_negative_drivers_have_negative_shap(self, fitted_explainer) -> None:
        exp, _, X = fitted_explainer
        for row in X[:20]:
            result = exp.explain_local(row)
            for d in result["negative_drivers"]:
                assert d["shap_value"] < 0, (
                    f"Negative driver {d['feature']} has non-negative SHAP: {d['shap_value']}"
                )

    def test_accepts_2d_input(self, fitted_explainer) -> None:
        exp, _, X = fitted_explainer
        result = exp.explain_local(X[0:1])  # (1, n_features)
        assert "predicted_probability" in result

    def test_feature_values_length_matches_features(self, fitted_explainer) -> None:
        exp, feature_names, X = fitted_explainer
        result = exp.explain_local(X[0])
        assert len(result["feature_values"]) == len(feature_names)

    def test_top_n_respected(self, fitted_explainer) -> None:
        exp, _, X = fitted_explainer
        top_n = 3
        result = exp.explain_local(X[0], top_n=top_n)
        assert len(result["positive_drivers"]) <= top_n
        assert len(result["negative_drivers"]) <= top_n

    def test_drivers_sorted_correctly(self, fitted_explainer) -> None:
        """Positive drivers sorted desc; negative sorted asc (most negative first)."""
        exp, _, X = fitted_explainer
        result = exp.explain_local(X[0])
        pos_shaps = [d["shap_value"] for d in result["positive_drivers"]]
        neg_shaps = [d["shap_value"] for d in result["negative_drivers"]]
        assert pos_shaps == sorted(pos_shaps, reverse=True)
        assert neg_shaps == sorted(neg_shaps)

    def test_get_shap_explainer_returns_object(self, fitted_explainer) -> None:
        exp, _, _ = fitted_explainer
        raw_exp = exp.get_shap_explainer()
        assert raw_exp is not None


# ─────────────────────────────────────────────────────────────────────────────
# Edge cases
# ─────────────────────────────────────────────────────────────────────────────

class TestSHAPExplainerEdgeCases:
    def test_single_row_background(self, lgbm_setup) -> None:
        """Explainer should work even with a single background row."""
        model, feature_names, X, _ = lgbm_setup
        exp = SHAPExplainer(model=model, feature_names=feature_names)
        exp.fit(X[:1])
        result = exp.explain_local(X[0])
        assert "predicted_probability" in result

    def test_pandas_dataframe_input_accepted(self, fitted_explainer) -> None:
        """Explainer should accept pandas DataFrames via _to_2d_array."""
        import pandas as pd
        exp, feature_names, X = fitted_explainer
        df = pd.DataFrame(X[:5], columns=feature_names)
        result = exp.explain_global(df)
        assert result["shap_values"].shape[0] == 5
