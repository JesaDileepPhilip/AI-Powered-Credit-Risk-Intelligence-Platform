"""
tests/test_predict.py — Unit tests for app.ml.predict.

Tests:
  - predict_proba() returns correct shape and value range
  - predict_single_dict() returns required dict keys
  - predict_batch() returns a DataFrame with correct columns
  - TARGET and SK_ID_CURR are removed automatically
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from app.ml.predict import predict_proba, predict_single_dict, predict_batch


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

def _make_mock_model(n_rows: int = 5) -> MagicMock:
    """Return a mock sklearn-compatible classifier."""
    model = MagicMock()
    proba = np.column_stack([
        np.linspace(0.8, 0.2, n_rows),
        np.linspace(0.2, 0.8, n_rows),
    ])
    model.predict_proba.return_value = proba
    return model


def _make_mock_pipeline() -> MagicMock:
    """Return a mock preprocessing pipeline that passes data through unchanged."""
    pipeline = MagicMock()

    def passthrough(df):
        return df

    pipeline.transform.side_effect = passthrough
    return pipeline


# ─────────────────────────────────────────────────────────────────────────────
# predict_proba
# ─────────────────────────────────────────────────────────────────────────────

class TestPredictProba:
    def test_returns_1d_array(self) -> None:
        model = _make_mock_model(n_rows=4)
        X = np.zeros((4, 3))
        result = predict_proba(model, X)
        assert result.ndim == 1

    def test_length_matches_input(self) -> None:
        n = 7
        model = _make_mock_model(n_rows=n)
        X = np.zeros((n, 5))
        result = predict_proba(model, X)
        assert len(result) == n

    def test_values_in_0_1_range(self) -> None:
        model = _make_mock_model(n_rows=10)
        X = np.zeros((10, 5))
        result = predict_proba(model, X)
        assert np.all(result >= 0.0)
        assert np.all(result <= 1.0)

    def test_returns_float_dtype(self) -> None:
        model = _make_mock_model(n_rows=3)
        X = np.zeros((3, 2))
        result = predict_proba(model, X)
        assert result.dtype == float or np.issubdtype(result.dtype, np.floating)


# ─────────────────────────────────────────────────────────────────────────────
# predict_single_dict
# ─────────────────────────────────────────────────────────────────────────────

class TestPredictSingleDict:
    """Tests for predict_single_dict using a mocked InferenceService."""

    def _mock_inference_service(self, probability: float = 0.35) -> MagicMock:
        service = MagicMock()
        service.predict.return_value = {
            "default_probability": probability,
            "risk_score": int(probability * 1000),
            "risk_band": "Medium Risk",
        }
        return service

    def test_returns_dict(self) -> None:
        with patch("app.ml.predict.InferenceService") as mock_cls:
            mock_cls.return_value = self._mock_inference_service(0.35)
            result = predict_single_dict({"feature_1": 1.0})
        assert isinstance(result, dict)

    def test_required_keys_present(self) -> None:
        with patch("app.ml.predict.InferenceService") as mock_cls:
            mock_cls.return_value = self._mock_inference_service(0.35)
            result = predict_single_dict({"feature_1": 1.0})
        for key in ("default_probability", "risk_score", "risk_band"):
            assert key in result

    def test_probability_value(self) -> None:
        with patch("app.ml.predict.InferenceService") as mock_cls:
            mock_cls.return_value = self._mock_inference_service(0.35)
            result = predict_single_dict({"feature_1": 1.0})
        assert result["default_probability"] == pytest.approx(0.35)

    def test_risk_score_type(self) -> None:
        with patch("app.ml.predict.InferenceService") as mock_cls:
            mock_cls.return_value = self._mock_inference_service(0.72)
            result = predict_single_dict({"x": 0.0})
        assert isinstance(result["risk_score"], int)

    def test_risk_band_is_string(self) -> None:
        with patch("app.ml.predict.InferenceService") as mock_cls:
            mock_cls.return_value = self._mock_inference_service(0.15)
            result = predict_single_dict({"x": 0.0})
        assert isinstance(result["risk_band"], str)


# ─────────────────────────────────────────────────────────────────────────────
# predict_batch
# ─────────────────────────────────────────────────────────────────────────────

class TestPredictBatch:
    """Tests for the batch prediction function using mocked InferenceService."""

    def _mock_inference_cls(self, n_rows: int = 3) -> MagicMock:
        """Return a mock InferenceService class whose instance handles batch calls."""

        def make_result(i):
            p = float(0.1 + 0.3 * (i % 3))
            return {
                "default_probability": round(p, 4),
                "risk_score": int(p * 1000),
                "risk_band": "Low Risk" if p < 0.2 else ("Medium Risk" if p < 0.5 else "High Risk"),
            }

        mock_instance = MagicMock()
        mock_instance.predict.side_effect = [make_result(i) for i in range(n_rows)]
        mock_cls = MagicMock(return_value=mock_instance)
        return mock_cls

    def test_returns_dataframe(self) -> None:
        df = pd.DataFrame({"feature_a": [1.0, 2.0, 3.0]})
        with patch("app.ml.predict.InferenceService", self._mock_inference_cls(3)):
            result = predict_batch(df)
        assert isinstance(result, pd.DataFrame)

    def test_row_count_matches_input(self) -> None:
        n = 5
        df = pd.DataFrame({"feature_a": range(n)})
        with patch("app.ml.predict.InferenceService", self._mock_inference_cls(n)):
            result = predict_batch(df)
        assert len(result) == n

    def test_output_columns(self) -> None:
        df = pd.DataFrame({"feature_a": [1.0, 2.0]})
        with patch("app.ml.predict.InferenceService", self._mock_inference_cls(2)):
            result = predict_batch(df)
        for col in ("default_probability", "risk_score", "risk_band"):
            assert col in result.columns

    def test_target_column_dropped_from_input(self) -> None:
        """Ensure predict_batch doesn't crash when TARGET is in the DataFrame."""
        df = pd.DataFrame({"feature_a": [1.0], "TARGET": [1], "SK_ID_CURR": [100]})
        with patch("app.ml.predict.InferenceService", self._mock_inference_cls(1)):
            result = predict_batch(df)
        assert len(result) == 1

    def test_probability_values_in_range(self) -> None:
        n = 4
        df = pd.DataFrame({"x": range(n)})
        with patch("app.ml.predict.InferenceService", self._mock_inference_cls(n)):
            result = predict_batch(df)
        assert (result["default_probability"] >= 0.0).all()
        assert (result["default_probability"] <= 1.0).all()
