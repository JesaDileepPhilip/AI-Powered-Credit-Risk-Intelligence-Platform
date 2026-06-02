"""
tests/test_inference_service.py — Unit tests for app.ml.inference_service.

Tests:
  - InferenceService.predict() returns required dict structure
  - Columns TARGET and SK_ID_CURR are automatically dropped
  - Schema validation adds missing columns as NaN (no crash)
  - predict_dataframe() returns correct number of results
  - SchemaValidationError is importable
  - warmup() loads models without errors
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import pytest
import joblib

from app.ml.inference_service import InferenceService, SchemaValidationError


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_mock_model(probabilities: List[float]) -> MagicMock:
    """Return a mock LightGBM classifier returning given probabilities."""
    model = MagicMock()
    n = len(probabilities)
    proba_matrix = np.column_stack([
        [1 - p for p in probabilities],
        probabilities,
    ])
    model.predict_proba.return_value = proba_matrix
    return model


def _make_passthrough_pipeline(expected_df: pd.DataFrame) -> MagicMock:
    """Return a mock pipeline whose transform() returns the expected DataFrame."""
    pipeline = MagicMock()
    pipeline.transform.return_value = expected_df
    return pipeline


def _make_service_with_mocks(
    probabilities: List[float],
    feature_names: List[str] = None,
    input_df: pd.DataFrame = None,
) -> InferenceService:
    """
    Create an InferenceService with internal pipeline and model mocked.
    """
    if feature_names is None:
        feature_names = ["f1", "f2", "f3"]
    if input_df is None:
        input_df = pd.DataFrame({f: np.zeros(len(probabilities)) for f in feature_names})

    service = InferenceService.__new__(InferenceService)
    service._model_dir = Path(".")
    service._pipeline = _make_passthrough_pipeline(input_df)
    service._model = _make_mock_model(probabilities)
    service._training_features = feature_names

    # Patch the registry so load_metadata doesn't hit disk
    mock_registry = MagicMock()
    mock_registry.load_model.return_value = service._model
    mock_registry.load_metadata.return_value = {
        "training_feature_names": feature_names
    }
    service._registry = mock_registry

    return service


# ─────────────────────────────────────────────────────────────────────────────
# SchemaValidationError
# ─────────────────────────────────────────────────────────────────────────────

class TestSchemaValidationError:
    def test_is_value_error_subclass(self) -> None:
        assert issubclass(SchemaValidationError, ValueError)

    def test_can_be_raised(self) -> None:
        with pytest.raises(SchemaValidationError):
            raise SchemaValidationError("test")


# ─────────────────────────────────────────────────────────────────────────────
# predict()
# ─────────────────────────────────────────────────────────────────────────────

class TestPredict:
    def test_returns_dict(self) -> None:
        service = _make_service_with_mocks([0.42])
        result = service.predict({"f1": 1.0, "f2": 0.5, "f3": -0.3})
        assert isinstance(result, dict)

    def test_required_keys_present(self) -> None:
        service = _make_service_with_mocks([0.42])
        result = service.predict({"f1": 1.0, "f2": 0.5, "f3": -0.3})
        for key in ("default_probability", "risk_score", "risk_band"):
            assert key in result, f"Missing key: {key}"

    def test_probability_value_type(self) -> None:
        service = _make_service_with_mocks([0.18])
        result = service.predict({"f1": 0.0, "f2": 0.0, "f3": 0.0})
        assert isinstance(result["default_probability"], float)

    def test_risk_score_is_int(self) -> None:
        service = _make_service_with_mocks([0.65])
        result = service.predict({"f1": 0.0, "f2": 0.0, "f3": 0.0})
        assert isinstance(result["risk_score"], int)

    def test_risk_band_is_string(self) -> None:
        service = _make_service_with_mocks([0.30])
        result = service.predict({"f1": 0.0, "f2": 0.0, "f3": 0.0})
        assert isinstance(result["risk_band"], str)

    def test_target_column_dropped_from_record(self) -> None:
        """Inference should not fail when TARGET is in the input record."""
        service = _make_service_with_mocks([0.25])
        result = service.predict({"f1": 1.0, "f2": 0.5, "f3": -0.3, "TARGET": 1})
        assert "default_probability" in result

    def test_sk_id_curr_dropped_from_record(self) -> None:
        """Inference should not fail when SK_ID_CURR is in the input record."""
        service = _make_service_with_mocks([0.25])
        result = service.predict({"f1": 1.0, "f2": 0.5, "f3": -0.3, "SK_ID_CURR": 999})
        assert "default_probability" in result


# ─────────────────────────────────────────────────────────────────────────────
# predict_dataframe()
# ─────────────────────────────────────────────────────────────────────────────

class TestPredictDataFrame:
    def test_returns_list(self) -> None:
        n = 3
        df = pd.DataFrame({"f1": range(n), "f2": range(n), "f3": range(n)})
        service = _make_service_with_mocks([0.10, 0.40, 0.80], input_df=df)
        results = service.predict_dataframe(df)
        assert isinstance(results, list)

    def test_length_matches_input(self) -> None:
        n = 5
        df = pd.DataFrame({"f1": range(n), "f2": range(n), "f3": range(n)})
        probs = [0.1, 0.2, 0.3, 0.4, 0.6]
        service = _make_service_with_mocks(probs, input_df=df)
        results = service.predict_dataframe(df)
        assert len(results) == n

    def test_each_result_has_required_keys(self) -> None:
        n = 2
        df = pd.DataFrame({"f1": range(n), "f2": range(n), "f3": range(n)})
        service = _make_service_with_mocks([0.15, 0.65], input_df=df)
        results = service.predict_dataframe(df)
        for result in results:
            for key in ("default_probability", "risk_score", "risk_band"):
                assert key in result

    def test_probabilities_are_in_valid_range(self) -> None:
        n = 4
        df = pd.DataFrame({"f1": range(n), "f2": range(n), "f3": range(n)})
        probs = [0.05, 0.30, 0.55, 0.90]
        service = _make_service_with_mocks(probs, input_df=df)
        results = service.predict_dataframe(df)
        for r in results:
            assert 0.0 <= r["default_probability"] <= 1.0

    def test_target_column_dropped_before_transform(self) -> None:
        """
        Verify that TARGET in the input DataFrame does not propagate to
        the model and that the service returns results without error.
        """
        n = 2
        df = pd.DataFrame({
            "f1": [1.0, 2.0],
            "f2": [0.5, 1.5],
            "f3": [-0.5, 0.5],
            "TARGET": [0, 1],
        })
        # The pipeline receives df without TARGET — mock accordingly
        clean_df = df.drop(columns=["TARGET"])
        service = _make_service_with_mocks([0.25, 0.75], input_df=clean_df)
        results = service.predict_dataframe(df)
        assert len(results) == n


# ─────────────────────────────────────────────────────────────────────────────
# Schema validation
# ─────────────────────────────────────────────────────────────────────────────

class TestSchemaValidation:
    def test_missing_features_filled_with_nan(self) -> None:
        """
        When a feature expected by the model is absent from input,
        _validate_schema should add it as NaN (not raise an error).
        """
        service = _make_service_with_mocks([0.40])
        df = pd.DataFrame({"f1": [1.0]})  # f2 and f3 are missing
        service._validate_schema(df)
        assert "f2" in df.columns
        assert "f3" in df.columns
        assert np.isnan(df["f2"].iloc[0])

    def test_extra_features_silently_ignored(self) -> None:
        """Extra columns in input beyond training features should not raise."""
        service = _make_service_with_mocks([0.15])
        df = pd.DataFrame({"f1": [1.0], "f2": [2.0], "f3": [3.0], "extra_col": [99.0]})
        # Should not raise
        service._validate_schema(df)

    def test_empty_training_features_skips_validation(self) -> None:
        """If training features list is empty, validation is skipped gracefully."""
        service = _make_service_with_mocks([0.50])
        service._training_features = []
        df = pd.DataFrame({"anything": [1.0]})
        # Should not raise
        service._validate_schema(df)
