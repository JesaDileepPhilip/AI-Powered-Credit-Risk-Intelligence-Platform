"""
tests/test_explanation_service.py — Unit tests for app.explainability.explanation_service.

Strategy:
  - All file I/O (model loading, pipeline loading, background CSV) is mocked.
  - The SHAP explainer and InferenceService are replaced with lightweight mocks
    so tests run in milliseconds without real model artifacts.
  - Tests cover: output schema, driver formatting, business narrative presence,
    batch mode, warmup, and graceful degradation when background CSV is missing.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from app.explainability.explanation_service import ExplanationService, TOP_DRIVERS_N


# ─────────────────────────────────────────────────────────────────────────────
# Mock builders
# ─────────────────────────────────────────────────────────────────────────────

def _make_mock_model(prob: float = 0.45) -> MagicMock:
    """Mock LightGBM model returning constant probability."""
    model = MagicMock()
    model.predict_proba.return_value = np.array([[1 - prob, prob]])
    return model


def _make_mock_pipeline(n_features: int = 5) -> MagicMock:
    """Mock preprocessing pipeline that returns a zero DataFrame."""
    pipeline = MagicMock()
    pipeline.transform.return_value = pd.DataFrame(
        np.zeros((1, n_features)),
        columns=[f"f_{i}" for i in range(n_features)],
    )
    return pipeline


def _make_mock_shap_explainer(
    n_features: int = 5,
    prob: float = 0.45,
    base: float = 0.08,
) -> MagicMock:
    """Mock SHAPExplainer returning synthetic local results."""
    shap_vals = np.linspace(-0.1, 0.2, n_features)
    feature_names = [f"f_{i}" for i in range(n_features)]

    pos = [
        {"feature": f"f_{i}", "value": 0.0, "shap_value": float(shap_vals[i])}
        for i in range(n_features) if shap_vals[i] > 0
    ]
    neg = [
        {"feature": f"f_{i}", "value": 0.0, "shap_value": float(shap_vals[i])}
        for i in range(n_features) if shap_vals[i] < 0
    ]

    mock_exp = MagicMock()
    mock_exp.explain_local.return_value = {
        "shap_values": shap_vals,
        "feature_names": feature_names,
        "feature_values": [0.0] * n_features,
        "expected_value": base,
        "predicted_probability": prob,
        "positive_drivers": sorted(pos, key=lambda x: x["shap_value"], reverse=True),
        "negative_drivers": sorted(neg, key=lambda x: x["shap_value"]),
    }
    return mock_exp


def _make_service_with_mocks(
    prob: float = 0.45,
    n_features: int = 5,
) -> ExplanationService:
    """
    Build an ExplanationService with all heavy dependencies mocked,
    bypassing file I/O entirely.
    """
    service = ExplanationService.__new__(ExplanationService)

    feature_names = [f"f_{i}" for i in range(n_features)]

    # Inject mocked inference service
    mock_inference = MagicMock()
    mock_pipeline = _make_mock_pipeline(n_features)
    mock_model = _make_mock_model(prob)
    mock_inference._load_pipeline.return_value = mock_pipeline
    mock_inference._load_model.return_value = mock_model
    mock_inference._load_training_features.return_value = feature_names
    mock_inference.warmup.return_value = None
    service._inference_service = mock_inference

    # Inject feature names
    service._feature_names = feature_names

    # Inject mocked SHAP explainer
    service._shap_explainer = _make_mock_shap_explainer(
        n_features=n_features, prob=prob
    )

    # Real business explainer (it's lightweight)
    from app.explainability.business_explainer import BusinessExplainer
    service._business_explainer = BusinessExplainer()

    # Other state
    service.top_drivers_n = TOP_DRIVERS_N
    service._model_dir = Path(".")
    service._background_csv = Path("nonexistent.csv")
    service._background_samples = 10
    service._lock = __import__("threading").RLock()

    return service


# ─────────────────────────────────────────────────────────────────────────────
# Output schema tests
# ─────────────────────────────────────────────────────────────────────────────

class TestExplanationServiceOutputSchema:
    def test_returns_dict(self) -> None:
        service = _make_service_with_mocks(prob=0.55)
        result = service.explain({"f_0": 1.0})
        assert isinstance(result, dict)

    def test_required_keys_present(self) -> None:
        service = _make_service_with_mocks(prob=0.45)
        result = service.explain({"f_0": 1.0})
        for key in (
            "default_probability",
            "risk_score",
            "risk_band",
            "positive_risk_drivers",
            "negative_risk_drivers",
            "expected_value",
            "business_narrative",
        ):
            assert key in result, f"Missing key: {key}"

    def test_default_probability_is_float(self) -> None:
        service = _make_service_with_mocks(prob=0.6)
        result = service.explain({})
        assert isinstance(result["default_probability"], float)

    def test_risk_score_is_int(self) -> None:
        service = _make_service_with_mocks(prob=0.6)
        result = service.explain({})
        assert isinstance(result["risk_score"], int)

    def test_risk_band_is_string(self) -> None:
        service = _make_service_with_mocks(prob=0.6)
        result = service.explain({})
        assert isinstance(result["risk_band"], str)

    def test_probability_in_unit_interval(self) -> None:
        service = _make_service_with_mocks(prob=0.3)
        result = service.explain({})
        assert 0.0 <= result["default_probability"] <= 1.0

    def test_risk_score_matches_probability(self) -> None:
        """risk_score should equal int(probability * 1000)."""
        prob = 0.37
        service = _make_service_with_mocks(prob=prob)
        result = service.explain({})
        expected_score = int(result["default_probability"] * 1000)
        assert result["risk_score"] == expected_score

    def test_positive_drivers_is_list(self) -> None:
        service = _make_service_with_mocks()
        result = service.explain({})
        assert isinstance(result["positive_risk_drivers"], list)

    def test_negative_drivers_is_list(self) -> None:
        service = _make_service_with_mocks()
        result = service.explain({})
        assert isinstance(result["negative_risk_drivers"], list)

    def test_driver_dict_has_required_keys(self) -> None:
        service = _make_service_with_mocks()
        result = service.explain({})
        for d in result["positive_risk_drivers"] + result["negative_risk_drivers"]:
            assert "feature" in d
            assert "impact" in d
            assert "feature_value" in d

    def test_business_narrative_is_string(self) -> None:
        service = _make_service_with_mocks(prob=0.8)
        result = service.explain({})
        assert isinstance(result["business_narrative"], str)
        assert len(result["business_narrative"]) > 20

    def test_shap_values_included_when_requested(self) -> None:
        service = _make_service_with_mocks()
        result = service.explain({}, include_shap_array=True)
        assert "shap_values" in result
        assert isinstance(result["shap_values"], list)

    def test_shap_values_excluded_when_not_requested(self) -> None:
        service = _make_service_with_mocks()
        result = service.explain({}, include_shap_array=False)
        assert "shap_values" not in result


# ─────────────────────────────────────────────────────────────────────────────
# Risk band correctness
# ─────────────────────────────────────────────────────────────────────────────

class TestExplanationServiceRiskBand:
    @pytest.mark.parametrize("prob,expected_band", [
        (0.10, "Low Risk"),
        (0.35, "Medium Risk"),
        (0.75, "High Risk"),
    ])
    def test_risk_band_from_probability(
        self, prob: float, expected_band: str
    ) -> None:
        service = _make_service_with_mocks(prob=prob)
        result = service.explain({})
        assert result["risk_band"] == expected_band


# ─────────────────────────────────────────────────────────────────────────────
# Batch explanation
# ─────────────────────────────────────────────────────────────────────────────

class TestExplanationServiceBatch:
    def test_batch_returns_list(self) -> None:
        service = _make_service_with_mocks()
        records = [{"f_0": float(i)} for i in range(3)]
        results = service.explain_batch(records)
        assert isinstance(results, list)

    def test_batch_length_matches_input(self) -> None:
        service = _make_service_with_mocks()
        n = 4
        records = [{"f_0": float(i)} for i in range(n)]
        results = service.explain_batch(records)
        assert len(results) == n

    def test_batch_each_result_has_required_keys(self) -> None:
        service = _make_service_with_mocks()
        records = [{"f_0": 1.0}, {"f_0": 2.0}]
        results = service.explain_batch(records, include_shap_array=False)
        for r in results:
            for key in ("default_probability", "risk_score", "risk_band",
                        "positive_risk_drivers", "negative_risk_drivers"):
                assert key in r

    def test_empty_batch_returns_empty_list(self) -> None:
        service = _make_service_with_mocks()
        results = service.explain_batch([])
        assert results == []


# ─────────────────────────────────────────────────────────────────────────────
# Column dropping
# ─────────────────────────────────────────────────────────────────────────────

class TestExplanationServiceColumnDropping:
    def test_target_column_stripped_gracefully(self) -> None:
        """Passing TARGET in the record should not crash the service."""
        service = _make_service_with_mocks()
        result = service.explain({"f_0": 1.0, "TARGET": 1, "SK_ID_CURR": 999})
        assert "default_probability" in result

    def test_sk_id_curr_stripped_gracefully(self) -> None:
        service = _make_service_with_mocks()
        result = service.explain({"f_0": 0.5, "SK_ID_CURR": 12345})
        assert "risk_band" in result


# ─────────────────────────────────────────────────────────────────────────────
# Background CSV fallback
# ─────────────────────────────────────────────────────────────────────────────

class TestExplanationServiceBackground:
    def test_missing_background_csv_uses_fallback(self) -> None:
        """
        When background CSV is absent, _load_background should return
        a zero-filled fallback without raising.
        """
        service = ExplanationService.__new__(ExplanationService)
        service._background_csv = Path("definitely_does_not_exist.csv")
        service._background_samples = 10

        feature_names = ["f_0", "f_1", "f_2"]
        bg = service._load_background(feature_names)

        assert isinstance(bg, np.ndarray)
        assert bg.shape == (50, 3)
        assert (bg == 0).all()
