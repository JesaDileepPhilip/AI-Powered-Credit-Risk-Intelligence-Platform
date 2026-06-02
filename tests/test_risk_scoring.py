"""
tests/test_risk_scoring.py — Unit tests for app.ml.risk_scoring.

Tests:
  - compute_risk_band() for all three bands and boundary conditions
  - compute_risk_score() integer mapping
  - score_single() full RiskResult output
  - score_batch() batch processing
  - Error handling for out-of-range probabilities
"""

from __future__ import annotations

import pytest

from app.ml.risk_scoring import (
    BAND_HIGH,
    BAND_LOW,
    BAND_MEDIUM,
    RiskResult,
    compute_risk_band,
    compute_risk_score,
    score_batch,
    score_single,
)


# ─────────────────────────────────────────────────────────────────────────────
# compute_risk_band
# ─────────────────────────────────────────────────────────────────────────────

class TestComputeRiskBand:
    """Tests for the risk band classification logic."""

    @pytest.mark.parametrize("prob,expected_band", [
        (0.00, BAND_LOW),
        (0.12, BAND_LOW),
        (0.19, BAND_LOW),
        (0.199, BAND_LOW),
    ])
    def test_low_risk(self, prob: float, expected_band: str) -> None:
        assert compute_risk_band(prob) == expected_band

    @pytest.mark.parametrize("prob,expected_band", [
        (0.20, BAND_MEDIUM),
        (0.38, BAND_MEDIUM),
        (0.499, BAND_MEDIUM),
        (0.49, BAND_MEDIUM),
    ])
    def test_medium_risk(self, prob: float, expected_band: str) -> None:
        assert compute_risk_band(prob) == expected_band

    @pytest.mark.parametrize("prob,expected_band", [
        (0.50, BAND_HIGH),
        (0.75, BAND_HIGH),
        (0.82, BAND_HIGH),
        (1.00, BAND_HIGH),
    ])
    def test_high_risk(self, prob: float, expected_band: str) -> None:
        assert compute_risk_band(prob) == expected_band

    @pytest.mark.parametrize("invalid_prob", [-0.001, 1.001, -1.0, 2.5])
    def test_raises_for_out_of_range(self, invalid_prob: float) -> None:
        with pytest.raises(ValueError, match="probability must be in"):
            compute_risk_band(invalid_prob)


# ─────────────────────────────────────────────────────────────────────────────
# compute_risk_score
# ─────────────────────────────────────────────────────────────────────────────

class TestComputeRiskScore:
    """Tests for the integer risk score computation."""

    @pytest.mark.parametrize("prob,expected_score", [
        (0.00, 0),
        (0.12, 120),
        (0.38, 380),
        (0.50, 500),
        (0.82, 820),
        (1.00, 1000),
    ])
    def test_score_values(self, prob: float, expected_score: int) -> None:
        assert compute_risk_score(prob) == expected_score

    def test_score_is_int(self) -> None:
        score = compute_risk_score(0.38)
        assert isinstance(score, int)

    @pytest.mark.parametrize("invalid_prob", [-0.01, 1.01])
    def test_raises_for_out_of_range(self, invalid_prob: float) -> None:
        with pytest.raises(ValueError):
            compute_risk_score(invalid_prob)


# ─────────────────────────────────────────────────────────────────────────────
# score_single
# ─────────────────────────────────────────────────────────────────────────────

class TestScoreSingle:
    """Tests for the full single-record scoring function."""

    def test_example_low_risk(self) -> None:
        result = score_single(0.12)
        assert result.risk_score == 120
        assert result.risk_band == BAND_LOW
        assert result.default_probability == 0.12

    def test_example_medium_risk(self) -> None:
        result = score_single(0.38)
        assert result.risk_score == 380
        assert result.risk_band == BAND_MEDIUM

    def test_example_high_risk(self) -> None:
        result = score_single(0.82)
        assert result.risk_score == 820
        assert result.risk_band == BAND_HIGH

    def test_returns_risk_result(self) -> None:
        result = score_single(0.5)
        assert isinstance(result, RiskResult)

    def test_to_dict_keys(self) -> None:
        result = score_single(0.25)
        d = result.to_dict()
        assert "default_probability" in d
        assert "risk_score" in d
        assert "risk_band" in d

    def test_to_dict_values_are_correct_types(self) -> None:
        d = score_single(0.42).to_dict()
        assert isinstance(d["default_probability"], float)
        assert isinstance(d["risk_score"], int)
        assert isinstance(d["risk_band"], str)

    def test_boundary_zero(self) -> None:
        result = score_single(0.0)
        assert result.risk_score == 0
        assert result.risk_band == BAND_LOW

    def test_boundary_one(self) -> None:
        result = score_single(1.0)
        assert result.risk_score == 1000
        assert result.risk_band == BAND_HIGH

    @pytest.mark.parametrize("invalid_prob", [-0.01, 1.01, -1.0, 2.0])
    def test_raises_for_invalid_probability(self, invalid_prob: float) -> None:
        with pytest.raises(ValueError, match="probability must be in"):
            score_single(invalid_prob)


# ─────────────────────────────────────────────────────────────────────────────
# score_batch
# ─────────────────────────────────────────────────────────────────────────────

class TestScoreBatch:
    """Tests for batch scoring."""

    def test_returns_list_of_risk_results(self) -> None:
        probs = [0.10, 0.35, 0.75]
        results = score_batch(probs)
        assert len(results) == 3
        assert all(isinstance(r, RiskResult) for r in results)

    def test_batch_order_preserved(self) -> None:
        probs = [0.05, 0.45, 0.95]
        results = score_batch(probs)
        assert results[0].risk_band == BAND_LOW
        assert results[1].risk_band == BAND_MEDIUM
        assert results[2].risk_band == BAND_HIGH

    def test_batch_single_element(self) -> None:
        results = score_batch([0.60])
        assert len(results) == 1
        assert results[0].risk_band == BAND_HIGH

    def test_batch_empty_list(self) -> None:
        results = score_batch([])
        assert results == []

    def test_batch_all_low_risk(self) -> None:
        results = score_batch([0.01, 0.10, 0.19])
        assert all(r.risk_band == BAND_LOW for r in results)
