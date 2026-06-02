"""
tests/test_business_explainer.py — Unit tests for app.explainability.business_explainer.

Tests:
  - BusinessExplainer.generate() returns a non-empty string for all risk bands
  - Narrative contains risk band keyword
  - Feature label mapping (exact match, substring match, fallback)
  - Narrative structure: intro + drivers + audit note always present
  - generate_from_explanation() convenience wrapper
  - Custom label override
  - Empty driver lists handled gracefully
"""

from __future__ import annotations

from typing import Any, Dict, List

import pytest

from app.explainability.business_explainer import BusinessExplainer, FEATURE_LABEL_MAP


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture()
def explainer() -> BusinessExplainer:
    return BusinessExplainer()


def _make_explanation(
    risk_band: str = "High Risk",
    prob: float = 0.75,
    score: int = 750,
    pos_drivers: int = 3,
    neg_drivers: int = 2,
) -> Dict[str, Any]:
    positive = [
        {"feature": f"AMT_CREDIT", "shap_value": 0.15, "impact": 0.15, "feature_value": 800000},
        {"feature": f"debt_to_income_ratio", "shap_value": 0.10, "impact": 0.10, "feature_value": 0.8},
        {"feature": f"EXT_SOURCE_2", "shap_value": 0.05, "impact": 0.05, "feature_value": 0.2},
    ][:pos_drivers]
    negative = [
        {"feature": "DAYS_EMPLOYED", "shap_value": -0.08, "impact": -0.08, "feature_value": -2000},
        {"feature": "FLAG_OWN_REALTY", "shap_value": -0.04, "impact": -0.04, "feature_value": 1},
    ][:neg_drivers]
    return {
        "risk_band": risk_band,
        "default_probability": prob,
        "risk_score": score,
        "positive_risk_drivers": positive,
        "negative_risk_drivers": negative,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Core generation
# ─────────────────────────────────────────────────────────────────────────────

class TestBusinessExplainerGenerate:
    @pytest.mark.parametrize("risk_band,prob,score", [
        ("High Risk", 0.82, 820),
        ("Medium Risk", 0.38, 380),
        ("Low Risk", 0.12, 120),
    ])
    def test_returns_non_empty_string(
        self, explainer: BusinessExplainer, risk_band: str, prob: float, score: int
    ) -> None:
        exp = _make_explanation(risk_band=risk_band, prob=prob, score=score)
        narrative = explainer.generate(
            risk_band=risk_band,
            default_probability=prob,
            risk_score=score,
            positive_drivers=exp["positive_risk_drivers"],
            negative_drivers=exp["negative_risk_drivers"],
        )
        assert isinstance(narrative, str)
        assert len(narrative) > 50

    @pytest.mark.parametrize("risk_band,prob,score,expected_keyword", [
        ("High Risk", 0.82, 820, "HIGH"),
        ("Medium Risk", 0.38, 380, "MODERATE"),   # template says "MODERATE default risk"
        ("Low Risk", 0.12, 120, "LOW"),
    ])
    def test_narrative_contains_risk_level(
        self, explainer: BusinessExplainer, risk_band: str, prob: float, score: int,
        expected_keyword: str,
    ) -> None:
        exp = _make_explanation(risk_band=risk_band, prob=prob, score=score)
        narrative = explainer.generate(
            risk_band=risk_band,
            default_probability=prob,
            risk_score=score,
            positive_drivers=exp["positive_risk_drivers"],
            negative_drivers=exp["negative_risk_drivers"],
        )
        assert expected_keyword in narrative.upper()

    def test_narrative_contains_probability(self, explainer: BusinessExplainer) -> None:
        exp = _make_explanation(prob=0.75)
        narrative = explainer.generate(
            risk_band="High Risk",
            default_probability=0.75,
            risk_score=750,
            positive_drivers=exp["positive_risk_drivers"],
            negative_drivers=exp["negative_risk_drivers"],
        )
        # 75% should appear somewhere in the narrative
        assert "75" in narrative

    def test_narrative_contains_risk_score(self, explainer: BusinessExplainer) -> None:
        exp = _make_explanation(score=820)
        narrative = explainer.generate(
            risk_band="High Risk",
            default_probability=0.82,
            risk_score=820,
            positive_drivers=exp["positive_risk_drivers"],
            negative_drivers=exp["negative_risk_drivers"],
        )
        assert "820" in narrative

    def test_narrative_mentions_positive_driver(self, explainer: BusinessExplainer) -> None:
        """The narrative should reference at least one positive driver concept."""
        exp = _make_explanation()
        narrative = explainer.generate(
            risk_band="High Risk",
            default_probability=0.8,
            risk_score=800,
            positive_drivers=exp["positive_risk_drivers"],
            negative_drivers=exp["negative_risk_drivers"],
        )
        # At least one driver label should appear
        assert "credit" in narrative.lower() or "income" in narrative.lower()

    def test_empty_positive_drivers_no_crash(self, explainer: BusinessExplainer) -> None:
        narrative = explainer.generate(
            risk_band="High Risk",
            default_probability=0.6,
            risk_score=600,
            positive_drivers=[],
            negative_drivers=[],
        )
        assert isinstance(narrative, str)
        assert len(narrative) > 10

    def test_empty_negative_drivers_high_risk_adds_note(
        self, explainer: BusinessExplainer
    ) -> None:
        """High Risk with no mitigants should note absence of mitigating factors."""
        exp = _make_explanation(prob=0.9, score=900)
        narrative = explainer.generate(
            risk_band="High Risk",
            default_probability=0.9,
            risk_score=900,
            positive_drivers=exp["positive_risk_drivers"],
            negative_drivers=[],
        )
        assert "mitigat" in narrative.lower()

    def test_audit_note_always_present(self, explainer: BusinessExplainer) -> None:
        for risk_band in ["High Risk", "Medium Risk", "Low Risk"]:
            exp = _make_explanation(risk_band=risk_band)
            narrative = explainer.generate(
                risk_band=risk_band,
                default_probability=exp["default_probability"],
                risk_score=exp["risk_score"],
                positive_drivers=exp["positive_risk_drivers"],
                negative_drivers=exp["negative_risk_drivers"],
            )
            assert "Risk score:" in narrative


# ─────────────────────────────────────────────────────────────────────────────
# Feature label mapping
# ─────────────────────────────────────────────────────────────────────────────

class TestFeatureLabelMapping:
    def test_exact_match_amt_credit(self, explainer: BusinessExplainer) -> None:
        label = explainer._feature_to_label("amt_credit")
        assert label == FEATURE_LABEL_MAP["amt_credit"]

    def test_case_insensitive_match(self, explainer: BusinessExplainer) -> None:
        label = explainer._feature_to_label("AMT_CREDIT")
        assert "credit" in label.lower()

    def test_substring_match_ext_source(self, explainer: BusinessExplainer) -> None:
        label = explainer._feature_to_label("EXT_SOURCE_2_processed")
        assert "credit score" in label.lower() or "external" in label.lower()

    def test_unknown_feature_fallback(self, explainer: BusinessExplainer) -> None:
        label = explainer._feature_to_label("XYZ_UNKNOWN_FEATURE_42")
        # Fallback should be human-readable (no raw underscore noise)
        assert isinstance(label, str)
        assert len(label) > 0

    def test_days_employed_mapped(self, explainer: BusinessExplainer) -> None:
        label = explainer._feature_to_label("DAYS_EMPLOYED")
        assert "employment" in label.lower()

    def test_custom_label_override(self) -> None:
        custom_explainer = BusinessExplainer(
            feature_label_map={"custom_feature": "my custom label"}
        )
        label = custom_explainer._feature_to_label("custom_feature")
        assert label == "my custom label"


# ─────────────────────────────────────────────────────────────────────────────
# generate_from_explanation()
# ─────────────────────────────────────────────────────────────────────────────

class TestGenerateFromExplanation:
    def test_returns_same_as_generate(self, explainer: BusinessExplainer) -> None:
        exp_dict = _make_explanation(risk_band="Medium Risk", prob=0.35, score=350)
        narrative_from = explainer.generate_from_explanation(exp_dict)
        narrative_direct = explainer.generate(
            risk_band="Medium Risk",
            default_probability=0.35,
            risk_score=350,
            positive_drivers=exp_dict["positive_risk_drivers"],
            negative_drivers=exp_dict["negative_risk_drivers"],
        )
        assert narrative_from == narrative_direct

    def test_accepts_all_risk_bands(self, explainer: BusinessExplainer) -> None:
        for band in ["High Risk", "Medium Risk", "Low Risk"]:
            exp_dict = _make_explanation(risk_band=band)
            result = explainer.generate_from_explanation(exp_dict)
            assert isinstance(result, str)
            assert len(result) > 0

    def test_with_impact_key_instead_of_shap_value(
        self, explainer: BusinessExplainer
    ) -> None:
        """ExplanationService uses 'impact' key; ensure backward compat."""
        exp_dict = {
            "risk_band": "High Risk",
            "default_probability": 0.7,
            "risk_score": 700,
            "positive_risk_drivers": [
                {"feature": "AMT_CREDIT", "impact": 0.12, "feature_value": 700000}
            ],
            "negative_risk_drivers": [
                {"feature": "DAYS_EMPLOYED", "impact": -0.06, "feature_value": -1500}
            ],
        }
        result = explainer.generate_from_explanation(exp_dict)
        assert isinstance(result, str)
        assert len(result) > 10


# ─────────────────────────────────────────────────────────────────────────────
# Describe strength
# ─────────────────────────────────────────────────────────────────────────────

class TestDescribeStrength:
    @pytest.mark.parametrize("shap_value,max_shap,expected_adverb", [
        (0.60, 1.0, "significantly"),
        (0.25, 1.0, "moderately"),
        (0.05, 1.0, "slightly"),
        (0.60, 0.60, "significantly"),  # ratio = 1.0 → strong
    ])
    def test_strength_labels(
        self,
        explainer: BusinessExplainer,
        shap_value: float,
        max_shap: float,
        expected_adverb: str,
    ) -> None:
        result = explainer._describe_strength(shap_value, max_shap)
        assert result == expected_adverb
