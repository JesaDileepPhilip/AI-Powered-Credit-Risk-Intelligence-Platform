"""
app/explainability/business_explainer.py — Human-readable explanation generator.

Converts raw SHAP driver lists into plain-English narratives suitable for:
  - Credit officers reviewing applicant profiles
  - Audit review and regulatory documentation
  - Business stakeholders without technical ML background

Design:
  - Rule-based template system: zero LLM dependency
  - Feature name → business label mapping with smart fallback
  - Sentence construction adapts to risk band and driver strength
  - Always states the overall risk decision first

Usage::

    from app.explainability.business_explainer import BusinessExplainer

    explainer = BusinessExplainer()
    narrative = explainer.generate(
        risk_band="High Risk",
        default_probability=0.82,
        positive_drivers=[{"feature": "AMT_CREDIT", "shap_value": 0.12, "value": 900000}],
        negative_drivers=[{"feature": "DAYS_EMPLOYED", "shap_value": -0.05, "value": -1200}],
    )
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from app.utils.logger import get_logger

logger = get_logger(__name__)

# ── Business label mapping ────────────────────────────────────────────────────
# Maps internal feature names / substrings → human-readable descriptions.
# Order matters: more specific patterns should come before general ones.
FEATURE_LABEL_MAP: Dict[str, str] = {
    # Credit amounts
    "amt_credit": "credit amount requested",
    "credit_to_income": "credit-to-income ratio",
    "amt_goods_price": "goods price relative to credit",
    "amt_annuity": "loan annuity amount",
    "annuity_to_income": "annuity-to-income ratio",
    # Income
    "amt_income_total": "total annual income",
    "debt_to_income": "debt-to-income ratio",
    # Employment
    "days_employed": "employment duration",
    "employment_age": "employment-to-age ratio",
    "name_income_type": "income type",
    "organization_type": "employer organisation type",
    # Demographics
    "days_birth": "applicant age",
    "code_gender": "applicant gender",
    "cnt_children": "number of dependants",
    "cnt_fam_members": "family size",
    # Education & social
    "name_education_type": "education level",
    "name_family_status": "family/marital status",
    # Property
    "flag_own_car": "vehicle ownership",
    "flag_own_realty": "property ownership",
    # External scores
    "ext_source_1": "external credit score (bureau 1)",
    "ext_source_2": "external credit score (bureau 2)",
    "ext_source_3": "external credit score (bureau 3)",
    # Bureau / history
    "bureau": "bureau credit history",
    "prev_app": "previous loan application history",
    "days_credit": "days since last credit bureau record",
    "credit_active": "active credit lines",
    "amt_credit_sum": "total outstanding credit",
    "amt_credit_sum_debt": "outstanding debt amount",
    # Contact / document
    "flag_document": "document submission",
    "flag_cont_mobile": "mobile contact availability",
    "flag_email": "email contact availability",
    # Region / geo
    "region_rating": "regional creditworthiness rating",
    "reg_city_not_work_city": "residence vs. workplace city mismatch",
    # Generic fallback applied if nothing matches
}

# ── Narrative templates ───────────────────────────────────────────────────────

_RISK_INTRO: Dict[str, str] = {
    "High Risk": (
        "Based on the model analysis, this applicant presents a HIGH default risk "
        f"with a predicted default probability of {{prob:.1%}}."
    ),
    "Medium Risk": (
        "Based on the model analysis, this applicant presents a MODERATE default risk "
        f"with a predicted default probability of {{prob:.1%}}."
    ),
    "Low Risk": (
        "Based on the model analysis, this applicant presents a LOW default risk "
        f"with a predicted default probability of {{prob:.1%}}."
    ),
}

_RISK_COLOUR: Dict[str, str] = {
    "High Risk": "primarily",
    "Medium Risk": "mainly",
    "Low Risk": "partly",
}


class BusinessExplainer:
    """
    Generates plain-English credit risk narratives from SHAP driver data.

    The narrative is structured as:
      1. Risk decision sentence (High / Medium / Low Risk + probability)
      2. Primary risk-increasing factors (positive SHAP drivers)
      3. Risk-mitigating factors (negative SHAP drivers), if any
      4. Audit note (score + interpretation guidance)

    Args:
        feature_label_map: Override or extend the default business label mapping.
        top_positive_n:    Max positive drivers to mention.
        top_negative_n:    Max negative drivers to mention.
    """

    def __init__(
        self,
        feature_label_map: Optional[Dict[str, str]] = None,
        top_positive_n: int = 3,
        top_negative_n: int = 2,
    ) -> None:
        self._labels = {**FEATURE_LABEL_MAP}
        if feature_label_map:
            self._labels.update(feature_label_map)
        self.top_positive_n = top_positive_n
        self.top_negative_n = top_negative_n

    # ── Label helpers ─────────────────────────────────────────────────────────

    def _feature_to_label(self, feature_name: str) -> str:
        """
        Convert an internal feature name to a business-friendly label.

        Tries exact match first, then substring search (case-insensitive),
        then falls back to a title-cased version of the raw name.
        """
        lower = feature_name.lower()

        # Exact match
        if lower in self._labels:
            return self._labels[lower]

        # Substring match (first hit wins — dict is ordered in Python 3.7+)
        for key, label in self._labels.items():
            if key in lower:
                return label

        # Graceful fallback
        return feature_name.replace("_", " ").replace("  ", " ").title()

    def _describe_strength(self, shap_value: float, max_shap: float) -> str:
        ratio = abs(shap_value) / max(max_shap, 1e-9)
        if ratio >= 0.5:
            return "significantly"
        if ratio >= 0.2:
            return "moderately"
        return "slightly"

    # ── Main API ──────────────────────────────────────────────────────────────

    def generate(
        self,
        risk_band: str,
        default_probability: float,
        risk_score: int,
        positive_drivers: List[Dict[str, Any]],
        negative_drivers: List[Dict[str, Any]],
    ) -> str:
        """
        Generate a plain-English explanation narrative.

        Args:
            risk_band:           "High Risk" | "Medium Risk" | "Low Risk"
            default_probability: Float in [0, 1].
            risk_score:          Integer risk score [0, 1000].
            positive_drivers:    List of dicts with keys: feature, shap_value[, value].
                                 Sorted by shap_value descending.
            negative_drivers:    List of dicts with keys: feature, shap_value[, value].
                                 Sorted by shap_value ascending (most negative first).

        Returns:
            Multi-sentence plain-English explanation string.
        """
        sentences = []

        # ── 1. Risk decision ──────────────────────────────────────────────────
        intro_template = _RISK_INTRO.get(risk_band, _RISK_INTRO["Medium Risk"])
        sentences.append(intro_template.format(prob=default_probability))

        # ── 2. Positive drivers (risk-increasing) ─────────────────────────────
        top_pos = positive_drivers[: self.top_positive_n]
        if top_pos:
            max_shap = max(abs(d.get("shap_value", d.get("impact", 0))) for d in top_pos)
            adverb = _RISK_COLOUR.get(risk_band, "mainly")

            driver_phrases = []
            for d in top_pos:
                sv = d.get("shap_value", d.get("impact", 0))
                feat_label = self._feature_to_label(d["feature"])
                strength = self._describe_strength(sv, max_shap)
                driver_phrases.append(f"{strength} elevated {feat_label}")

            if len(driver_phrases) == 1:
                drivers_str = driver_phrases[0]
            elif len(driver_phrases) == 2:
                drivers_str = f"{driver_phrases[0]} and {driver_phrases[1]}"
            else:
                drivers_str = (
                    ", ".join(driver_phrases[:-1]) + f", and {driver_phrases[-1]}"
                )

            sentences.append(
                f"The risk is {adverb} driven by {drivers_str}."
            )

        # ── 3. Negative drivers (risk-reducing) ───────────────────────────────
        top_neg = negative_drivers[: self.top_negative_n]
        if top_neg:
            mitigant_phrases = []
            for d in top_neg:
                feat_label = self._feature_to_label(d["feature"])
                mitigant_phrases.append(feat_label)

            if len(mitigant_phrases) == 1:
                mit_str = mitigant_phrases[0]
            else:
                mit_str = " and ".join(mitigant_phrases)

            sentences.append(
                f"Partially mitigating factors include {mit_str}, "
                "which reduce the estimated default probability."
            )
        else:
            if risk_band == "High Risk":
                sentences.append(
                    "No significant risk-mitigating factors were identified for this applicant."
                )

        # ── 4. Audit note ─────────────────────────────────────────────────────
        sentences.append(
            f"Risk score: {risk_score} / 1000. "
            "This assessment is generated by the Credit Risk Intelligence Platform "
            "and should be reviewed alongside manual underwriting criteria."
        )

        narrative = " ".join(sentences)
        logger.debug(f"Business narrative generated ({len(narrative)} chars)")
        return narrative

    def generate_from_explanation(self, explanation: Dict[str, Any]) -> str:
        """
        Convenience wrapper that unpacks an ExplanationService result dict.

        Args:
            explanation: Dict returned by ExplanationService.explain().

        Returns:
            Plain-English narrative string.
        """
        return self.generate(
            risk_band=explanation["risk_band"],
            default_probability=explanation["default_probability"],
            risk_score=explanation["risk_score"],
            positive_drivers=explanation.get("positive_risk_drivers", []),
            negative_drivers=explanation.get("negative_risk_drivers", []),
        )
