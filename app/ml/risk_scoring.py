"""
app/ml/risk_scoring.py — Risk scoring layer for credit default probability.

Converts a raw default probability [0.0, 1.0] into:
  - A numeric risk_score  = int(probability * 1000)   range [0, 1000]
  - A risk_band string:
        Low Risk    : probability in [0.00, 0.20)
        Medium Risk : probability in [0.20, 0.50)
        High Risk   : probability in [0.50, 1.00]

Examples:
    0.12 -> score 120  -> Low Risk
    0.38 -> score 380  -> Medium Risk
    0.82 -> score 820  -> High Risk
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from app.utils.logger import get_logger

logger = get_logger(__name__)

# ── Band thresholds (inclusive lower, exclusive upper except last) ─────────────
_LOW_UPPER: float = 0.20
_MEDIUM_UPPER: float = 0.50

BAND_LOW = "Low Risk"
BAND_MEDIUM = "Medium Risk"
BAND_HIGH = "High Risk"


@dataclass(frozen=True)
class RiskResult:
    """Structured result from the risk scoring layer."""

    default_probability: float
    risk_score: int
    risk_band: str

    def to_dict(self) -> dict:
        return {
            "default_probability": round(self.default_probability, 6),
            "risk_score": self.risk_score,
            "risk_band": self.risk_band,
        }


def compute_risk_band(probability: float) -> str:
    """
    Map a default probability to a risk band label.

    Args:
        probability: Float in [0.0, 1.0].

    Returns:
        One of "Low Risk", "Medium Risk", "High Risk".

    Raises:
        ValueError: If probability is outside [0.0, 1.0].
    """
    if not 0.0 <= probability <= 1.0:
        raise ValueError(
            f"probability must be in [0.0, 1.0], got {probability:.6f}"
        )
    if probability < _LOW_UPPER:
        return BAND_LOW
    if probability < _MEDIUM_UPPER:
        return BAND_MEDIUM
    return BAND_HIGH


def compute_risk_score(probability: float) -> int:
    """
    Convert probability to an integer risk score on [0, 1000].

    Args:
        probability: Float in [0.0, 1.0].

    Returns:
        Integer risk score.
    """
    if not 0.0 <= probability <= 1.0:
        raise ValueError(
            f"probability must be in [0.0, 1.0], got {probability:.6f}"
        )
    return int(probability * 1000)


def score_single(probability: float) -> RiskResult:
    """
    Compute risk result for a single applicant.

    Args:
        probability: Default probability in [0.0, 1.0].

    Returns:
        :class:`RiskResult` dataclass.

    Example::

        result = score_single(0.38)
        # RiskResult(default_probability=0.38, risk_score=380, risk_band='Medium Risk')
    """
    if not 0.0 <= probability <= 1.0:
        raise ValueError(
            f"probability must be in [0.0, 1.0], got {probability:.6f}"
        )

    band = compute_risk_band(probability)
    score = compute_risk_score(probability)

    logger.debug(
        f"Risk scoring: prob={probability:.4f} -> score={score} -> {band}"
    )
    return RiskResult(
        default_probability=float(probability),
        risk_score=score,
        risk_band=band,
    )


def score_batch(probabilities: List[float]) -> List[RiskResult]:
    """
    Compute risk results for a batch of applicants.

    Args:
        probabilities: List of default probabilities in [0.0, 1.0].

    Returns:
        List of :class:`RiskResult` instances in the same order.
    """
    results = [score_single(p) for p in probabilities]
    low = sum(1 for r in results if r.risk_band == BAND_LOW)
    mid = sum(1 for r in results if r.risk_band == BAND_MEDIUM)
    high = sum(1 for r in results if r.risk_band == BAND_HIGH)
    logger.info(
        f"Batch risk scoring ({len(results)} records): "
        f"Low={low} | Medium={mid} | High={high}"
    )
    return results
