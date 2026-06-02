"""
app/ml/predict.py — Batch prediction utilities.

Provides:
  - predict_proba()         : Return raw probability arrays from the trained model
  - predict_batch()         : Run the full inference pipeline and return a DataFrame
                              with default_probability, risk_score, risk_band

Internally delegates to InferenceService for schema validation and preprocessing,
then calls risk_scoring.score_batch() to produce the final output.

Usage::

    from app.ml.predict import predict_batch
    result_df = predict_batch(input_df)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from app.ml.inference_service import InferenceService
from app.ml.risk_scoring import score_batch, RiskResult
from app.utils.logger import get_logger

logger = get_logger(__name__)


def predict_proba(
    model: Any,
    X: np.ndarray,
) -> np.ndarray:
    """
    Extract positive-class probabilities from a fitted classifier.

    Args:
        model: Fitted sklearn-compatible classifier with ``predict_proba``.
        X:     Feature matrix (numpy array or DataFrame).

    Returns:
        1-D numpy array of probabilities for the positive class.
    """
    probas = model.predict_proba(X)
    if probas.ndim == 2:
        return probas[:, 1].astype(float)
    return probas.astype(float)


def predict_batch(
    df: pd.DataFrame,
    model_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Run the end-to-end inference pipeline on a raw input DataFrame.

    Steps:
      1. Load the preprocessing pipeline and model via InferenceService.
      2. Transform features.
      3. Predict default probabilities.
      4. Score and band each probability.

    Args:
        df:         Raw input DataFrame. May include TARGET and SK_ID_CURR columns;
                    they will be automatically removed before inference.
        model_dir:  Override the default ``models/`` directory.

    Returns:
        DataFrame with columns:
          - ``default_probability``
          - ``risk_score``
          - ``risk_band``
    """
    service = InferenceService(model_dir=model_dir)
    results: List[Dict] = []

    logger.info(f"Running batch prediction on {len(df):,} rows …")
    for _, row in df.iterrows():
        result = service.predict(row.to_dict())
        results.append(result)

    out_df = pd.DataFrame(results)
    logger.info(
        f"Batch prediction complete: {len(out_df):,} predictions generated."
    )
    return out_df


def predict_single_dict(
    record: Dict,
    model_dir: Optional[Path] = None,
) -> Dict:
    """
    Run inference on a single record represented as a plain dict.

    Args:
        record:    Dict of feature values. TARGET and SK_ID_CURR are ignored if present.
        model_dir: Override the default models directory.

    Returns:
        Dict with ``default_probability``, ``risk_score``, ``risk_band``.
    """
    service = InferenceService(model_dir=model_dir)
    return service.predict(record)
