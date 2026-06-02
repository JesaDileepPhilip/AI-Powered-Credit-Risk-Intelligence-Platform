"""
app/explainability/shap_explainer.py — SHAP TreeExplainer wrapper.

Responsibilities:
  - Fit a shap.TreeExplainer on the trained LightGBM model
  - Compute global SHAP values for a dataset (background)
  - Compute local SHAP values for a single customer record
  - Expose raw shap_values arrays + expected_value for downstream use
  - Lazy-load the model and preprocessor from ModelRegistry

Design:
  - SHAPExplainer is stateful: call fit() once, then explain_global() /
    explain_local() as many times as needed.
  - All heavy shap imports are inside the class to keep import time fast.

Usage::

    explainer = SHAPExplainer()
    explainer.fit(X_background, feature_names)

    # Global
    global_result = explainer.explain_global(X_background)

    # Local
    local_result = explainer.explain_local(x_single)
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from app.utils.logger import get_logger

logger = get_logger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
# Maximum number of background samples fed to the SHAP explainer.
# Larger values improve accuracy but increase compute time.
SHAP_BACKGROUND_SAMPLES: int = 500
TOP_N_FEATURES: int = 20


class SHAPExplainer:
    """
    SHAP TreeExplainer wrapper for LightGBM credit risk models.

    Args:
        model:         Fitted LGBMClassifier (or any tree-based sklearn estimator).
        feature_names: List of feature names in the same order as the training
                       feature matrix. Required for named explanations.
        background_samples: Max rows of training data used as the SHAP background.

    Example::

        from app.explainability.shap_explainer import SHAPExplainer
        explainer = SHAPExplainer(model=lgbm_model, feature_names=feature_list)
        explainer.fit(X_train)
        result = explainer.explain_local(x_record)
    """

    def __init__(
        self,
        model: Any,
        feature_names: List[str],
        background_samples: int = SHAP_BACKGROUND_SAMPLES,
    ) -> None:
        self.model = model
        self.feature_names = list(feature_names)
        self.background_samples = background_samples

        self._explainer: Optional[Any] = None
        self._expected_value: Optional[float] = None
        self._is_fitted: bool = False

    # ── Fitting ───────────────────────────────────────────────────────────────

    def fit(self, X_background: np.ndarray) -> "SHAPExplainer":
        """
        Initialise the SHAP TreeExplainer using a background dataset.

        Args:
            X_background: 2-D numpy array of training features (n_samples, n_features).
                          A random sub-sample of ``background_samples`` rows is used
                          to speed up computation.

        Returns:
            self (for method chaining)
        """
        import shap

        n = len(X_background)
        if n > self.background_samples:
            rng = np.random.default_rng(42)
            idx = rng.choice(n, size=self.background_samples, replace=False)
            background = X_background[idx]
            logger.info(
                f"SHAPExplainer: sub-sampled {self.background_samples} / {n} rows "
                "for background"
            )
        else:
            background = X_background
            logger.info(f"SHAPExplainer: using all {n} rows as background")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._explainer = shap.TreeExplainer(
                self.model,
                data=background,
                feature_perturbation="interventional",
                model_output="probability",
            )

        # Retrieve expected value (base rate for the positive class)
        ev = self._explainer.expected_value
        if isinstance(ev, (list, np.ndarray)):
            # Binary classification: take the positive-class value
            self._expected_value = float(ev[1]) if len(ev) > 1 else float(ev[0])
        else:
            self._expected_value = float(ev)

        self._is_fitted = True
        logger.info(
            f"SHAPExplainer fitted. Base value (expected_value) = "
            f"{self._expected_value:.4f}"
        )
        return self

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _check_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError(
                "SHAPExplainer is not fitted. Call fit(X_background) first."
            )

    def _to_2d_array(self, X: Any) -> np.ndarray:
        """Ensure X is a 2-D float32 numpy array."""
        if hasattr(X, "values"):
            X = X.values
        X = np.asarray(X, dtype=np.float32)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        return X

    def _compute_shap_values(self, X: np.ndarray) -> np.ndarray:
        """
        Run the SHAP TreeExplainer and return the positive-class SHAP values.

        For binary classification LightGBM models SHAP returns either:
          - A (n_samples, n_features) array        [single output]
          - A list of two (n_samples, n_features)  [multi-output]
        We always take the positive-class slice.
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            raw = self._explainer.shap_values(X, check_additivity=False)

        if isinstance(raw, list):
            # Binary classification: index 1 = positive class
            shap_vals = np.array(raw[1])
        else:
            shap_vals = np.array(raw)

        return shap_vals

    # ── Global explanations ───────────────────────────────────────────────────

    def explain_global(
        self, X: np.ndarray
    ) -> Dict[str, Any]:
        """
        Compute global SHAP feature importances over a dataset.

        Args:
            X: 2-D feature matrix (n_samples, n_features).

        Returns:
            Dict with keys:
              - ``shap_values``:         (n_samples, n_features) array
              - ``mean_abs_shap``:       (n_features,) array — mean |SHAP| per feature
              - ``feature_names``:       list of feature names
              - ``expected_value``:      float — model base rate
              - ``top_features``:        list of dicts [{feature, importance}]
                                         sorted by mean |SHAP| descending
        """
        self._check_fitted()
        X2d = self._to_2d_array(X)
        shap_vals = self._compute_shap_values(X2d)

        mean_abs = np.abs(shap_vals).mean(axis=0)
        order = np.argsort(mean_abs)[::-1]

        top_features = [
            {
                "feature": self.feature_names[i],
                "importance": float(mean_abs[i]),
            }
            for i in order[:TOP_N_FEATURES]
        ]

        logger.info(
            f"Global SHAP computed over {X2d.shape[0]:,} samples. "
            f"Top feature: {top_features[0]['feature']} "
            f"(mean|SHAP|={top_features[0]['importance']:.4f})"
        )

        return {
            "shap_values": shap_vals,
            "mean_abs_shap": mean_abs,
            "feature_names": self.feature_names,
            "expected_value": self._expected_value,
            "top_features": top_features,
        }

    # ── Local explanations ────────────────────────────────────────────────────

    def explain_local(
        self,
        x: np.ndarray,
        top_n: int = 10,
    ) -> Dict[str, Any]:
        """
        Compute a local (per-instance) SHAP explanation.

        Args:
            x:     1-D feature vector or (1, n_features) array.
            top_n: Number of top positive and negative drivers to return.

        Returns:
            Dict with keys:
              - ``shap_values``:           (n_features,) array
              - ``feature_names``:         list of feature names
              - ``feature_values``:        list of raw feature values
              - ``expected_value``:        float — model base rate
              - ``predicted_probability``: float — model output for this record
              - ``positive_drivers``:      list of dicts [{feature, value, shap_value}]
                                           SHAP > 0  (increases default risk), sorted desc
              - ``negative_drivers``:      list of dicts [{feature, value, shap_value}]
                                           SHAP < 0  (reduces default risk), sorted asc
        """
        self._check_fitted()
        X2d = self._to_2d_array(x)
        shap_vals = self._compute_shap_values(X2d)[0]  # single row → 1-D

        predicted_prob = float(self._expected_value + shap_vals.sum())
        predicted_prob = max(0.0, min(1.0, predicted_prob))

        # Build per-feature records
        feature_records = [
            {
                "feature": self.feature_names[i],
                "value": float(X2d[0, i]),
                "shap_value": float(shap_vals[i]),
            }
            for i in range(len(self.feature_names))
        ]

        positive = sorted(
            [r for r in feature_records if r["shap_value"] > 0],
            key=lambda r: r["shap_value"],
            reverse=True,
        )[:top_n]

        negative = sorted(
            [r for r in feature_records if r["shap_value"] < 0],
            key=lambda r: r["shap_value"],
        )[:top_n]

        logger.debug(
            f"Local SHAP: predicted_prob={predicted_prob:.4f} | "
            f"base={self._expected_value:.4f} | "
            f"top positive driver: "
            f"{positive[0]['feature'] if positive else 'none'}"
        )

        return {
            "shap_values": shap_vals,
            "feature_names": self.feature_names,
            "feature_values": X2d[0].tolist(),
            "expected_value": self._expected_value,
            "predicted_probability": predicted_prob,
            "positive_drivers": positive,
            "negative_drivers": negative,
        }

    # ── SHAP Explainer object (for plotting) ──────────────────────────────────

    def get_shap_explainer(self) -> Any:
        """Return the underlying shap.TreeExplainer object (for custom plots)."""
        self._check_fitted()
        return self._explainer

    @property
    def expected_value(self) -> float:
        """Return the base model output (expected value / base rate)."""
        self._check_fitted()
        return self._expected_value  # type: ignore[return-value]
