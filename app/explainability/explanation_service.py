"""
app/explainability/explanation_service.py — Reusable end-to-end explanation service.

Integrates Phase 3 InferenceService with Phase 4 SHAP explainability to provide
a single explain(record) → structured_result workflow.

Output schema::

    {
        "default_probability": float,
        "risk_score":          int,
        "risk_band":           str,
        "positive_risk_drivers": [
            {"feature": str, "impact": float, "feature_value": float},
            ...
        ],
        "negative_risk_drivers": [
            {"feature": str, "impact": float, "feature_value": float},
            ...
        ],
        "expected_value":      float,
        "business_narrative":  str,
        "shap_values":         list[float],
    }

Usage::

    from app.explainability.explanation_service import ExplanationService

    service = ExplanationService()
    result = service.explain({"AMT_CREDIT": 500000, ...})
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from app.explainability.business_explainer import BusinessExplainer
from app.explainability.shap_explainer import SHAPExplainer
from app.ml.inference_service import InferenceService
from app.ml.model_registry import ModelRegistry
from app.ml.risk_scoring import score_single
from app.utils.logger import get_logger
from config import settings

logger = get_logger(__name__)

# Columns to always strip from raw input before processing
_DROP_COLUMNS = {"TARGET", "SK_ID_CURR"}

# Preprocessing pipeline filename (from Phase 2)
_PIPELINE_FILENAME = "preprocessing_pipeline.pkl"

# Number of top risk drivers to include in the structured output
TOP_DRIVERS_N: int = 5


class ExplanationService:
    """
    Reusable, stateful service that produces predictions + SHAP explanations.

    On first call the model, preprocessing pipeline, and SHAP explainer are
    lazy-loaded and cached.  A small background dataset (from the processed
    train CSV) is used as the SHAP explainer background.

    Args:
        model_dir:           Override the default ``models/`` directory.
        background_csv:      Path to a processed CSV used as SHAP background.
                             Defaults to ``data/processed/train.csv``.
        background_samples:  Maximum rows sampled from background_csv.
        top_drivers_n:       Number of positive/negative drivers to return.
    """

    def __init__(
        self,
        model_dir: Optional[Path] = None,
        background_csv: Optional[Path] = None,
        background_samples: int = 500,
        top_drivers_n: int = TOP_DRIVERS_N,
    ) -> None:
        self._model_dir = model_dir or settings.models_dir
        self._background_csv = background_csv or (
            settings.data_processed_dir / "train.csv"
        )
        self._background_samples = background_samples
        self.top_drivers_n = top_drivers_n

        # Lazy-loaded components
        self._inference_service: Optional[InferenceService] = None
        self._shap_explainer: Optional[SHAPExplainer] = None
        self._business_explainer = BusinessExplainer()
        self._feature_names: Optional[List[str]] = None

        self._lock = threading.RLock()

    # ── Private loaders ───────────────────────────────────────────────────────

    def _get_inference_service(self) -> InferenceService:
        if self._inference_service is None:
            with self._lock:
                if self._inference_service is None:
                    self._inference_service = InferenceService(
                        model_dir=self._model_dir
                    )
                    self._inference_service.warmup()
        return self._inference_service

    def _get_feature_names(self) -> List[str]:
        if self._feature_names is None:
            with self._lock:
                if self._feature_names is None:
                    svc = self._get_inference_service()
                    self._feature_names = svc._load_training_features()
        return self._feature_names

    def _get_shap_explainer(self) -> SHAPExplainer:
        """Lazy-load and fit the SHAP explainer using background training data."""
        if self._shap_explainer is None:
            with self._lock:
                if self._shap_explainer is None:
                    registry = ModelRegistry(models_dir=self._model_dir)
                    model = registry.load_model()
                    feature_names = self._get_feature_names()

                    # Load background dataset
                    X_background = self._load_background(feature_names)

                    explainer = SHAPExplainer(
                        model=model,
                        feature_names=feature_names,
                        background_samples=self._background_samples,
                    )
                    explainer.fit(X_background)
                    self._shap_explainer = explainer
                    logger.info("SHAP explainer fitted and cached.")
        return self._shap_explainer

    def _load_background(self, feature_names: List[str]) -> np.ndarray:
        """
        Load the background dataset for SHAP from the processed train CSV.

        If the CSV doesn't exist, generates a small zero-filled fallback so the
        service degrades gracefully (useful in testing).
        """
        if not self._background_csv.exists():
            logger.warning(
                f"Background CSV not found: {self._background_csv}. "
                "Using zero-filled fallback background (n=50). "
                "Run Phase 2 preprocessing to generate proper background data."
            )
            return np.zeros((50, len(feature_names)), dtype=np.float32)

        df = pd.read_csv(self._background_csv, low_memory=False)

        # Drop non-feature columns
        drop = [c for c in _DROP_COLUMNS if c in df.columns]
        df = df.drop(columns=drop, errors="ignore")

        # Align to training feature order
        available = [f for f in feature_names if f in df.columns]
        if available:
            df = df[available]

        X = df.fillna(0.0).values.astype(np.float32)
        logger.info(
            f"Background dataset loaded: {X.shape[0]:,} rows × {X.shape[1]} features"
        )
        return X

    # ── Input preprocessing ───────────────────────────────────────────────────

    def _preprocess_record(
        self,
        record: Dict[str, Any],
    ) -> np.ndarray:
        """
        Apply the Phase 2 preprocessing pipeline to a raw record dict and
        return the aligned feature vector ready for the model.
        """
        svc = self._get_inference_service()
        pipeline = svc._load_pipeline()
        feature_names = self._get_feature_names()

        df = pd.DataFrame([record])
        drop = [c for c in _DROP_COLUMNS if c in df.columns]
        df = df.drop(columns=drop, errors="ignore")

        # Schema validation: add missing columns as NaN
        for col in feature_names:
            if col not in df.columns:
                df[col] = np.nan

        transformed = pipeline.transform(df)

        # Drop residual columns
        residual = [c for c in _DROP_COLUMNS if c in transformed.columns]
        if residual:
            transformed = transformed.drop(columns=residual)

        # Align to training feature order
        available = [f for f in feature_names if f in transformed.columns]
        transformed = transformed[available]

        return transformed.values.astype(np.float32)

    # ── Public API ────────────────────────────────────────────────────────────

    def explain(
        self,
        record: Dict[str, Any],
        customer_id: str = "unknown",
        include_shap_array: bool = True,
    ) -> Dict[str, Any]:
        """
        Produce a full prediction + SHAP explanation for a single applicant.

        Args:
            record:             Raw feature dict. TARGET and SK_ID_CURR
                                are dropped automatically.
            customer_id:        Identifier for logging (not used in computation).
            include_shap_array: If True, include the raw SHAP value array in output.

        Returns:
            Structured dict with prediction, risk band, SHAP drivers, and
            a plain-English business narrative.

        Schema::

            {
                "default_probability":   float,
                "risk_score":            int,
                "risk_band":             str,
                "positive_risk_drivers": [{feature, impact, feature_value}, ...],
                "negative_risk_drivers": [{feature, impact, feature_value}, ...],
                "expected_value":        float,
                "business_narrative":    str,
                "shap_values":           list[float]   # if include_shap_array=True
            }
        """
        logger.info(f"ExplanationService.explain() for customer_id={customer_id!r}")

        # ── 1. Preprocess ─────────────────────────────────────────────────────
        X = self._preprocess_record(record)  # (1, n_features)

        # ── 2. Predict ────────────────────────────────────────────────────────
        svc = self._get_inference_service()
        model = svc._load_model()
        prob = float(model.predict_proba(X)[0, 1])
        risk = score_single(prob)

        # ── 3. SHAP local explanation ─────────────────────────────────────────
        shap_exp = self._get_shap_explainer()
        local = shap_exp.explain_local(X[0], top_n=self.top_drivers_n)

        # ── 4. Format drivers ─────────────────────────────────────────────────
        positive_drivers = [
            {
                "feature": d["feature"],
                "impact": round(d["shap_value"], 6),
                "feature_value": round(d["value"], 6),
            }
            for d in local["positive_drivers"]
        ]
        negative_drivers = [
            {
                "feature": d["feature"],
                "impact": round(d["shap_value"], 6),
                "feature_value": round(d["value"], 6),
            }
            for d in local["negative_drivers"]
        ]

        # ── 5. Business narrative ─────────────────────────────────────────────
        narrative = self._business_explainer.generate(
            risk_band=risk.risk_band,
            default_probability=prob,
            risk_score=risk.risk_score,
            positive_drivers=positive_drivers,
            negative_drivers=negative_drivers,
        )

        result: Dict[str, Any] = {
            "default_probability": round(prob, 6),
            "risk_score": risk.risk_score,
            "risk_band": risk.risk_band,
            "positive_risk_drivers": positive_drivers,
            "negative_risk_drivers": negative_drivers,
            "expected_value": round(local["expected_value"], 6),
            "business_narrative": narrative,
        }
        if include_shap_array:
            result["shap_values"] = [round(float(v), 6) for v in local["shap_values"]]

        logger.info(
            f"Explanation complete: prob={prob:.4f} | band={risk.risk_band} | "
            f"top_positive={positive_drivers[0]['feature'] if positive_drivers else 'none'}"
        )
        return result

    def explain_batch(
        self,
        records: List[Dict[str, Any]],
        include_shap_array: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Produce explanations for a list of applicant records.

        Args:
            records:            List of raw feature dicts.
            include_shap_array: If True, include raw SHAP arrays in each result.

        Returns:
            List of explanation dicts (same order as input).
        """
        logger.info(f"ExplanationService.explain_batch() — {len(records)} records")
        results = []
        for i, record in enumerate(records):
            result = self.explain(
                record,
                customer_id=str(i),
                include_shap_array=include_shap_array,
            )
            results.append(result)
        return results

    def warmup(self) -> None:
        """
        Pre-load all components into memory.

        Call once at application startup to avoid cold-start latency.
        """
        self._get_inference_service()
        self._get_feature_names()
        self._get_shap_explainer()
        logger.info("ExplanationService warmup complete.")
