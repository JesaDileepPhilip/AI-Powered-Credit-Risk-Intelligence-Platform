"""
app/ml/inference_service.py — Reusable inference layer for credit default prediction.

Responsibilities:
  - Load the persisted preprocessing pipeline (Phase 2) from models/
  - Load the persisted LightGBM model (Phase 3) from models/
  - Validate feature schema against training columns
  - Automatically drop TARGET and SK_ID_CURR if present
  - Return structured prediction dict:
      {
        "default_probability": float,
        "risk_score": int,
        "risk_band": str
      }

Usage::

    from app.ml.inference_service import InferenceService

    service = InferenceService()
    result = service.predict({"AMT_CREDIT": 500000, ...})
    # {"default_probability": 0.23, "risk_score": 230, "risk_band": "Medium Risk"}
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
import threading

import numpy as np
import pandas as pd

from app.ml.model_registry import ModelRegistry
from app.ml.risk_scoring import score_single
from app.utils.helpers import sanitise_feature_names
from app.utils.logger import get_logger
from config import settings

logger = get_logger(__name__)

# Columns that are always dropped before inference
_DROP_COLUMNS: Set[str] = {"TARGET", "SK_ID_CURR"}

# Preprocessing pipeline filename (from Phase 2)
_PIPELINE_FILENAME: str = "preprocessing_pipeline.pkl"
# Feature metadata filename (from Phase 2)
_FEATURE_METADATA_FILENAME: str = "feature_metadata.json"


class SchemaValidationError(ValueError):
    """Raised when the input DataFrame is missing required feature columns."""
    pass


class InferenceService:
    """
    Reusable, stateful inference service.

    On first call to ``predict()`` or ``predict_batch()``, the preprocessing
    pipeline and model are loaded lazily and cached in memory.

    Args:
        model_dir: Override the default ``models/`` directory.
    """

    def __init__(self, model_dir: Optional[Path] = None) -> None:
        self._model_dir: Path = model_dir or settings.models_dir
        self._registry = ModelRegistry(models_dir=self._model_dir)

        # Lazily loaded
        self._pipeline: Optional[Any] = None
        self._model: Optional[Any] = None
        self._training_features: Optional[List[str]] = None

        # ── THREAD SAFETY: Protect lazy-loading from concurrent calls ────────
        self._lock = threading.RLock()

    # ── Private loaders ───────────────────────────────────────────────────────

    def _load_pipeline(self) -> Any:
        """Load and cache the Phase-2 preprocessing pipeline.
        
        Thread-safe: Uses RLock to prevent concurrent loads.
        """
        if self._pipeline is None:
            with self._lock:
                # Double-check pattern after acquiring lock
                if self._pipeline is None:
                    import joblib
                    pipeline_path = self._model_dir / _PIPELINE_FILENAME
                    if not pipeline_path.exists():
                        raise FileNotFoundError(
                            f"Preprocessing pipeline not found: {pipeline_path}\n"
                            "Run Phase 2 preprocessing first."
                        )
                    self._pipeline = joblib.load(pipeline_path)
                    logger.info(f"Preprocessing pipeline loaded ← {pipeline_path}")
        return self._pipeline

    def _load_model(self) -> Any:
        """Load and cache the trained LightGBM model.
        
        Thread-safe: Uses RLock to prevent concurrent loads.
        """
        if self._model is None:
            with self._lock:
                if self._model is None:
                    self._model = self._registry.load_model()
        return self._model

    def _load_training_features(self) -> List[str]:
        """
        Return the list of feature columns (excluding TARGET) that the model
        was trained on.  Read from model_metadata.json when available,
        otherwise infer from the model's booster.
        
        Thread-safe: Uses RLock to prevent concurrent loads.
        """
        if self._training_features is None:
            with self._lock:
                if self._training_features is None:
                    try:
                        meta = self._registry.load_metadata()
                        self._training_features = meta.get("training_feature_names", [])
                        if not self._training_features:
                            raise KeyError("training_feature_names not in metadata")
                    except (FileNotFoundError, KeyError):
                        # Fall back to reading from the booster directly
                        model = self._load_model()
                        if hasattr(model, "booster_"):
                            self._training_features = model.booster_.feature_name()
                        else:
                            self._training_features = list(
                                getattr(model, "feature_name_", [])
                            )
                    logger.info(
                        f"Training feature schema loaded: {len(self._training_features)} features"
                    )
        return self._training_features

    # ── Schema validation ─────────────────────────────────────────────────────

    def _validate_schema(self, df: pd.DataFrame) -> None:
        """
        Verify that ``df`` contains all expected training features.

        Missing features are added as NaN so the pipeline can impute them.
        Extra unexpected features are silently ignored.

        Raises:
            SchemaValidationError: if the feature list is empty.
        """
        training_features = self._load_training_features()
        if not training_features:
            return  # no schema to validate against

        missing = [f for f in training_features if f not in df.columns]
        if missing:
            logger.warning(
                f"Schema validation: {len(missing)} expected features not present "
                f"in input — will be filled with NaN for imputation. "
                f"First missing: {missing[:5]}"
            )
            for col in missing:
                df[col] = np.nan

    # ── Public API ────────────────────────────────────────────────────────────

    def predict(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predict credit default risk for a single applicant record.

        Args:
            record: Dict of raw feature values.  May include TARGET and
                    SK_ID_CURR; they will be removed automatically.

        Returns:
            Dict::

                {
                    "default_probability": float,
                    "risk_score": int,
                    "risk_band": str
                }

        Raises:
            ValueError: If record is missing required feature schema.
        """
        # ── CRITICAL VALIDATION: Warn if TARGET present in single prediction ───
        if "TARGET" in record:
            logger.debug(
                "TARGET column present in single prediction input. "
                "This should not occur in production inference; "
                "it will be automatically dropped."
            )

        df = pd.DataFrame([record])
        results = self.predict_dataframe(df)
        return results[0]

    def predict_dataframe(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Predict credit default risk for a DataFrame of applicants.

        Args:
            df: Raw input DataFrame.  TARGET and SK_ID_CURR are dropped automatically.

        Returns:
            List of result dicts (same order as input rows).

        Raises:
            ValueError: If feature count mismatch detected after transformation.
        """
        pipeline = self._load_pipeline()
        model = self._load_model()

        # ── 1. Drop forbidden columns ─────────────────────────────────────────
        cols_to_drop = [c for c in _DROP_COLUMNS if c in df.columns]
        if cols_to_drop:
            logger.debug(f"Dropping columns from input: {cols_to_drop}")
            df = df.drop(columns=cols_to_drop)

        # ── CRITICAL VALIDATION: Verify TARGET was dropped if it existed ───────
        if "TARGET" in cols_to_drop:
            logger.info("✓ TARGET column correctly identified and removed from inference data")
        if "SK_ID_CURR" in cols_to_drop:
            logger.info("✓ SK_ID_CURR column correctly identified and removed from inference data")

        # ── 2. Schema validation (adds missing cols as NaN) ───────────────────
        self._validate_schema(df)

        # ── 3. Apply preprocessing pipeline ──────────────────────────────────
        transformed = pipeline.transform(df)

        # Drop any residual non-feature columns that sneak through
        residual_drop = [c for c in _DROP_COLUMNS if c in transformed.columns]
        if residual_drop:
            logger.warning(f"Dropped residual columns from pipeline output: {residual_drop}")
            transformed = transformed.drop(columns=residual_drop)

        # ── 4. Align columns to training feature order ────────────────────────
        training_features = self._load_training_features()
        if training_features:
            available = [f for f in training_features if f in transformed.columns]
            transformed = transformed[available]

            # ── CRITICAL VALIDATION: Verify feature count matches ──────────────
            if len(available) != len(training_features):
                missing_count = len(training_features) - len(available)
                logger.error(
                    f"Feature count mismatch: expected {len(training_features)}, "
                    f"but only {len(available)} found in transformed data. "
                    f"Missing {missing_count} features."
                )
                raise ValueError(
                    f"Feature count mismatch after preprocessing. "
                    f"Expected {len(training_features)} features, got {len(available)}. "
                    f"Model will not be executed. Please check feature engineering pipeline."
                )

        X = transformed.values if hasattr(transformed, "values") else transformed

        # ── CRITICAL VALIDATION: Verify X shape before prediction ──────────────
        if X.ndim != 2:
            raise ValueError(
                f"Feature matrix has invalid shape: {X.shape}. "
                f"Expected 2D array (n_samples, n_features)."
            )

        n_rows, n_features = X.shape
        if training_features and n_features != len(training_features):
            raise ValueError(
                f"Feature matrix shape mismatch: got {n_features} features, "
                f"but model expects {len(training_features)}. "
                f"This indicates a preprocessing or schema validation error."
            )
        logger.info(f"✓ Feature matrix shape validated: {n_rows:,} samples × {n_features} features")

        # ── 5. Predict probabilities ──────────────────────────────────────────
        probas = model.predict_proba(X)[:, 1].astype(float)

        # ── 6. Score ──────────────────────────────────────────────────────────
        results = []
        for p in probas:
            risk = score_single(float(p))
            results.append(risk.to_dict())

        return results

    def warmup(self) -> None:
        """
        Pre-load the pipeline and model into memory.

        Call this at application startup to avoid cold-start latency on the
        first prediction request.
        """
        self._load_pipeline()
        self._load_model()
        self._load_training_features()
        logger.info("InferenceService warmup complete.")
