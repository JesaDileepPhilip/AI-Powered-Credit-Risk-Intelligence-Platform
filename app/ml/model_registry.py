"""
app/ml/model_registry.py — Model and preprocessing-pipeline persistence layer.

Responsibilities:
  - Save / load the trained LightGBM model (pickle via joblib)
  - Save / load model metadata  (JSON)
  - Save / load training metrics (JSON)
  - Provide a typed ModelRegistry context that centralises all paths

All paths default to the project-level ``models/`` directory defined in
``config.settings``, but can be overridden for testing.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import joblib

from app.utils.helpers import make_json_serialisable
from app.utils.logger import get_logger
from config import settings

logger = get_logger(__name__)

# ── Default filenames ─────────────────────────────────────────────────────────
LGBM_MODEL_FILENAME: str = "lightgbm_model.pkl"
MODEL_METADATA_FILENAME: str = "model_metadata.json"
TRAINING_METRICS_FILENAME: str = "training_metrics.json"


class ModelRegistry:
    """
    Centralised persistence layer for trained model artefacts.

    Example::

        registry = ModelRegistry()
        registry.save_model(lgbm_clf)
        registry.save_metadata(metadata_dict)
        registry.save_training_metrics(metrics_dict)

        model = ModelRegistry.load_model()
    """

    def __init__(self, models_dir: Optional[Path] = None) -> None:
        self.models_dir: Path = models_dir or settings.models_dir
        self.models_dir.mkdir(parents=True, exist_ok=True)

    # ── Paths ─────────────────────────────────────────────────────────────────

    @property
    def model_path(self) -> Path:
        return self.models_dir / LGBM_MODEL_FILENAME

    @property
    def metadata_path(self) -> Path:
        return self.models_dir / MODEL_METADATA_FILENAME

    @property
    def training_metrics_path(self) -> Path:
        return self.models_dir / TRAINING_METRICS_FILENAME

    # ── Save ──────────────────────────────────────────────────────────────────

    def save_model(self, model: Any) -> Path:
        """Persist a trained model to disk using joblib."""
        joblib.dump(model, self.model_path)
        logger.info(f"Model saved → {self.model_path}")
        return self.model_path

    def save_metadata(self, metadata: Dict[str, Any]) -> Path:
        """Write model metadata JSON file."""
        safe = make_json_serialisable(metadata)
        safe["saved_at"] = datetime.now().isoformat()
        with open(self.metadata_path, "w", encoding="utf-8") as fh:
            json.dump(safe, fh, indent=2, default=str)
        logger.info(f"Model metadata saved → {self.metadata_path}")
        return self.metadata_path

    def save_training_metrics(self, metrics: Dict[str, Any]) -> Path:
        """Write training metrics JSON file."""
        safe = make_json_serialisable(metrics)
        safe["saved_at"] = datetime.now().isoformat()
        with open(self.training_metrics_path, "w", encoding="utf-8") as fh:
            json.dump(safe, fh, indent=2, default=str)
        logger.info(f"Training metrics saved → {self.training_metrics_path}")
        return self.training_metrics_path

    # ── Load ──────────────────────────────────────────────────────────────────

    def load_model(self) -> Any:
        """Load and return the persisted LightGBM model."""
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"No model found at {self.model_path}. "
                "Run train.py first to train and save the model."
            )
        model = joblib.load(self.model_path)
        logger.info(f"Model loaded ← {self.model_path}")
        return model

    def load_metadata(self) -> Dict[str, Any]:
        """Load and return model metadata dict."""
        if not self.metadata_path.exists():
            raise FileNotFoundError(f"Metadata not found: {self.metadata_path}")
        with open(self.metadata_path, encoding="utf-8") as fh:
            return json.load(fh)

    def load_training_metrics(self) -> Dict[str, Any]:
        """Load and return training metrics dict."""
        if not self.training_metrics_path.exists():
            raise FileNotFoundError(
                f"Training metrics not found: {self.training_metrics_path}"
            )
        with open(self.training_metrics_path, encoding="utf-8") as fh:
            return json.load(fh)

    # ── Convenience class-methods ─────────────────────────────────────────────

    @classmethod
    def default(cls) -> "ModelRegistry":
        """Return a registry pointing at the project-default models directory."""
        return cls(models_dir=settings.models_dir)
