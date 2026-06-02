"""
app/ml/preprocessing.py — Imputation, encoding, and sklearn Pipeline orchestration.

Components:
  - ``MissingValueImputer``: median (numeric) / mode (categorical) with stats tracking
  - ``CategoricalEncoder``: label encoding (binary) / one-hot (multi-category)
  - ``PreprocessingPipeline``: end-to-end orchestrator
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from app.ml.artifact_manager import ArtifactManager
from app.ml.data_splitter import DataSplitter
from app.ml.data_validator import DataValidator
from app.ml.feature_engineering import FeatureEngineer
from app.ml.feature_selector import FeatureSelector
from app.ml.settings import (
    BINARY_CATEGORY_MAX_UNIQUE,
    CATEGORICAL_IMPUTATION_STRATEGY,
    DATA_PROCESSED_DIR,
    DATA_RAW_DIR,
    ENCODING_STRATEGY,
    ID_COLUMNS,
    INPUT_FILENAME,
    LABEL_ENCODER_UNKNOWN_VALUE,
    NUMERIC_IMPUTATION_STRATEGY,
    TARGET_COLUMN,
)
from app.utils.helpers import timer
from app.utils.logger import get_logger

logger = get_logger(__name__)


def _is_categorical(series: pd.Series) -> bool:
    """Return True if the column should be treated as categorical."""
    if pd.api.types.is_object_dtype(series) or isinstance(series.dtype, pd.CategoricalDtype):
        return True
    if pd.api.types.is_string_dtype(series):
        return True
    return False


def _safe_label_transform(
    encoder: LabelEncoder,
    values: pd.Series,
) -> np.ndarray:
    """
    Apply a fitted LabelEncoder, mapping unseen categories to -1.

    Prevents inference failures when production data contains categories
    not observed during training.
    """
    str_values = values.astype(str)
    known = set(encoder.classes_)
    mapping = {cls: idx for idx, cls in enumerate(encoder.classes_)}
    return str_values.map(
        lambda v: mapping.get(v, LABEL_ENCODER_UNKNOWN_VALUE)
    ).to_numpy(dtype=np.float32)


class MissingValueImputer(BaseEstimator, TransformerMixin):
    """
    Impute missing values and track imputation statistics.

    Numeric columns: median imputation.
    Categorical columns: most frequent (mode) imputation.
    """

    def __init__(self) -> None:
        self.imputation_values_: Dict[str, Any] = {}
        self.imputation_stats_: Dict[str, Any] = {}
        self.numeric_columns_: List[str] = []
        self.categorical_columns_: List[str] = []

    def fit(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
    ) -> "MissingValueImputer":
        """Learn imputation values from training data."""
        self.numeric_columns_ = []
        self.categorical_columns_ = []
        self.imputation_values_ = {}
        self.imputation_stats_ = {
            "numeric_strategy": NUMERIC_IMPUTATION_STRATEGY,
            "categorical_strategy": CATEGORICAL_IMPUTATION_STRATEGY,
            "columns": {},
        }

        protected = set(ID_COLUMNS) | {TARGET_COLUMN}

        for col in X.columns:
            if col in protected:
                continue

            missing_count = int(X[col].isnull().sum())
            missing_pct = round(missing_count / len(X) * 100, 2)

            if _is_categorical(X[col]):
                self.categorical_columns_.append(col)
                mode_vals = X[col].mode(dropna=True)
                fill_value = mode_vals.iloc[0] if len(mode_vals) > 0 else "MISSING"
                self.imputation_values_[col] = fill_value
                strategy = CATEGORICAL_IMPUTATION_STRATEGY
            else:
                self.numeric_columns_.append(col)
                median_val = X[col].median()
                fill_value = float(median_val) if pd.notna(median_val) else 0.0
                self.imputation_values_[col] = fill_value
                strategy = NUMERIC_IMPUTATION_STRATEGY

            self.imputation_stats_["columns"][col] = {
                "strategy": strategy,
                "missing_count_before": missing_count,
                "missing_pct_before": missing_pct,
                "imputed_value": self._serialise_value(fill_value),
            }

        total_imputed = sum(
            v["missing_count_before"]
            for v in self.imputation_stats_["columns"].values()
        )
        self.imputation_stats_["total_values_imputed"] = total_imputed
        logger.info(
            f"MissingValueImputer fit — {len(self.numeric_columns_)} numeric, "
            f"{len(self.categorical_columns_)} categorical columns; "
            f"{total_imputed:,} values to impute"
        )
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply learned imputation values."""
        X_out = X.copy()
        for col, value in self.imputation_values_.items():
            if col in X_out.columns:
                X_out[col] = X_out[col].fillna(value)
        return X_out

    @staticmethod
    def _serialise_value(value: Any) -> Any:
        """Convert imputation values to JSON-safe types."""
        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, (np.floating,)):
            return float(value)
        if pd.isna(value):
            return None
        return value

    def get_imputation_stats(self) -> Dict[str, Any]:
        """Return imputation statistics from the last ``fit()``."""
        return dict(self.imputation_stats_)


class CategoricalEncoder(BaseEstimator, TransformerMixin):
    """
    Encode categorical columns: label encoding for binary, one-hot for multi.

    Encoders are persisted as attributes for inference and metadata export.
    """

    def __init__(self, binary_max_unique: int = BINARY_CATEGORY_MAX_UNIQUE) -> None:
        self.binary_max_unique = binary_max_unique
        self.label_encoders_: Dict[str, LabelEncoder] = {}
        self.onehot_encoders_: Dict[str, OneHotEncoder] = {}
        self.encoding_map_: Dict[str, str] = {}
        self.feature_names_out_: List[str] = []
        self._passthrough_numeric_: List[str] = []

    def fit(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
    ) -> "CategoricalEncoder":
        """Identify encoding strategy per column and fit encoders."""
        self.label_encoders_ = {}
        self.onehot_encoders_ = {}
        self.encoding_map_ = {}
        self._passthrough_numeric_ = []
        protected = set(ID_COLUMNS) | {TARGET_COLUMN}

        cat_cols: List[str] = []
        for col in X.columns:
            if col in protected:
                continue
            if _is_categorical(X[col]):
                cat_cols.append(col)
            else:
                self._passthrough_numeric_.append(col)

        for col in cat_cols:
            n_unique = int(X[col].nunique(dropna=True))
            if n_unique <= self.binary_max_unique:
                le = LabelEncoder()
                le.fit(X[col].astype(str))
                self.label_encoders_[col] = le
                self.encoding_map_[col] = ENCODING_STRATEGY["binary"]
            else:
                ohe = OneHotEncoder(
                    sparse_output=False,
                    handle_unknown="ignore",
                    dtype=np.float32,
                )
                ohe.fit(X[[col]].astype(str))
                self.onehot_encoders_[col] = ohe
                self.encoding_map_[col] = ENCODING_STRATEGY["multi_category"]

        self.feature_names_out_ = self._build_feature_names()
        logger.info(
            f"CategoricalEncoder fit — {len(self.label_encoders_)} label-encoded, "
            f"{len(self.onehot_encoders_)} one-hot encoded columns"
        )
        return self

    def _build_feature_names(self) -> List[str]:
        """Build output column names after encoding."""
        names: List[str] = [c for c in ID_COLUMNS]

        for col in self._passthrough_numeric_:
            names.append(col)

        for col in self.label_encoders_:
            names.append(f"{col}_le")

        for col, ohe in self.onehot_encoders_.items():
            names.extend(list(ohe.get_feature_names_out([col])))

        names.append(TARGET_COLUMN)
        return names

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply encoders and return a numeric feature matrix."""
        parts: List[pd.DataFrame] = []

        for id_col in ID_COLUMNS:
            if id_col in X.columns:
                parts.append(X[[id_col]].reset_index(drop=True))

        if self._passthrough_numeric_:
            num_cols = [c for c in self._passthrough_numeric_ if c in X.columns]
            if num_cols:
                parts.append(X[num_cols].reset_index(drop=True).astype(np.float32))

        for col, le in self.label_encoders_.items():
            encoded = _safe_label_transform(le, X[col])
            parts.append(pd.DataFrame({f"{col}_le": encoded}))

        for col, ohe in self.onehot_encoders_.items():
            arr = ohe.transform(X[[col]].astype(str))
            cat_names = ohe.get_feature_names_out([col])
            parts.append(pd.DataFrame(arr, columns=list(cat_names)))

        if TARGET_COLUMN in X.columns:
            parts.append(X[[TARGET_COLUMN]].reset_index(drop=True))

        result = pd.concat(parts, axis=1)
        return result

    def get_encoding_info(self) -> Dict[str, Any]:
        """Return encoding strategy details for metadata export."""
        return {
            "strategy": dict(ENCODING_STRATEGY),
            "columns": dict(self.encoding_map_),
            "label_encoded_columns": list(self.label_encoders_.keys()),
            "one_hot_encoded_columns": list(self.onehot_encoders_.keys()),
            "unknown_label_value": LABEL_ENCODER_UNKNOWN_VALUE,
            "output_feature_count": len(self.feature_names_out_),
        }


class PreprocessingPipeline:
    """
    End-to-end preprocessing orchestrator for Home Credit Default Risk.

    Pipeline steps:
      1. Validate raw data
      2. Stratified train/test split
      3. Feature engineering
      4. Feature selection (fit on train)
      5. Missing value imputation (fit on train)
      6. Categorical encoding (fit on train)
      7. Persist artefacts and generate report

    Example::

        pipeline = PreprocessingPipeline()
        result = pipeline.run()
    """

    def __init__(
        self,
        raw_dir: Optional[Path] = None,
        processed_dir: Optional[Path] = None,
        models_dir: Optional[Path] = None,
        documents_dir: Optional[Path] = None,
    ) -> None:
        self.raw_dir = raw_dir or DATA_RAW_DIR
        self.processed_dir = processed_dir or DATA_PROCESSED_DIR
        self.models_dir = models_dir
        self.documents_dir = documents_dir
        self.validator = DataValidator()
        self.splitter = DataSplitter()
        self.artifact_manager = ArtifactManager(
            processed_dir=self.processed_dir,
            models_dir=models_dir,
            documents_dir=documents_dir,
        )

        self.sklearn_pipeline_: Optional[Pipeline] = None
        self.metadata_: Dict[str, Any] = {}

    def _build_sklearn_pipeline(self) -> Pipeline:
        """Construct the sklearn Pipeline of transformers."""
        self.feature_engineer_ = FeatureEngineer()
        self.feature_selector_ = FeatureSelector()
        self.imputer_ = MissingValueImputer()
        self.encoder_ = CategoricalEncoder()

        return Pipeline(
            steps=[
                ("feature_engineering", self.feature_engineer_),
                ("feature_selector", self.feature_selector_),
                ("imputer", self.imputer_),
                ("encoder", self.encoder_),
            ]
        )

    def _collect_metadata(
        self,
        validation_report: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Assemble metadata dict for JSON export."""
        dropped = (
            self.feature_selector_.dropped_high_missing_
            + self.feature_selector_.dropped_zero_variance_
        )
        return {
            "validation": validation_report,
            "split": self.splitter.get_split_report(),
            "created_features": self.feature_engineer_.get_feature_summary(),
            "dropped_columns": dropped,
            "feature_selection_report": self.feature_selector_.get_selection_report(),
            "imputation_stats": self.imputer_.get_imputation_stats(),
            "encoding_info": self.encoder_.get_encoding_info(),
            "missing_value_strategy": {
                "numeric": NUMERIC_IMPUTATION_STRATEGY,
                "categorical": CATEGORICAL_IMPUTATION_STRATEGY,
            },
            "output_columns": {
                "train": int(len(self.metadata_.get("train_columns", []))),
                "test": int(len(self.metadata_.get("test_columns", []))),
            },
        }

    @timer
    def run(
        self,
        input_path: Optional[Path] = None,
    ) -> Dict[str, Path]:
        """
        Execute the full preprocessing pipeline.

        Args:
            input_path: Override path to ``application_train.csv``.

        Returns:
            Dict with paths to all generated artefacts.
        """
        csv_path = input_path or (self.raw_dir / INPUT_FILENAME)
        logger.info(f"Loading training data from: {csv_path}")

        if not csv_path.exists():
            raise FileNotFoundError(
                f"Training file not found: {csv_path}\n"
                f"Place {INPUT_FILENAME} in {self.raw_dir}"
            )

        df = pd.read_csv(csv_path, low_memory=False)
        validation_report = self.validator.validate(df)

        train_df, test_df = self.splitter.split(df)

        self.sklearn_pipeline_ = self._build_sklearn_pipeline()
        train_processed = self.sklearn_pipeline_.fit_transform(train_df)
        test_processed = self.sklearn_pipeline_.transform(test_df)

        self.metadata_ = {
            "train_columns": list(train_processed.columns),
            "test_columns": list(test_processed.columns),
        }
        metadata = self._collect_metadata(validation_report)

        paths = self.artifact_manager.save_all(
            pipeline=self.sklearn_pipeline_,
            train_df=train_processed,
            test_df=test_processed,
            metadata=metadata,
        )

        logger.info("Preprocessing pipeline complete.")
        return paths


if __name__ == "__main__":
    from config import settings

    settings.ensure_directories()
    result = PreprocessingPipeline().run()
    print("\n✅  Preprocessing complete.")
    for name, path in result.items():
        print(f"    {name} → {path}")
