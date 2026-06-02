"""
app/ml/data_validator.py — Schema, dtype, and target validation for training data.

Validates ``application_train.csv`` (or equivalent) before preprocessing.
"""

from __future__ import annotations

from typing import Any, Dict, List, Set

import pandas as pd

from app.ml.settings import (
    REQUIRED_COLUMNS,
    TARGET_COLUMN,
    VALID_TARGET_VALUES,
)
from app.utils.logger import get_logger

logger = get_logger(__name__)


class DataValidator:
    """
    Validates raw training DataFrames before feature engineering.

    Checks:
      - Required columns present
      - TARGET is numeric with values in {0, 1}
      - No null TARGET values
      - Basic dtype sanity for key columns

    Example::

        validator = DataValidator()
        report = validator.validate(df)
    """

    def __init__(self) -> None:
        self._validation_report: Dict[str, Any] = {}

    def validate(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Run all validation checks.

        Args:
            df: Raw application training DataFrame.

        Returns:
            Validation report dict with check results.

        Raises:
            ValueError: When any validation check fails.
        """
        logger.info(f"Validating dataset: {df.shape[0]:,} rows × {df.shape[1]} columns")

        self._validation_report = {
            "rows": int(len(df)),
            "columns": int(len(df.columns)),
            "checks_passed": [],
        }

        self._validate_schema(df)
        self._validate_target(df)
        self._validate_dtypes(df)

        logger.info(
            f"Validation passed ✓  ({len(self._validation_report['checks_passed'])} checks)"
        )
        return self._validation_report

    def _validate_schema(self, df: pd.DataFrame) -> None:
        """Ensure all required columns are present."""
        present = set(df.columns)
        missing = REQUIRED_COLUMNS - present
        if missing:
            raise ValueError(
                f"Schema validation failed — missing required column(s): {sorted(missing)}"
            )
        self._validation_report["checks_passed"].append("schema")

    def _validate_target(self, df: pd.DataFrame) -> None:
        """Validate TARGET column values and completeness."""
        target = df[TARGET_COLUMN]

        null_count = int(target.isnull().sum())
        if null_count:
            raise ValueError(
                f"Target validation failed — {null_count:,} null values in '{TARGET_COLUMN}'"
            )

        if not pd.api.types.is_numeric_dtype(target):
            raise ValueError(
                f"Target validation failed — '{TARGET_COLUMN}' must be numeric, "
                f"got {target.dtype}"
            )

        unique_vals: Set[Any] = set(target.unique())
        unexpected = unique_vals - VALID_TARGET_VALUES
        if unexpected:
            raise ValueError(
                f"Target validation failed — unexpected values {unexpected}. "
                f"Expected only {VALID_TARGET_VALUES}"
            )

        dist = target.value_counts().sort_index().to_dict()
        self._validation_report["target_distribution"] = {
            int(k): int(v) for k, v in dist.items()
        }
        self._validation_report["default_rate_pct"] = round(float(target.mean() * 100), 4)
        self._validation_report["checks_passed"].append("target")

    def _validate_dtypes(self, df: pd.DataFrame) -> None:
        """Flag columns with unsupported or inconsistent dtypes."""
        issues: List[str] = []

        for col in df.columns:
            if col == TARGET_COLUMN:
                continue
            dtype = df[col].dtype
            if pd.api.types.is_object_dtype(dtype):
                # Object columns are acceptable (categorical source data)
                continue
            if pd.api.types.is_numeric_dtype(dtype):
                continue
            if isinstance(dtype, pd.CategoricalDtype):
                continue
            if pd.api.types.is_bool_dtype(dtype):
                continue
            issues.append(f"{col}: {dtype}")

        if issues:
            logger.warning(
                f"Unusual dtypes detected in {len(issues)} column(s): {issues[:5]}"
            )

        self._validation_report["dtype_summary"] = {
            "numeric": int(
                sum(1 for c in df.columns if pd.api.types.is_numeric_dtype(df[c]))
            ),
            "object": int(
                sum(1 for c in df.columns if pd.api.types.is_object_dtype(df[c]))
            ),
            "other": int(len(df.columns) - sum(
                1 for c in df.columns
                if pd.api.types.is_numeric_dtype(df[c])
                or pd.api.types.is_object_dtype(df[c])
            )),
        }
        self._validation_report["checks_passed"].append("dtypes")

    @property
    def validation_report(self) -> Dict[str, Any]:
        """Report from the last ``validate()`` call."""
        return dict(self._validation_report)
