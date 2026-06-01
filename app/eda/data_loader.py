"""
app/eda/data_loader.py — Production data loader for the Home Credit Default Risk dataset.

Responsibilities:
  - Load one or more CSV dataset files from ``data/raw/``.
  - Validate schema and target variable integrity for application_train.
  - Reduce memory footprint via dtype downcasting.
  - Provide a clean dict[str, DataFrame] interface to consumers.

Usage:
    from app.eda.data_loader import DataLoader

    loader = DataLoader()
    datasets = loader.load_all()
    app_df   = datasets["application_train"]
    bureau   = datasets.get("bureau")          # None if file absent
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from app.utils.helpers import reduce_memory_usage, timer
from app.utils.logger import get_logger
from config import settings

logger = get_logger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

_REQUIRED_APP_TRAIN_COLS: frozenset = frozenset({"SK_ID_CURR", "TARGET"})
_VALID_TARGET_VALUES: frozenset[int] = frozenset({0, 1})


# ─────────────────────────────────────────────────────────────────────────────
# DataLoader
# ─────────────────────────────────────────────────────────────────────────────


class DataLoader:
    """
    Loads and validates Home Credit Default Risk dataset files.

    Files are loaded from ``settings.data_raw_dir``.  The primary file
    (``application_train.csv``) is mandatory; supplementary files are
    optional — missing ones are silently skipped with a warning log.

    Attributes:
        raw_dir:  Directory that contains the CSV files.
        datasets: Dict populated after calling ``load_all()`` or individual
                  ``load_*`` methods.

    Example::

        loader = DataLoader()
        datasets = loader.load_all()
        df = loader.get_dataset("application_train")
    """

    def __init__(self) -> None:
        self.raw_dir: Path = settings.data_raw_dir
        self._datasets: Dict[str, pd.DataFrame] = {}
        logger.info(f"DataLoader initialised. Raw data directory: {self.raw_dir}")

    # ── Private helpers ───────────────────────────────────────────────────────

    def _load_csv(self, filename: str, required: bool = True) -> Optional[pd.DataFrame]:
        """
        Load a single CSV file with memory reduction.

        Args:
            filename: Filename (not full path) inside ``raw_dir``.
            required: Raise ``FileNotFoundError`` when True and file is absent.

        Returns:
            Loaded DataFrame, or ``None`` when optional file is absent.

        Raises:
            FileNotFoundError: When *required* is True and the file is missing.
            RuntimeError:      When pandas fails to parse the file.
        """
        filepath = self.raw_dir / filename

        if not filepath.exists():
            if required:
                raise FileNotFoundError(
                    f"Required dataset file not found: {filepath}\n"
                    f"Please download it from Kaggle and place it in: {self.raw_dir}"
                )
            logger.warning(f"Optional dataset file not found — skipping: {filepath}")
            return None

        logger.info(f"Loading '{filename}' ...")
        try:
            df = pd.read_csv(filepath, low_memory=False)
        except Exception as exc:
            raise RuntimeError(f"Failed to parse '{filename}': {exc}") from exc

        logger.info(f"Loaded '{filename}': {df.shape[0]:,} rows × {df.shape[1]} columns")
        df = reduce_memory_usage(df, verbose=True)
        return df

    # ── Validation ────────────────────────────────────────────────────────────

    def _validate_application_train(self, df: pd.DataFrame) -> None:
        """
        Run schema and data-quality checks on application_train.

        Raises:
            ValueError: On schema or target-variable issues.
        """
        missing_cols = _REQUIRED_APP_TRAIN_COLS - set(df.columns)
        if missing_cols:
            raise ValueError(
                f"application_train.csv is missing required column(s): {missing_cols}"
            )

        null_target_count = int(df["TARGET"].isnull().sum())
        if null_target_count:
            logger.warning(
                f"TARGET column has {null_target_count:,} null values "
                f"({null_target_count / len(df) * 100:.2f}% of rows)."
            )

        unexpected_targets = set(df["TARGET"].dropna().unique()) - _VALID_TARGET_VALUES
        if unexpected_targets:
            raise ValueError(
                f"TARGET column contains unexpected values: {unexpected_targets}. "
                f"Expected only {{0, 1}}."
            )

        id_duplicates = int(df["SK_ID_CURR"].duplicated().sum())
        if id_duplicates:
            logger.warning(
                f"SK_ID_CURR has {id_duplicates:,} duplicate values — "
                "expected one row per loan application."
            )

        target_dist = df["TARGET"].value_counts().sort_index().to_dict()
        default_rate = df["TARGET"].mean() * 100
        logger.info(
            f"Validation passed ✓  |  TARGET distribution: {target_dist}  |  "
            f"Default rate: {default_rate:.2f}%"
        )

    # ── Public loaders ────────────────────────────────────────────────────────

    def load_application_train(self) -> pd.DataFrame:
        """
        Load and validate ``application_train.csv``.

        Returns:
            DataFrame with all raw application features.

        Raises:
            FileNotFoundError: When the file is absent.
            ValueError:        When schema validation fails.
            RuntimeError:      On unexpected internal failure.
        """
        df = self._load_csv(settings.app_train_filename, required=True)
        if df is None:
            # Should never occur because required=True; guard against future refactors
            raise RuntimeError(
                f"_load_csv returned None for required file '{settings.app_train_filename}'. "
                "This is an internal error — please report it."
            )
        self._validate_application_train(df)
        self._datasets["application_train"] = df
        return df

    def load_bureau(self) -> Optional[pd.DataFrame]:
        """
        Load ``bureau.csv`` (optional).

        Returns:
            DataFrame or ``None`` when the file is absent.
        """
        df = self._load_csv(settings.bureau_filename, required=False)
        if df is not None:
            self._datasets["bureau"] = df
        return df

    def load_previous_application(self) -> Optional[pd.DataFrame]:
        """
        Load ``previous_application.csv`` (optional).

        Returns:
            DataFrame or ``None`` when the file is absent.
        """
        df = self._load_csv(settings.previous_app_filename, required=False)
        if df is not None:
            self._datasets["previous_application"] = df
        return df

    @timer
    def load_all(self) -> Dict[str, pd.DataFrame]:
        """
        Load all available datasets in one call.

        Loads application_train (required), bureau, and previous_application
        (both optional).

        Returns:
            Dict mapping dataset name → DataFrame for every file that was found.
        """
        logger.info("Starting full data-loading pipeline ...")

        self.load_application_train()
        self.load_bureau()
        self.load_previous_application()

        logger.info(
            f"Data loading complete.  "
            f"Datasets in memory: {list(self._datasets.keys())}"
        )
        return self._datasets

    # ── Accessors ─────────────────────────────────────────────────────────────

    @property
    def datasets(self) -> Dict[str, pd.DataFrame]:
        """All loaded DataFrames, keyed by dataset name."""
        return self._datasets

    def get_dataset(self, name: str) -> Optional[pd.DataFrame]:
        """Return a single dataset by name, or ``None`` if not loaded."""
        return self._datasets.get(name)

    def get_loaded_names(self) -> list:
        """Return names of all datasets currently in memory."""
        return list(self._datasets.keys())

    def __repr__(self) -> str:  # pragma: no cover
        loaded = self.get_loaded_names()
        return f"DataLoader(raw_dir={self.raw_dir!r}, loaded={loaded})"
