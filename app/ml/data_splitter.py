"""
app/ml/data_splitter.py — Stratified train/test splitting.

Preserves target class distribution in both splits.
"""

from __future__ import annotations

from typing import Dict, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from app.ml.settings import RANDOM_SEED, TARGET_COLUMN, TEST_SIZE
from app.utils.logger import get_logger

logger = get_logger(__name__)


class DataSplitter:
    """
    Perform a stratified train/test split on a validated DataFrame.

    Example::

        splitter = DataSplitter()
        train_df, test_df = splitter.split(df)
    """

    def __init__(
        self,
        test_size: float = TEST_SIZE,
        random_state: int = RANDOM_SEED,
    ) -> None:
        self.test_size = test_size
        self.random_state = random_state
        self.split_report_: Dict[str, float] = {}

    def split(
        self,
        df: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split *df* into stratified train and test sets.

        Args:
            df: DataFrame containing ``TARGET``.

        Returns:
            Tuple of (train_df, test_df).

        Raises:
            ValueError: When TARGET is missing or has insufficient class counts.
        """
        if TARGET_COLUMN not in df.columns:
            raise ValueError(f"'{TARGET_COLUMN}' column required for stratified split")

        y = df[TARGET_COLUMN]
        class_counts = y.value_counts()
        if (class_counts < 2).any():
            raise ValueError(
                "Stratified split requires at least 2 samples per class. "
                f"Class counts: {class_counts.to_dict()}"
            )

        train_df, test_df = train_test_split(
            df,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y,
        )

        train_df = train_df.reset_index(drop=True)
        test_df = test_df.reset_index(drop=True)

        train_rate = float(train_df[TARGET_COLUMN].mean() * 100)
        test_rate = float(test_df[TARGET_COLUMN].mean() * 100)

        self.split_report_ = {
            "train_rows": int(len(train_df)),
            "test_rows": int(len(test_df)),
            "train_default_rate_pct": round(train_rate, 4),
            "test_default_rate_pct": round(test_rate, 4),
            "test_size": self.test_size,
            "random_state": self.random_state,
        }

        logger.info(
            f"Stratified split — train: {len(train_df):,} rows "
            f"(default {train_rate:.2f}%), "
            f"test: {len(test_df):,} rows (default {test_rate:.2f}%)"
        )
        return train_df, test_df

    def get_split_report(self) -> Dict[str, float]:
        """Return metrics from the last ``split()`` call."""
        return dict(self.split_report_)
