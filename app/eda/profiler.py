"""
app/eda/profiler.py — Comprehensive statistical profiler for tabular datasets.

Produces:
  - Dataset dimensions
  - Column type breakdown
  - Missing value analysis
  - Duplicate analysis
  - Target variable distribution
  - Numeric summary statistics (mean, std, skewness, kurtosis, …)
  - Categorical summary statistics
  - Domain-specific feature categorisation
  - Point-biserial correlation with TARGET for numeric features

Usage:
    from app.eda.profiler import DataProfiler

    profiler = DataProfiler(df, dataset_name="application_train")
    profile  = profiler.run()
    missing  = profiler.get_missing_value_stats()
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats

from app.utils.logger import get_logger
from config import settings

logger = get_logger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Domain Feature Categorisation (Home Credit Default Risk)
# ─────────────────────────────────────────────────────────────────────────────

FEATURE_CATEGORIES: Dict[str, List[str]] = {
    "demographic": [
        "CODE_GENDER",
        "FLAG_OWN_CAR",
        "FLAG_OWN_REALTY",
        "CNT_CHILDREN",
        "CNT_FAM_MEMBERS",
        "NAME_FAMILY_STATUS",
        "NAME_HOUSING_TYPE",
        "DAYS_BIRTH",
        "REGION_POPULATION_RELATIVE",
        "REGION_RATING_CLIENT",
        "REGION_RATING_CLIENT_W_CITY",
        "REG_REGION_NOT_LIVE_REGION",
        "REG_REGION_NOT_WORK_REGION",
        "LIVE_REGION_NOT_WORK_REGION",
        "REG_CITY_NOT_LIVE_CITY",
        "REG_CITY_NOT_WORK_CITY",
        "LIVE_CITY_NOT_WORK_CITY",
        "ORGANIZATION_TYPE",
        "FONDKAPREMONT_MODE",
        "HOUSETYPE_MODE",
        "WALLSMATERIAL_MODE",
        "EMERGENCYSTATE_MODE",
    ],
    "financial": [
        "AMT_INCOME_TOTAL",
        "AMT_CREDIT",
        "AMT_ANNUITY",
        "AMT_GOODS_PRICE",
        "NAME_CONTRACT_TYPE",
        "NAME_INCOME_TYPE",
        "NAME_EDUCATION_TYPE",
        "NAME_TYPE_SUITE",
        "OCCUPATION_TYPE",
        "DAYS_EMPLOYED",
        "DAYS_REGISTRATION",
        "DAYS_ID_PUBLISH",
        "OWN_CAR_AGE",
        "DAYS_LAST_PHONE_CHANGE",
    ],
    "credit_history": [
        "EXT_SOURCE_1",
        "EXT_SOURCE_2",
        "EXT_SOURCE_3",
        "FLAG_DOCUMENT_2",
        "FLAG_DOCUMENT_3",
        "FLAG_DOCUMENT_4",
        "FLAG_DOCUMENT_5",
        "FLAG_DOCUMENT_6",
        "FLAG_DOCUMENT_7",
        "FLAG_DOCUMENT_8",
        "FLAG_DOCUMENT_9",
        "FLAG_DOCUMENT_10",
        "FLAG_DOCUMENT_11",
        "FLAG_DOCUMENT_12",
        "FLAG_DOCUMENT_13",
        "FLAG_DOCUMENT_14",
        "FLAG_DOCUMENT_15",
        "FLAG_DOCUMENT_16",
        "FLAG_DOCUMENT_17",
        "FLAG_DOCUMENT_18",
        "FLAG_DOCUMENT_19",
        "FLAG_DOCUMENT_20",
        "FLAG_DOCUMENT_21",
    ],
    "behavioural": [
        "DEF_30_CNT_SOCIAL_CIRCLE",
        "DEF_60_CNT_SOCIAL_CIRCLE",
        "OBS_30_CNT_SOCIAL_CIRCLE",
        "OBS_60_CNT_SOCIAL_CIRCLE",
        "AMT_REQ_CREDIT_BUREAU_HOUR",
        "AMT_REQ_CREDIT_BUREAU_DAY",
        "AMT_REQ_CREDIT_BUREAU_WEEK",
        "AMT_REQ_CREDIT_BUREAU_MON",
        "AMT_REQ_CREDIT_BUREAU_QRT",
        "AMT_REQ_CREDIT_BUREAU_YEAR",
        "FLAG_MOBIL",
        "FLAG_EMP_PHONE",
        "FLAG_WORK_PHONE",
        "FLAG_CONT_MOBILE",
        "FLAG_PHONE",
        "FLAG_EMAIL",
    ],
}

# Columns that are identifiers / targets — excluded from categorisation
_META_COLS: frozenset = frozenset({"SK_ID_CURR", "TARGET", "SK_ID_BUREAU", "SK_ID_PREV"})


# ─────────────────────────────────────────────────────────────────────────────
# DataProfiler
# ─────────────────────────────────────────────────────────────────────────────


class DataProfiler:
    """
    Generates a comprehensive statistical profile of a tabular dataset.

    Args:
        df:           DataFrame to profile.
        dataset_name: Human-readable name used in log messages and reports.

    Example::

        profiler = DataProfiler(df, "application_train")
        profile  = profiler.run()
        print(profile["dimensions"])
    """

    def __init__(self, df: pd.DataFrame, dataset_name: str = "dataset") -> None:
        self.df = df
        self.dataset_name = dataset_name
        self._profile: Optional[Dict[str, Any]] = None
        logger.info(
            f"DataProfiler initialised for '{dataset_name}' "
            f"({df.shape[0]:,} rows × {df.shape[1]} columns)"
        )

    # ── Dimensions ────────────────────────────────────────────────────────────

    def get_dimensions(self) -> Dict[str, int]:
        """Return row count, column count, and total cell count."""
        return {
            "rows": int(self.df.shape[0]),
            "columns": int(self.df.shape[1]),
            "total_cells": int(self.df.shape[0] * self.df.shape[1]),
        }

    # ── Column Types ──────────────────────────────────────────────────────────

    def get_column_types(self) -> Dict[str, Dict[str, Any]]:
        """
        Return a mapping of column name → type metadata.

        Each value dict has keys:
          ``dtype``, ``kind`` (integer | float | boolean | categorical | string),
          ``unique_count``, ``sample_values``.
        """
        type_info: Dict[str, Dict[str, Any]] = {}

        for col in self.df.columns:
            dtype = self.df[col].dtype

            if pd.api.types.is_bool_dtype(dtype):
                kind = "boolean"
            elif pd.api.types.is_integer_dtype(dtype):
                kind = "integer"
            elif pd.api.types.is_float_dtype(dtype):
                kind = "float"
            elif isinstance(dtype, pd.CategoricalDtype):
                kind = "categorical"
            else:
                kind = "string"

            type_info[col] = {
                "dtype": str(dtype),
                "kind": kind,
                "unique_count": int(self.df[col].nunique(dropna=True)),
                "sample_values": self.df[col].dropna().head(3).tolist(),
            }

        return type_info

    # ── Missing Values ────────────────────────────────────────────────────────

    def get_missing_value_stats(self) -> pd.DataFrame:
        """
        Return a DataFrame of columns that have at least one missing value.

        Columns:
            ``column``, ``missing_count``, ``missing_pct`` (sorted descending).
        """
        total = len(self.df)
        missing_count = self.df.isnull().sum()
        missing_pct = (missing_count / total * 100).round(2)

        result = (
            pd.DataFrame(
                {
                    "column": missing_count.index,
                    "missing_count": missing_count.values,
                    "missing_pct": missing_pct.values,
                }
            )
            .query("missing_count > 0")
            .sort_values("missing_pct", ascending=False)
            .reset_index(drop=True)
        )

        logger.info(
            f"Missing value analysis: {len(result)} / {self.df.shape[1]} columns "
            f"have at least one missing value."
        )
        return result

    # ── Duplicates ────────────────────────────────────────────────────────────

    def get_duplicate_stats(self, subset: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Return duplicate-row statistics, optionally checking on a column subset.

        Also checks SK_ID_CURR uniqueness when present.
        """
        n_duplicates = int(self.df.duplicated(subset=subset).sum())
        n_rows = int(len(self.df))

        result: Dict[str, Any] = {
            "total_rows": n_rows,
            "duplicate_rows": n_duplicates,
            "duplicate_pct": round(n_duplicates / n_rows * 100, 4),
            "is_unique": n_duplicates == 0,
        }

        if "SK_ID_CURR" in self.df.columns:
            id_dups = int(self.df["SK_ID_CURR"].duplicated().sum())
            result["SK_ID_CURR_duplicates"] = id_dups

        return result

    # ── Target Distribution ───────────────────────────────────────────────────

    def get_target_distribution(self) -> Optional[Dict[str, Any]]:
        """
        Return TARGET class distribution stats.

        Returns ``None`` when the TARGET column is absent.
        """
        if "TARGET" not in self.df.columns:
            logger.warning("'TARGET' column not found — skipping target distribution.")
            return None

        target = self.df["TARGET"].dropna()
        value_counts = target.value_counts().sort_index()
        value_pcts = (target.value_counts(normalize=True).sort_index() * 100).round(2)

        return {
            "counts": {int(k): int(v) for k, v in value_counts.items()},
            "percentages": {int(k): float(v) for k, v in value_pcts.items()},
            "default_rate": round(float(target.mean()) * 100, 4),
            "imbalance_ratio": round(
                float((target == 0).sum() / max((target == 1).sum(), 1)), 2
            ),
            "total_valid": int(len(target)),
            "null_count": int(self.df["TARGET"].isnull().sum()),
        }

    # ── Numeric Summary ───────────────────────────────────────────────────────

    def get_numeric_summary(self) -> pd.DataFrame:
        """
        Return descriptive stats for all numeric columns.

        Extends pandas ``describe()`` with skewness, kurtosis,
        missing percentage, and unique count.
        """
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()

        if not numeric_cols:
            return pd.DataFrame()

        summary = self.df[numeric_cols].describe().T
        summary["skewness"] = self.df[numeric_cols].skew().round(4)
        summary["kurtosis"] = self.df[numeric_cols].kurtosis().round(4)
        summary["missing_pct"] = (
            self.df[numeric_cols].isnull().sum() / len(self.df) * 100
        ).round(2)
        summary["unique_count"] = self.df[numeric_cols].nunique()

        return summary.round(4)

    # ── Categorical Summary ───────────────────────────────────────────────────

    def get_categorical_summary(self) -> pd.DataFrame:
        """
        Return a summary table for all string / object / categorical columns.

        Columns:
            ``column``, ``unique_values``, ``missing_pct``,
            ``top_value``, ``top_value_count``, ``top_value_pct``.
        """
        cat_cols = self.df.select_dtypes(include=["object", "string", "category"]).columns.tolist()

        if not cat_cols:
            return pd.DataFrame()

        rows = []
        for col in cat_cols:
            vc = self.df[col].value_counts(dropna=True)
            top_val = vc.index[0] if len(vc) > 0 else None
            top_cnt = int(vc.iloc[0]) if len(vc) > 0 else 0

            rows.append(
                {
                    "column": col,
                    "unique_values": int(self.df[col].nunique(dropna=True)),
                    "missing_pct": round(
                        self.df[col].isnull().sum() / len(self.df) * 100, 2
                    ),
                    "top_value": top_val,
                    "top_value_count": top_cnt,
                    # Use non-null count as denominator for a meaningful percentage
                    "top_value_pct": round(
                        top_cnt / max(self.df[col].notna().sum(), 1) * 100, 2
                    ),
                }
            )

        return pd.DataFrame(rows)

    # ── Feature Categorisation ────────────────────────────────────────────────

    def categorize_features(self) -> Dict[str, List[str]]:
        """
        Map dataset columns into domain-specific feature categories.

        Categories: demographic, financial, credit_history, behavioural, other.
        Columns not matched to any predefined category land in 'other'.

        Returns:
            Dict of category name → list of column names present in the DataFrame.
        """
        available = set(self.df.columns)
        result: Dict[str, List[str]] = {}
        categorised: set = set()

        for category, feature_list in FEATURE_CATEGORIES.items():
            matched = [f for f in feature_list if f in available]
            result[category] = matched
            categorised.update(matched)

        # Remaining columns (excluding meta columns) go to "other"
        result["other"] = [
            c for c in self.df.columns
            if c not in categorised and c not in _META_COLS
        ]

        for cat, cols in result.items():
            logger.info(f"  Feature category '{cat}': {len(cols)} features")

        return result

    # ── Target Correlations ───────────────────────────────────────────────────

    def get_correlation_with_target(self, top_n: int = 20) -> Optional[pd.DataFrame]:
        """
        Compute point-biserial correlation of each numeric feature with TARGET.

        Args:
            top_n: Number of top features (by |correlation|) to return.

        Returns:
            DataFrame with columns ``feature``, ``correlation``,
            ``abs_correlation``, ``p_value``, ``significant``,
            sorted by ``abs_correlation`` descending.
            Returns ``None`` when TARGET is absent.
        """
        if "TARGET" not in self.df.columns:
            return None

        numeric_cols = [
            c for c in self.df.select_dtypes(include=[np.number]).columns
            if c not in _META_COLS
        ]

        # Only use rows where TARGET is valid — do NOT impute TARGET
        target_series = self.df["TARGET"].dropna().astype(int)
        rows = []

        for col in numeric_cols:
            # Skip columns that are almost entirely null (≥99% missing)
            null_pct = self.df[col].isnull().mean()
            if null_pct >= 0.99:
                logger.debug(f"Skipping correlation for '{col}': {null_pct:.0%} null")
                continue

            # Align feature to the same index as the valid TARGET rows,
            # then fill remaining feature nulls with the column median
            col_series = self.df.loc[target_series.index, col]
            col_median = col_series.median()
            col_data = col_series.fillna(col_median if pd.notna(col_median) else 0)

            try:
                corr_stat = stats.pointbiserialr(target_series, col_data)
                rows.append(
                    {
                        "feature": col,
                        "correlation": round(float(corr_stat.statistic), 4),
                        "abs_correlation": round(abs(float(corr_stat.statistic)), 4),
                        "p_value": round(float(corr_stat.pvalue), 6),
                        "significant": bool(corr_stat.pvalue < 0.05),
                    }
                )
            except Exception as exc:
                logger.debug(f"Skipping correlation for '{col}': {exc}")

        if not rows:
            return None

        result_df = (
            pd.DataFrame(rows)
            .dropna(subset=["abs_correlation"])  # drop NaN from constant-column inputs
            .sort_values("abs_correlation", ascending=False)
            .head(top_n)
            .reset_index(drop=True)
        )
        return result_df

    # ── Full Pipeline ─────────────────────────────────────────────────────────

    def run(self) -> Dict[str, Any]:
        """
        Execute the complete profiling pipeline.

        Runs all ``get_*`` methods and assembles the results into a single
        structured dictionary suitable for JSON serialisation or report generation.

        Returns:
            Full profile dict.
        """
        logger.info(f"Running full profile for '{self.dataset_name}' ...")

        target_corr_df = self.get_correlation_with_target(
            top_n=settings.eda_correlation_top_n
        )

        self._profile = {
            "dataset_name": self.dataset_name,
            "dimensions": self.get_dimensions(),
            "column_types": self.get_column_types(),
            "missing_value_stats": self.get_missing_value_stats().to_dict(orient="records"),
            "duplicate_stats": self.get_duplicate_stats(),
            "target_distribution": self.get_target_distribution(),
            "numeric_summary": self.get_numeric_summary().to_dict(orient="index"),
            "categorical_summary": self.get_categorical_summary().to_dict(orient="records"),
            "feature_categories": self.categorize_features(),
            "target_correlations": (
                target_corr_df.to_dict(orient="records")
                if target_corr_df is not None
                else None
            ),
        }

        logger.info(f"Profiling complete for '{self.dataset_name}'.")
        return self._profile

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def profile(self) -> Optional[Dict[str, Any]]:
        """Cached profile dict from the last ``run()`` call, or ``None``."""
        return self._profile

    def __repr__(self) -> str:  # pragma: no cover
        status = "profiled" if self._profile else "not yet profiled"
        return (
            f"DataProfiler(dataset='{self.dataset_name}', "
            f"shape={self.df.shape}, status='{status}')"
        )
