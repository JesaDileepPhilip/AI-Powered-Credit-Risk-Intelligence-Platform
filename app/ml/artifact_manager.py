"""
app/ml/artifact_manager.py — Persist preprocessing artefacts and generate reports.

Saves:
  - data/processed/train.csv
  - data/processed/test.csv
  - models/preprocessing_pipeline.pkl
  - models/feature_metadata.json
  - documents/feature_engineering_report.md
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline

from app.ml.settings import (
    DATA_PROCESSED_DIR,
    DOCUMENTS_DIR,
    ENCODING_STRATEGY,
    FEATURE_METADATA_FILENAME,
    FE_REPORT_FILENAME,
    MODELS_DIR,
    PIPELINE_FILENAME,
    TEST_FILENAME,
    TRAIN_FILENAME,
)
from app.utils.helpers import make_json_serialisable
from app.utils.logger import get_logger
from config import settings

logger = get_logger(__name__)


class ArtifactManager:
    """
    Save and load preprocessing pipeline artefacts.

    Example::

        manager = ArtifactManager()
        paths = manager.save_all(pipeline, train_df, test_df, metadata)
    """

    def __init__(
        self,
        processed_dir: Optional[Path] = None,
        models_dir: Optional[Path] = None,
        documents_dir: Optional[Path] = None,
    ) -> None:
        self.processed_dir = processed_dir or DATA_PROCESSED_DIR
        self.models_dir = models_dir or MODELS_DIR
        self.documents_dir = documents_dir or DOCUMENTS_DIR

        for d in (self.processed_dir, self.models_dir, self.documents_dir):
            d.mkdir(parents=True, exist_ok=True)

    def save_train_test(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
    ) -> Dict[str, Path]:
        """Write processed train and test CSV files."""
        train_path = self.processed_dir / TRAIN_FILENAME
        test_path = self.processed_dir / TEST_FILENAME

        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)

        logger.info(f"Saved train set: {train_path} ({len(train_df):,} rows)")
        logger.info(f"Saved test set:  {test_path} ({len(test_df):,} rows)")

        return {"train_csv": train_path, "test_csv": test_path}

    def save_pipeline(self, pipeline: Pipeline) -> Path:
        """Serialise the sklearn preprocessing pipeline."""
        path = self.models_dir / PIPELINE_FILENAME
        joblib.dump(pipeline, path)
        logger.info(f"Saved preprocessing pipeline: {path}")
        return path

    def save_metadata(self, metadata: Dict[str, Any]) -> Path:
        """Write feature metadata JSON."""
        path = self.models_dir / FEATURE_METADATA_FILENAME
        serialisable = make_json_serialisable(metadata)

        with open(path, "w", encoding="utf-8") as fh:
            json.dump(serialisable, fh, indent=2, default=str)

        logger.info(f"Saved feature metadata: {path}")
        return path

    def generate_report(self, metadata: Dict[str, Any]) -> Path:
        """
        Write the feature engineering Markdown report.

        Includes dropped columns, created features, encoding and imputation strategy.
        """
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        selection = metadata.get("feature_selection_report", {})
        created = metadata.get("created_features", {})
        imputation = metadata.get("imputation_stats", {})
        encoding = metadata.get("encoding_info", {})
        missing_strategy = metadata.get("missing_value_strategy", {})
        split_info = metadata.get("split", {})

        dropped_high = selection.get("dropped_high_missing", [])
        dropped_zero = selection.get("dropped_zero_variance", [])

        def _fmt_int(value: Any) -> str:
            return f"{value:,}" if isinstance(value, int) else str(value)

        lines = [
            f"# Feature Engineering Report — {settings.app_name}",
            f"\n**Generated:** {ts}  ",
            f"**Platform Version:** {settings.app_version}",

            "\n---\n",
            "## 1. Executive Summary",
            f"\n| Metric | Value |",
            "| --- | --- |",
            f"| Train rows | {_fmt_int(split_info.get('train_rows', 'N/A'))} |",
            f"| Test rows | {_fmt_int(split_info.get('test_rows', 'N/A'))} |",
            f"| Train default rate | {split_info.get('train_default_rate_pct', 'N/A')}% |",
            f"| Test default rate | {split_info.get('test_default_rate_pct', 'N/A')}% |",
            f"| Features created | {len(created.get('created_features', []))} |",
            f"| Columns dropped (high missing) | {len(dropped_high)} |",
            f"| Columns dropped (zero variance) | {len(dropped_zero)} |",
            f"| Total values imputed | {_fmt_int(imputation.get('total_values_imputed', 'N/A'))} |",

            "\n---\n",
            "## 2. Created Features",
        ]

        for feat in created.get("created_features", []):
            lines.append(f"- `{feat}`")

        skipped = created.get("skipped_features", {})
        if skipped:
            lines.append("\n**Skipped (missing source columns):**")
            for feat, reason in skipped.items():
                lines.append(f"- `{feat}`: {reason}")

        lines += [
            "\n---\n",
            "## 3. Dropped Columns",
            "\n### 3.1 High Missing (>80%)",
        ]

        if dropped_high:
            lines += ["| Column | Missing % |", "| --- | --- |"]
            for item in dropped_high:
                lines.append(f"| `{item['column']}` | {item['missing_pct']:.2f}% |")
        else:
            lines.append("_None._")

        lines += ["\n### 3.2 Zero Variance"]
        if dropped_zero:
            for col in dropped_zero:
                lines.append(f"- `{col}`")
        else:
            lines.append("_None._")

        lines += [
            "\n---\n",
            "## 4. Missing Value Strategy",
            f"\n- **Numeric:** {missing_strategy.get('numeric', 'median')} imputation",
            f"- **Categorical:** {missing_strategy.get('categorical', 'most_frequent')} imputation",
            f"- **Total values imputed:** {imputation.get('total_values_imputed', 0):,}",
        ]

        imputed_cols = imputation.get("columns", {})
        if imputed_cols:
            top_imputed = sorted(
                imputed_cols.items(),
                key=lambda x: x[1].get("missing_count_before", 0),
                reverse=True,
            )[:15]
            lines += [
                "\n### Top Imputed Columns",
                "| Column | Strategy | Missing Count | Imputed Value |",
                "| --- | --- | --- | --- |",
            ]
            for col, info in top_imputed:
                lines.append(
                    f"| `{col}` | {info.get('strategy', '')} | "
                    f"{info.get('missing_count_before', 0):,} | "
                    f"{info.get('imputed_value', '')} |"
                )

        lines += [
            "\n---\n",
            "## 5. Encoding Strategy",
            f"\n- **Binary categorical (≤2 unique):** {ENCODING_STRATEGY['binary']}",
            f"- **Multi-category (>2 unique):** {ENCODING_STRATEGY['multi_category']}",
            "\n### Label-Encoded Columns",
        ]

        for col in encoding.get("label_encoded_columns", []):
            lines.append(f"- `{col}`")

        lines += ["\n### One-Hot Encoded Columns"]
        for col in encoding.get("one_hot_encoded_columns", []):
            lines.append(f"- `{col}`")

        lines += [
            "\n---\n",
            f"*Report generated by {settings.app_name} v{settings.app_version} on {ts}*",
        ]

        report_path = self.documents_dir / FE_REPORT_FILENAME
        with open(report_path, "w", encoding="utf-8") as fh:
            fh.write("\n".join(lines))

        logger.info(f"Saved feature engineering report: {report_path}")
        return report_path

    def save_all(
        self,
        pipeline: Pipeline,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        metadata: Dict[str, Any],
    ) -> Dict[str, Path]:
        """
        Persist all preprocessing artefacts in one call.

        Returns:
            Dict mapping artefact name → file path.
        """
        paths: Dict[str, Path] = {}

        paths.update(self.save_train_test(train_df, test_df))
        paths["pipeline_pkl"] = self.save_pipeline(pipeline)
        paths["metadata_json"] = self.save_metadata(metadata)
        paths["report_md"] = self.generate_report(metadata)

        return paths

    @staticmethod
    def load_pipeline(path: Optional[Path] = None) -> Pipeline:
        """Load a saved preprocessing pipeline."""
        load_path = path or (MODELS_DIR / PIPELINE_FILENAME)
        return joblib.load(load_path)

    @staticmethod
    def load_metadata(path: Optional[Path] = None) -> Dict[str, Any]:
        """Load feature metadata JSON."""
        load_path = path or (MODELS_DIR / FEATURE_METADATA_FILENAME)
        with open(load_path, encoding="utf-8") as fh:
            return json.load(fh)
