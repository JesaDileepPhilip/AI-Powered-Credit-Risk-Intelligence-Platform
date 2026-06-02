"""
tests/test_preprocessing.py — Unit tests for Phase 2 preprocessing pipeline.

Covers validation, imputation, encoding, splitting, and end-to-end orchestration.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from app.ml.data_splitter import DataSplitter
from app.ml.data_validator import DataValidator
from app.ml.preprocessing import CategoricalEncoder, MissingValueImputer, PreprocessingPipeline


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def valid_training_df() -> pd.DataFrame:
    """Minimal valid training DataFrame with mixed column types."""
    n = 200
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {
            "SK_ID_CURR": range(100_001, 100_001 + n),
            "TARGET": rng.choice([0, 1], size=n, p=[0.92, 0.08]),
            "AMT_INCOME_TOTAL": rng.uniform(40_000, 400_000, n),
            "AMT_CREDIT": rng.uniform(50_000, 500_000, n),
            "AMT_ANNUITY": rng.uniform(5_000, 50_000, n),
            "DAYS_BIRTH": rng.integers(-25_000, -7_000, n),
            "DAYS_EMPLOYED": rng.choice(
                list(range(-4_000, -100)) + [365_243], size=n
            ),
            "CODE_GENDER": rng.choice(["M", "F"], size=n),
            "NAME_CONTRACT_TYPE": rng.choice(
                ["Cash loans", "Revolving loans", "Unknown"], size=n
            ),
            "FLAG_OWN_CAR": rng.choice(["Y", "N"], size=n),
            "MOSTLY_NULL": np.where(rng.random(n) > 0.05, np.nan, 1.0),
            "CONSTANT_COL": [7.0] * n,
        }
    )


@pytest.fixture()
def raw_csv(tmp_path, valid_training_df):
    """Write valid training CSV to a temp raw directory."""
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    csv_path = raw_dir / "application_train.csv"
    valid_training_df.to_csv(csv_path, index=False)
    return raw_dir, csv_path


# ---------------------------------------------------------------------------
# DataValidator
# ---------------------------------------------------------------------------


class TestDataValidator:
    def test_valid_dataframe_passes(self, valid_training_df):
        report = DataValidator().validate(valid_training_df)
        assert "schema" in report["checks_passed"]
        assert "target" in report["checks_passed"]

    def test_missing_target_raises(self, valid_training_df):
        df = valid_training_df.drop(columns=["TARGET"])
        with pytest.raises(ValueError, match="Schema validation failed"):
            DataValidator().validate(df)

    def test_invalid_target_value_raises(self, valid_training_df):
        df = valid_training_df.copy()
        df.loc[0, "TARGET"] = 2
        with pytest.raises(ValueError, match="unexpected values"):
            DataValidator().validate(df)

    def test_null_target_raises(self, valid_training_df):
        df = valid_training_df.copy()
        df.loc[0, "TARGET"] = np.nan
        with pytest.raises(ValueError, match="null values"):
            DataValidator().validate(df)


# ---------------------------------------------------------------------------
# DataSplitter
# ---------------------------------------------------------------------------


class TestDataSplitter:
    def test_stratified_split_preserves_rows(self, valid_training_df):
        train, test = DataSplitter(test_size=0.2, random_state=42).split(
            valid_training_df
        )
        assert len(train) + len(test) == len(valid_training_df)

    def test_default_rates_are_similar(self, valid_training_df):
        splitter = DataSplitter(test_size=0.2, random_state=42)
        train, test = splitter.split(valid_training_df)
        train_rate = train["TARGET"].mean()
        test_rate = test["TARGET"].mean()
        assert abs(train_rate - test_rate) < 0.05

    def test_split_report_populated(self, valid_training_df):
        splitter = DataSplitter(test_size=0.2, random_state=42)
        splitter.split(valid_training_df)
        report = splitter.get_split_report()
        assert report["train_rows"] > 0
        assert report["test_rows"] > 0


# ---------------------------------------------------------------------------
# MissingValueImputer
# ---------------------------------------------------------------------------


class TestMissingValueImputer:
    def test_imputes_numeric_with_median(self, valid_training_df):
        df = valid_training_df.copy()
        df.loc[0:5, "AMT_CREDIT"] = np.nan
        imputer = MissingValueImputer()
        result = imputer.fit_transform(df)
        assert result["AMT_CREDIT"].isnull().sum() == 0
        assert "AMT_CREDIT" in imputer.imputation_stats_["columns"]

    def test_imputes_categorical_with_mode(self, valid_training_df):
        df = valid_training_df.copy()
        df.loc[0:3, "CODE_GENDER"] = np.nan
        imputer = MissingValueImputer()
        result = imputer.fit_transform(df)
        assert result["CODE_GENDER"].isnull().sum() == 0

    def test_imputation_stats_tracked(self, valid_training_df):
        imputer = MissingValueImputer()
        imputer.fit(valid_training_df)
        stats = imputer.get_imputation_stats()
        assert stats["total_values_imputed"] >= 0
        assert "columns" in stats


# ---------------------------------------------------------------------------
# CategoricalEncoder
# ---------------------------------------------------------------------------


class TestCategoricalEncoder:
    def test_binary_column_label_encoded(self, valid_training_df):
        df = valid_training_df[["SK_ID_CURR", "TARGET", "FLAG_OWN_CAR"]].copy()
        encoder = CategoricalEncoder()
        result = encoder.fit_transform(df)
        assert "FLAG_OWN_CAR_le" in result.columns
        assert "FLAG_OWN_CAR" not in result.columns

    def test_multi_category_one_hot_encoded(self, valid_training_df):
        df = valid_training_df[
            ["SK_ID_CURR", "TARGET", "NAME_CONTRACT_TYPE"]
        ].copy()
        encoder = CategoricalEncoder()
        result = encoder.fit_transform(df)
        ohe_cols = [c for c in result.columns if "NAME_CONTRACT_TYPE" in c]
        assert len(ohe_cols) >= 2

    def test_encoding_info_populated(self, valid_training_df):
        encoder = CategoricalEncoder()
        encoder.fit(valid_training_df)
        info = encoder.get_encoding_info()
        assert "label_encoded_columns" in info
        assert "one_hot_encoded_columns" in info


# ---------------------------------------------------------------------------
# End-to-end pipeline
# ---------------------------------------------------------------------------


class TestPreprocessingPipeline:
    def test_run_produces_all_artifacts(self, raw_csv, tmp_path):
        raw_dir, _ = raw_csv
        processed_dir = tmp_path / "processed"
        models_dir = tmp_path / "models"
        documents_dir = tmp_path / "documents"

        with patch("app.ml.preprocessing.DATA_RAW_DIR", raw_dir), patch(
            "app.ml.preprocessing.DATA_PROCESSED_DIR", processed_dir
        ), patch("app.ml.preprocessing.ArtifactManager") as MockAM:
            mock_instance = MockAM.return_value
            mock_instance.save_all.return_value = {
                "train_csv": processed_dir / "train.csv",
                "test_csv": processed_dir / "test.csv",
                "pipeline_pkl": models_dir / "preprocessing_pipeline.pkl",
                "metadata_json": models_dir / "feature_metadata.json",
                "report_md": documents_dir / "feature_engineering_report.md",
            }

            pipeline = PreprocessingPipeline(
                raw_dir=raw_dir,
                processed_dir=processed_dir,
            )
            pipeline.artifact_manager = mock_instance
            pipeline.artifact_manager.models_dir = models_dir
            pipeline.artifact_manager.documents_dir = documents_dir

            # Run pipeline logic without mocked save — use real save to tmp
            from app.ml.artifact_manager import ArtifactManager

            pipeline.artifact_manager = ArtifactManager(
                processed_dir=processed_dir,
                models_dir=models_dir,
                documents_dir=documents_dir,
            )
            paths = pipeline.run()

        assert paths["train_csv"].exists()
        assert paths["test_csv"].exists()
        assert paths["pipeline_pkl"].exists()
        assert paths["metadata_json"].exists()
        assert paths["report_md"].exists()

    def test_processed_train_has_no_nulls_in_features(self, raw_csv, tmp_path):
        raw_dir, _ = raw_csv
        processed_dir = tmp_path / "processed"
        models_dir = tmp_path / "models"
        documents_dir = tmp_path / "documents"

        pipeline = PreprocessingPipeline(
            raw_dir=raw_dir,
            processed_dir=processed_dir,
        )
        pipeline.artifact_manager = __import__(
            "app.ml.artifact_manager", fromlist=["ArtifactManager"]
        ).ArtifactManager(
            processed_dir=processed_dir,
            models_dir=models_dir,
            documents_dir=documents_dir,
        )
        pipeline.run()

        train = pd.read_csv(processed_dir / "train.csv")
        feature_cols = [c for c in train.columns if c not in ("SK_ID_CURR", "TARGET")]
        null_counts = train[feature_cols].isnull().sum()
        assert null_counts.sum() == 0

    def test_engineered_features_in_output(self, raw_csv, tmp_path):
        raw_dir, _ = raw_csv
        processed_dir = tmp_path / "processed"
        models_dir = tmp_path / "models"
        documents_dir = tmp_path / "documents"

        pipeline = PreprocessingPipeline(
            raw_dir=raw_dir,
            processed_dir=processed_dir,
        )
        pipeline.artifact_manager = __import__(
            "app.ml.artifact_manager", fromlist=["ArtifactManager"]
        ).ArtifactManager(
            processed_dir=processed_dir,
            models_dir=models_dir,
            documents_dir=documents_dir,
        )
        pipeline.run()

        train = pd.read_csv(processed_dir / "train.csv")
        for feat in (
            "debt_to_income_ratio",
            "credit_to_income_ratio",
            "annuity_to_income_ratio",
            "employment_age_ratio",
        ):
            assert feat in train.columns

    def test_missing_file_raises(self, tmp_path):
        raw_dir = tmp_path / "empty_raw"
        raw_dir.mkdir()
        pipeline = PreprocessingPipeline(raw_dir=raw_dir)
        with pytest.raises(FileNotFoundError):
            pipeline.run()


class TestPipelineSerialization:
    """Verify saved pipeline can transform held-out and unlabelled data."""

    def test_saved_pipeline_transforms_unseen_rows(self, raw_csv, tmp_path):
        import joblib
        from app.ml.artifact_manager import ArtifactManager

        raw_dir, _ = raw_csv
        processed_dir = tmp_path / "processed"
        models_dir = tmp_path / "models"
        documents_dir = tmp_path / "documents"

        pipeline = PreprocessingPipeline(
            raw_dir=raw_dir,
            processed_dir=processed_dir,
            models_dir=models_dir,
            documents_dir=documents_dir,
        )
        paths = pipeline.run()

        loaded = joblib.load(paths["pipeline_pkl"])
        df = pd.read_csv(raw_dir / "application_train.csv")
        unseen = df.iloc[-10:].copy()
        result = loaded.transform(unseen)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 10

    def test_saved_pipeline_transforms_without_target(self, raw_csv, tmp_path):
        import joblib

        raw_dir, _ = raw_csv
        processed_dir = tmp_path / "processed"
        models_dir = tmp_path / "models"
        documents_dir = tmp_path / "documents"

        pipeline = PreprocessingPipeline(
            raw_dir=raw_dir,
            processed_dir=processed_dir,
            models_dir=models_dir,
            documents_dir=documents_dir,
        )
        paths = pipeline.run()

        loaded = joblib.load(paths["pipeline_pkl"])
        df = pd.read_csv(raw_dir / "application_train.csv").drop(columns=["TARGET"])
        result = loaded.transform(df.head(20))
        assert "TARGET" not in result.columns
        assert len(result) == 20

    def test_label_encoder_handles_unseen_category(self, valid_training_df):
        train = valid_training_df.iloc[:150].copy()
        test = valid_training_df.iloc[150:160].copy()
        test.loc[test.index[0], "CODE_GENDER"] = "UNKNOWN_CAT"

        from sklearn.pipeline import Pipeline
        from app.ml.feature_engineering import FeatureEngineer
        from app.ml.feature_selector import FeatureSelector

        pipe = Pipeline([
            ("fe", FeatureEngineer()),
            ("fs", FeatureSelector(missing_threshold=0.8)),
            ("imp", MissingValueImputer()),
            ("enc", CategoricalEncoder()),
        ])
        pipe.fit(train)
        result = pipe.transform(test)
        assert "CODE_GENDER_le" in result.columns
        assert result.loc[result.index[0], "CODE_GENDER_le"] == -1

    def test_artifact_paths_respect_constructor(self, raw_csv, tmp_path):
        raw_dir, _ = raw_csv
        custom_processed = tmp_path / "custom" / "processed"
        custom_models = tmp_path / "custom" / "models"
        custom_docs = tmp_path / "custom" / "docs"

        pipeline = PreprocessingPipeline(
            raw_dir=raw_dir,
            processed_dir=custom_processed,
            models_dir=custom_models,
            documents_dir=custom_docs,
        )
        paths = pipeline.run()

        assert paths["train_csv"].parent == custom_processed
        assert paths["pipeline_pkl"].parent == custom_models
        assert paths["report_md"].parent == custom_docs
