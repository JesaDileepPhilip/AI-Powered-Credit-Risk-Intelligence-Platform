"""
tests/test_train.py — Unit and integration tests for app.ml.train.

Tests:
  - CreditRiskTrainer._load_data() with synthetic CSVs
  - Baseline Logistic Regression training and evaluation
  - compute_scale_pos_weight correctness
  - Cross-validation shape and value ranges
  - Full pipeline smoke test using small synthetic data
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import pytest

from app.ml.train import CreditRiskTrainer


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

def _make_synthetic_csv(
    n_rows: int = 400,
    n_features: int = 10,
    imbalance_ratio: float = 0.1,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create reproducible synthetic train/test DataFrames that mimic the
    Phase 2 output schema (numeric features + TARGET + SK_ID_CURR).
    """
    rng = np.random.default_rng(seed)
    n_pos = max(2, int(n_rows * imbalance_ratio))
    n_neg = n_rows - n_pos
    y = np.concatenate([np.ones(n_pos, dtype=int), np.zeros(n_neg, dtype=int)])
    rng.shuffle(y)

    feature_cols = {f"feature_{i}": rng.standard_normal(n_rows).astype(np.float32)
                    for i in range(n_features)}
    train_df = pd.DataFrame(feature_cols)
    train_df["SK_ID_CURR"] = np.arange(1, n_rows + 1)
    train_df["TARGET"] = y

    # Test set
    n_test = max(10, n_rows // 4)
    y_test = rng.integers(0, 2, size=n_test)
    test_cols = {f"feature_{i}": rng.standard_normal(n_test).astype(np.float32)
                 for i in range(n_features)}
    test_df = pd.DataFrame(test_cols)
    test_df["SK_ID_CURR"] = np.arange(n_rows + 1, n_rows + 1 + n_test)
    test_df["TARGET"] = y_test

    return train_df, test_df


@pytest.fixture()
def tmp_dirs():
    """Provide temporary directories for models and documents."""
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        processed = base / "processed"
        models = base / "models"
        docs = base / "documents"
        for d in (processed, models, docs):
            d.mkdir(parents=True, exist_ok=True)
        yield processed, models, docs


@pytest.fixture()
def synthetic_csvs(tmp_dirs):
    """Write synthetic train/test CSVs and return paths + directories."""
    processed_dir, models_dir, docs_dir = tmp_dirs
    train_df, test_df = _make_synthetic_csv(n_rows=400, n_features=8)
    train_path = processed_dir / "train.csv"
    test_path = processed_dir / "test.csv"
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    return train_path, test_path, models_dir, docs_dir


@pytest.fixture()
def trainer(synthetic_csvs):
    """Instantiate CreditRiskTrainer with synthetic data paths."""
    train_path, test_path, models_dir, docs_dir = synthetic_csvs
    t = CreditRiskTrainer(
        train_csv=train_path,
        test_csv=test_path,
        models_dir=models_dir,
        documents_dir=docs_dir,
    )
    t._load_data()
    return t


# ─────────────────────────────────────────────────────────────────────────────
# Data loading tests
# ─────────────────────────────────────────────────────────────────────────────

class TestDataLoading:
    def test_x_train_is_numpy_array(self, trainer: CreditRiskTrainer) -> None:
        assert isinstance(trainer._X_train, np.ndarray)

    def test_y_train_is_numpy_array(self, trainer: CreditRiskTrainer) -> None:
        assert isinstance(trainer._y_train, np.ndarray)

    def test_x_train_shape_correct(self, trainer: CreditRiskTrainer) -> None:
        assert trainer._X_train.ndim == 2
        assert trainer._X_train.shape[0] == 400  # n_rows

    def test_feature_names_not_empty(self, trainer: CreditRiskTrainer) -> None:
        assert len(trainer._feature_names) > 0

    def test_target_column_not_in_features(self, trainer: CreditRiskTrainer) -> None:
        assert "TARGET" not in trainer._feature_names

    def test_id_column_not_in_features(self, trainer: CreditRiskTrainer) -> None:
        assert "SK_ID_CURR" not in trainer._feature_names

    def test_missing_train_csv_raises(self, tmp_dirs: tuple) -> None:
        processed, models, docs = tmp_dirs
        t = CreditRiskTrainer(
            train_csv=processed / "nonexistent_train.csv",
            test_csv=processed / "nonexistent_test.csv",
            models_dir=models,
            documents_dir=docs,
        )
        with pytest.raises(FileNotFoundError):
            t._load_data()

    def test_y_train_binary(self, trainer: CreditRiskTrainer) -> None:
        unique_vals = set(np.unique(trainer._y_train).tolist())
        assert unique_vals.issubset({0, 1})


# ─────────────────────────────────────────────────────────────────────────────
# Scale pos weight
# ─────────────────────────────────────────────────────────────────────────────

class TestScalePosWeight:
    def test_returns_positive_float(self, trainer: CreditRiskTrainer) -> None:
        spw = trainer._compute_scale_pos_weight()
        assert spw > 0.0

    def test_spw_reflects_imbalance(self, trainer: CreditRiskTrainer) -> None:
        """For 10% positives, SPW should be approximately 9.0."""
        spw = trainer._compute_scale_pos_weight()
        # Allow generous tolerance for small synthetic datasets
        assert 3.0 <= spw <= 20.0


# ─────────────────────────────────────────────────────────────────────────────
# Baseline model
# ─────────────────────────────────────────────────────────────────────────────

class TestBaselineModel:
    def test_baseline_metrics_keys(self, trainer: CreditRiskTrainer) -> None:
        metrics = trainer._train_baseline()
        for key in ("roc_auc", "precision", "recall", "f1_score", "confusion_matrix"):
            assert key in metrics, f"Missing key: {key}"

    def test_roc_auc_in_range(self, trainer: CreditRiskTrainer) -> None:
        metrics = trainer._train_baseline()
        assert 0.0 <= metrics["roc_auc"] <= 1.0

    def test_precision_in_range(self, trainer: CreditRiskTrainer) -> None:
        metrics = trainer._train_baseline()
        assert 0.0 <= metrics["precision"] <= 1.0

    def test_recall_in_range(self, trainer: CreditRiskTrainer) -> None:
        metrics = trainer._train_baseline()
        assert 0.0 <= metrics["recall"] <= 1.0

    def test_confusion_matrix_shape(self, trainer: CreditRiskTrainer) -> None:
        metrics = trainer._train_baseline()
        cm = metrics["confusion_matrix"]
        assert len(cm) == 2
        assert len(cm[0]) == 2
        assert len(cm[1]) == 2


# ─────────────────────────────────────────────────────────────────────────────
# Cross-validation
# ─────────────────────────────────────────────────────────────────────────────

class TestCrossValidation:
    def test_cv_result_keys(self, trainer: CreditRiskTrainer) -> None:
        # Set up training arrays required before CV
        trainer._X_train_final = trainer._X_train
        trainer._y_train_final = trainer._y_train
        cv = trainer._cross_validate()
        for key in ("mean_roc_auc", "std_roc_auc", "fold_scores", "n_folds"):
            assert key in cv

    def test_cv_mean_in_range(self, trainer: CreditRiskTrainer) -> None:
        trainer._X_train_final = trainer._X_train
        trainer._y_train_final = trainer._y_train
        cv = trainer._cross_validate()
        assert 0.0 <= cv["mean_roc_auc"] <= 1.0

    def test_cv_std_non_negative(self, trainer: CreditRiskTrainer) -> None:
        trainer._X_train_final = trainer._X_train
        trainer._y_train_final = trainer._y_train
        cv = trainer._cross_validate()
        assert cv["std_roc_auc"] >= 0.0

    def test_cv_fold_count(self, trainer: CreditRiskTrainer) -> None:
        trainer._X_train_final = trainer._X_train
        trainer._y_train_final = trainer._y_train
        from app.ml.train import CV_FOLDS
        cv = trainer._cross_validate()
        assert len(cv["fold_scores"]) == CV_FOLDS


# ─────────────────────────────────────────────────────────────────────────────
# Smoke test — full pipeline (fast params)
# ─────────────────────────────────────────────────────────────────────────────

class TestFullPipelineSmoke:
    """
    End-to-end smoke test using tiny hyperparameter search (n_iter=2).

    Validates that train.run() completes and returns the expected keys.
    """

    def test_run_returns_required_keys(self, synthetic_csvs, monkeypatch: pytest.MonkeyPatch) -> None:
        train_path, test_path, models_dir, docs_dir = synthetic_csvs

        # Speed up search drastically for CI
        monkeypatch.setattr("app.ml.train.N_ITER_SEARCH", 2)
        monkeypatch.setattr(
            "app.ml.train.LGBM_PARAM_GRID",
            {
                "num_leaves": [15, 31],
                "max_depth": [3, 5],
                "learning_rate": [0.1],
                "min_child_samples": [5],
                "n_estimators": [20, 50],
                "subsample": [0.8],
                "colsample_bytree": [0.8],
            },
        )

        trainer = CreditRiskTrainer(
            train_csv=train_path,
            test_csv=test_path,
            models_dir=models_dir,
            documents_dir=docs_dir,
        )
        output = trainer.run()

        for key in ("model_path", "metadata_path", "metrics_path", "report_path", "summary"):
            assert key in output, f"Missing output key: {key}"

    def test_model_file_is_created(self, synthetic_csvs, monkeypatch: pytest.MonkeyPatch) -> None:
        train_path, test_path, models_dir, docs_dir = synthetic_csvs
        monkeypatch.setattr("app.ml.train.N_ITER_SEARCH", 2)
        monkeypatch.setattr(
            "app.ml.train.LGBM_PARAM_GRID",
            {
                "num_leaves": [15],
                "max_depth": [3],
                "learning_rate": [0.1],
                "min_child_samples": [5],
                "n_estimators": [20],
                "subsample": [0.8],
                "colsample_bytree": [0.8],
            },
        )

        trainer = CreditRiskTrainer(
            train_csv=train_path,
            test_csv=test_path,
            models_dir=models_dir,
            documents_dir=docs_dir,
        )
        output = trainer.run()
        assert Path(output["model_path"]).exists()

    def test_training_metrics_json_valid(self, synthetic_csvs, monkeypatch: pytest.MonkeyPatch) -> None:
        train_path, test_path, models_dir, docs_dir = synthetic_csvs
        monkeypatch.setattr("app.ml.train.N_ITER_SEARCH", 2)
        monkeypatch.setattr(
            "app.ml.train.LGBM_PARAM_GRID",
            {
                "num_leaves": [15],
                "max_depth": [3],
                "learning_rate": [0.1],
                "min_child_samples": [5],
                "n_estimators": [20],
                "subsample": [0.8],
                "colsample_bytree": [0.8],
            },
        )

        trainer = CreditRiskTrainer(
            train_csv=train_path,
            test_csv=test_path,
            models_dir=models_dir,
            documents_dir=docs_dir,
        )
        output = trainer.run()
        metrics_path = Path(output["metrics_path"])
        assert metrics_path.exists()
        with open(metrics_path) as fh:
            data = json.load(fh)
        assert "baseline" in data
        assert "final" in data

    def test_summary_roc_auc_in_range(self, synthetic_csvs, monkeypatch: pytest.MonkeyPatch) -> None:
        train_path, test_path, models_dir, docs_dir = synthetic_csvs
        monkeypatch.setattr("app.ml.train.N_ITER_SEARCH", 2)
        monkeypatch.setattr(
            "app.ml.train.LGBM_PARAM_GRID",
            {
                "num_leaves": [15],
                "max_depth": [3],
                "learning_rate": [0.1],
                "min_child_samples": [5],
                "n_estimators": [20],
                "subsample": [0.8],
                "colsample_bytree": [0.8],
            },
        )

        trainer = CreditRiskTrainer(
            train_csv=train_path,
            test_csv=test_path,
            models_dir=models_dir,
            documents_dir=docs_dir,
        )
        output = trainer.run()
        auc = output["summary"]["final_roc_auc"]
        assert 0.0 <= auc <= 1.0
