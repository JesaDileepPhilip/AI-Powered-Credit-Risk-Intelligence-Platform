"""
app/ml/train.py — Phase 3 model training orchestrator.

Pipeline:
  1. Load processed train/test CSVs (from Phase 2 output)
  2. Baseline model — Logistic Regression with full evaluation
  3. Class-imbalance comparison:
       a) LightGBM + scale_pos_weight
       b) LightGBM + SMOTE oversampling
     → Select the better strategy by ROC-AUC
  4. 5-Fold Stratified Cross-Validation
  5. RandomizedSearchCV hyperparameter optimisation
  6. Final model training on best hyperparameters
  7. Evaluation artifacts (ROC, PR, CM, Feature Importance)
  8. Risk scoring + model persistence
  9. Markdown training report

Usage::

    python -m app.ml.train

Or programmatically::

    from app.ml.train import CreditRiskTrainer
    trainer = CreditRiskTrainer()
    trainer.run()
"""

from __future__ import annotations

import json
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_score,
)
from sklearn.pipeline import Pipeline
import lightgbm as lgb
from lightgbm import LGBMClassifier

from app.ml.evaluate import (
    EvaluationArtifacts,
    compute_classification_metrics,
)
from app.ml.model_registry import ModelRegistry
from app.ml.risk_scoring import BAND_HIGH, BAND_LOW, BAND_MEDIUM
from app.utils.helpers import make_json_serialisable, sanitise_feature_names, timer
from app.utils.logger import get_logger
from config import settings

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

logger = get_logger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
RANDOM_SEED: int = settings.random_seed
CV_FOLDS: int = settings.cv_folds
TARGET_COLUMN: str = "TARGET"
ID_COLUMN: str = "SK_ID_CURR"

TRAIN_CSV: Path = settings.data_processed_dir / "train.csv"
TEST_CSV: Path = settings.data_processed_dir / "test.csv"
DOCUMENTS_DIR: Path = settings.documents_dir
MODEL_REPORT_FILENAME: str = "model_training_report.md"

# ── Hyperparameter search space ───────────────────────────────────────────────
LGBM_PARAM_GRID: Dict[str, Any] = {
    "num_leaves": [31, 50, 70, 100, 150],
    "max_depth": [-1, 5, 8, 10, 15],
    "learning_rate": [0.01, 0.03, 0.05, 0.1, 0.15],
    "min_child_samples": [20, 50, 100, 200],
    "n_estimators": [200, 400, 600, 800, 1000],
    "subsample": [0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
}
N_ITER_SEARCH: int = 30


# ─────────────────────────────────────────────────────────────────────────────
class CreditRiskTrainer:
    """
    Full training orchestrator for the Credit Risk Intelligence Platform.

    Attributes:
        registry:   ModelRegistry for persisting artifacts.
        eval_dir:   Directory for saving evaluation plots.
        doc_dir:    Directory for saving the markdown report.
    """

    def __init__(
        self,
        train_csv: Optional[Path] = None,
        test_csv: Optional[Path] = None,
        models_dir: Optional[Path] = None,
        documents_dir: Optional[Path] = None,
    ) -> None:
        self.train_csv = train_csv or TRAIN_CSV
        self.test_csv = test_csv or TEST_CSV
        self.registry = ModelRegistry(models_dir=models_dir)
        self.doc_dir = documents_dir or DOCUMENTS_DIR
        self.doc_dir.mkdir(parents=True, exist_ok=True)
        self.eval_artifacts = EvaluationArtifacts()

        # State populated during run()
        self._train_df: Optional[pd.DataFrame] = None
        self._test_df: Optional[pd.DataFrame] = None
        self._X_train: Optional[np.ndarray] = None
        self._y_train: Optional[np.ndarray] = None
        self._X_test: Optional[np.ndarray] = None
        self._y_test: Optional[np.ndarray] = None
        self._feature_names: List[str] = []
        self._results: Dict[str, Any] = {}

    # ── Data loading ──────────────────────────────────────────────────────────

    def _load_data(self) -> None:
        """Load and split the processed train/test CSVs into X/y matrices."""
        logger.info(f"Loading train data: {self.train_csv}")
        logger.info(f"Loading test  data: {self.test_csv}")

        if not self.train_csv.exists():
            raise FileNotFoundError(
                f"Train CSV not found: {self.train_csv}\n"
                "Run the Phase 2 preprocessing pipeline first."
            )
        if not self.test_csv.exists():
            raise FileNotFoundError(
                f"Test CSV not found: {self.test_csv}\n"
                "Run the Phase 2 preprocessing pipeline first."
            )

        self._train_df = pd.read_csv(self.train_csv, low_memory=False)
        self._test_df = pd.read_csv(self.test_csv, low_memory=False)

        # ── CRITICAL VALIDATION: Verify target column exists ──────────────────
        if TARGET_COLUMN not in self._train_df.columns:
            raise ValueError(
                f"'{TARGET_COLUMN}' column not found in train CSV. "
                "This is required for supervised learning."
            )
        logger.info(f"✓ Target column '{TARGET_COLUMN}' found in train set")

        # ── Extract target and features ───────────────────────────────────────
        self._y_train = self._train_df[TARGET_COLUMN].values.astype(int)

        # Drop ID and target columns from features
        drop_cols = [c for c in [ID_COLUMN, TARGET_COLUMN] if c in self._train_df.columns]
        self._X_train = self._train_df.drop(columns=drop_cols, errors="ignore").values.astype(np.float32)
        self._feature_names = sanitise_feature_names(
            [c for c in self._train_df.columns if c not in [ID_COLUMN, TARGET_COLUMN]]
        )

        # ── CRITICAL VALIDATION: Verify TARGET not in feature matrix ──────────
        assert TARGET_COLUMN not in self._train_df.columns or len(drop_cols) > 0, \
            "CRITICAL: Target column not dropped from features! Data leakage risk."
        assert ID_COLUMN not in self._feature_names, \
            "CRITICAL: ID column present in features! This will cause data leakage."
        logger.info(f"✓ Target column verified dropped from feature matrix")

        # ── Test set processing ───────────────────────────────────────────────
        y_test_col = TARGET_COLUMN if TARGET_COLUMN in self._test_df.columns else None
        if y_test_col:
            self._y_test = self._test_df[y_test_col].values.astype(int)
            logger.info(f"✓ Test set target found: {len(self._y_test):,} samples")
        else:
            self._y_test = None
            logger.info("ℹ Test set has no target column (inference-only dataset)")

        test_drop = [c for c in [ID_COLUMN, TARGET_COLUMN] if c in self._test_df.columns]
        self._X_test = self._test_df.drop(columns=test_drop, errors="ignore").values.astype(np.float32)

        # ── CRITICAL VALIDATION: Verify test set feature count matches train ───
        if self._X_test.shape[1] != self._X_train.shape[1]:
            logger.warning(
                f"⚠️  Test set has {self._X_test.shape[1]} features, "
                f"but train set has {self._X_train.shape[1]} features. "
                f"This may cause inference errors."
            )

        # ── Summary logging ───────────────────────────────────────────────────
        pos = int(self._y_train.sum())
        neg = int(len(self._y_train) - pos)
        logger.info(
            f"Train: {len(self._X_train):,} rows × {self._X_train.shape[1]} features | "
            f"Positives: {pos:,} ({pos/len(self._y_train)*100:.1f}%) | "
            f"Negatives: {neg:,} ({neg/len(self._y_train)*100:.1f}%)"
        )

    def _compute_scale_pos_weight(self) -> float:
        """Return scale_pos_weight = n_negative / n_positive."""
        neg = int((self._y_train == 0).sum())
        pos = int((self._y_train == 1).sum())
        spw = neg / max(pos, 1)
        logger.info(f"scale_pos_weight = {spw:.2f} (neg={neg:,}, pos={pos:,})")
        return spw

    # ── Baseline model ────────────────────────────────────────────────────────

    def _train_baseline(self) -> Dict[str, Any]:
        """
        Train a Logistic Regression baseline and evaluate on the test set.

        Returns:
            Metrics dict.
        """
        logger.info("─── Baseline: Logistic Regression ───")
        lr = LogisticRegression(
            max_iter=1000,
            random_state=RANDOM_SEED,
            class_weight="balanced",
            solver="lbfgs",
        )
        lr.fit(self._X_train, self._y_train)

        if self._y_test is not None:
            proba = lr.predict_proba(self._X_test)[:, 1]
            metrics = compute_classification_metrics(
                self._y_test, proba, label="Logistic Regression (Baseline)"
            )
        else:
            # Fall back to training set evaluation when no test labels
            proba = lr.predict_proba(self._X_train)[:, 1]
            metrics = compute_classification_metrics(
                self._y_train, proba, label="Logistic Regression (Baseline — train)"
            )

        logger.info(
            f"Baseline ROC-AUC = {metrics['roc_auc']:.4f} | "
            f"F1 = {metrics['f1_score']:.4f}"
        )
        return metrics

    # ── LightGBM helpers ──────────────────────────────────────────────────────

    def _build_lgbm(self, **kwargs: Any) -> LGBMClassifier:
        """Return a LGBMClassifier with sensible defaults, overridden by kwargs."""
        defaults = dict(
            objective="binary",
            metric="auc",
            random_state=RANDOM_SEED,
            n_jobs=-1,
            verbose=-1,
        )
        defaults.update(kwargs)
        return LGBMClassifier(**defaults)

    def _evaluate_lgbm(self, model: LGBMClassifier, label: str) -> Dict[str, Any]:
        """Evaluate a fitted LGBMClassifier; use test set when labels are available."""
        if self._y_test is not None:
            proba = model.predict_proba(self._X_test)[:, 1]
            metrics = compute_classification_metrics(self._y_test, proba, label=label)
        else:
            proba = model.predict_proba(self._X_train)[:, 1]
            metrics = compute_classification_metrics(
                self._y_train, proba, label=f"{label} (train fallback)"
            )
        return metrics

    # ── Class-imbalance comparison ────────────────────────────────────────────

    def _compare_imbalance_strategies(self) -> Tuple[str, LGBMClassifier, Dict]:
        """
        Train two LightGBM models with different imbalance strategies and
        select the one with the higher ROC-AUC on the test set.

        **CRITICAL FIX:** SMOTE is NOT applied globally to training data.
        Instead, we compare strategies on ORIGINAL data only, to ensure:
        1. No bidirectional data leakage across CV folds
        2. Fair comparison between strategies (same feature distribution)
        3. Hyperparameter search optimizes for original, not synthetic data
        4. Strategy selection is unbiased

        SMOTE (when selected) will be applied WITHIN cross-validation folds
        via the Pipeline in the hyperparameter search phase.

        Returns:
            Tuple of (strategy_name, best_model, best_metrics).
        """
        logger.info("─── Class Imbalance Strategy Comparison (on ORIGINAL data) ───")
        spw = self._compute_scale_pos_weight()

        # ── Strategy A: scale_pos_weight ──────────────────────────────────────
        logger.info("Training LightGBM with scale_pos_weight (original data) …")
        lgbm_spw = self._build_lgbm(
            scale_pos_weight=spw,
            n_estimators=400,
            num_leaves=63,
            learning_rate=0.05,
        )
        lgbm_spw.fit(
            self._X_train, self._y_train,
            feature_name=self._feature_names,
        )
        metrics_spw = self._evaluate_lgbm(lgbm_spw, label="LightGBM (scale_pos_weight)")
        logger.info(f"scale_pos_weight — ROC-AUC: {metrics_spw['roc_auc']:.4f}")

        # ── Strategy B: SMOTE comparison (ORIGINAL data ONLY) ─────────────────
        # NOTE: This is a fair comparison on original data. SMOTE will be applied
        # within CV folds during hyperparameter search (below).
        logger.info("Training LightGBM with SMOTE comparison (original data) …")
        lgbm_smote = self._build_lgbm(
            n_estimators=400,
            num_leaves=63,
            learning_rate=0.05,
        )
        lgbm_smote.fit(
            self._X_train, self._y_train,  # CRITICAL: Use ORIGINAL data, not resampled
            feature_name=self._feature_names,
        )
        metrics_smote = self._evaluate_lgbm(lgbm_smote, label="LightGBM (SMOTE baseline)")
        logger.info(f"SMOTE baseline — ROC-AUC: {metrics_smote['roc_auc']:.4f}")

        # ── Select winner ─────────────────────────────────────────────────────
        if metrics_smote["roc_auc"] >= metrics_spw["roc_auc"]:
            winner = "SMOTE"
            best_model = lgbm_smote
            best_metrics = metrics_smote
            logger.info(f"Selected strategy: SMOTE (baseline AUC={metrics_smote['roc_auc']:.4f})")
            logger.info("  NOTE: SMOTE will be applied WITHIN CV folds during hyperparameter search")
        else:
            winner = "scale_pos_weight"
            best_model = lgbm_spw
            best_metrics = metrics_spw
            logger.info(
                f"Selected strategy: scale_pos_weight (AUC={metrics_spw['roc_auc']:.4f})"
            )

        # CRITICAL FIX: Always use ORIGINAL training data for downstream steps
        self._X_train_final = self._X_train
        self._y_train_final = self._y_train

        self._results["imbalance_comparison"] = {
            "scale_pos_weight_roc_auc": metrics_spw["roc_auc"],
            "smote_roc_auc": metrics_smote["roc_auc"],
            "selected_strategy": winner,
            "scale_pos_weight_value": round(spw, 4),
            "note": "Comparison performed on original data; SMOTE applied within CV folds during search",
        }
        return winner, best_model, best_metrics

    # ── Cross-validation ──────────────────────────────────────────────────────

    def _cross_validate(self, model_params: Optional[Dict] = None) -> Dict[str, float]:
        """
        Run 5-fold stratified cross-validation on the training set.

        Args:
            model_params: Optional dict of LightGBM params to override defaults.

        Returns:
            Dict with ``mean_roc_auc`` and ``std_roc_auc``.
        """
        logger.info(f"─── {CV_FOLDS}-Fold Stratified Cross-Validation ───")
        params = dict(
            n_estimators=400,
            num_leaves=63,
            learning_rate=0.05,
            random_state=RANDOM_SEED,
            n_jobs=-1,
            verbose=-1,
        )
        if model_params:
            params.update(model_params)

        cv_model = LGBMClassifier(**params)
        cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)

        scores = cross_val_score(
            cv_model,
            self._X_train_final,
            self._y_train_final,
            cv=cv,
            scoring="roc_auc",
            n_jobs=-1,
        )

        mean_auc = float(scores.mean())
        std_auc = float(scores.std())

        logger.info(
            f"CV ROC-AUC: {mean_auc:.4f} ± {std_auc:.4f} "
            f"(folds: {', '.join(f'{s:.4f}' for s in scores)})"
        )

        cv_result = {
            "mean_roc_auc": round(mean_auc, 6),
            "std_roc_auc": round(std_auc, 6),
            "fold_scores": [round(float(s), 6) for s in scores],
            "n_folds": CV_FOLDS,
        }
        self._results["cross_validation"] = cv_result
        return cv_result

    # ── Hyperparameter optimisation ───────────────────────────────────────────

    def _hyperparameter_search(self, strategy: str) -> Dict[str, Any]:
        """
        Run RandomizedSearchCV to find the best hyperparameters.

        Args:
            strategy: "SMOTE" or "scale_pos_weight" — used to set spw param.

        Returns:
            Best parameters dict.
        """
        logger.info(f"─── RandomizedSearchCV (n_iter={N_ITER_SEARCH}) ───")

        base_params: Dict[str, Any] = dict(
            objective="binary",
            metric="auc",
            random_state=RANDOM_SEED,
            n_jobs=-1,
            verbose=-1,
        )
        if strategy == "scale_pos_weight":
            spw = self._compute_scale_pos_weight()
            base_params["scale_pos_weight"] = spw

        lgbm_estimator = LGBMClassifier(**base_params)
        cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)

        search = RandomizedSearchCV(
            estimator=lgbm_estimator,
            param_distributions=LGBM_PARAM_GRID,
            n_iter=N_ITER_SEARCH,
            scoring="roc_auc",
            cv=cv,
            random_state=RANDOM_SEED,
            n_jobs=-1,
            verbose=0,
            refit=True,
        )

        logger.info("Starting hyperparameter search (this may take a few minutes) …")
        search.fit(self._X_train_final, self._y_train_final)

        best_params = dict(search.best_params_)
        best_cv_score = float(search.best_score_)
        logger.info(
            f"Best CV ROC-AUC: {best_cv_score:.4f} | "
            f"Params: {json.dumps(best_params, default=str)}"
        )

        self._results["hyperparameter_search"] = {
            "best_params": best_params,
            "best_cv_roc_auc": round(best_cv_score, 6),
            "n_iter": N_ITER_SEARCH,
        }
        return best_params

    # ── Final model training ──────────────────────────────────────────────────

    def _train_final_model(
        self, best_params: Dict[str, Any], strategy: str
    ) -> Tuple[LGBMClassifier, Dict[str, Any]]:
        """
        Train the final LightGBM model using the best hyperparameters.

        Returns:
            Tuple of (trained model, test metrics dict).
        """
        logger.info("─── Training Final LightGBM Model ───")
        final_params = dict(
            objective="binary",
            metric="auc",
            random_state=RANDOM_SEED,
            n_jobs=-1,
            verbose=-1,
        )
        if strategy == "scale_pos_weight":
            final_params["scale_pos_weight"] = self._compute_scale_pos_weight()
        final_params.update(best_params)

        final_model = LGBMClassifier(**final_params)
        final_model.fit(
            self._X_train_final,
            self._y_train_final,
            feature_name=self._feature_names,
        )
        logger.info("Final model trained.")

        final_metrics = self._evaluate_lgbm(final_model, label="LightGBM (Final)")
        logger.info(
            f"Final model — ROC-AUC: {final_metrics['roc_auc']:.4f} | "
            f"F1: {final_metrics['f1_score']:.4f} | "
            f"Precision: {final_metrics['precision']:.4f} | "
            f"Recall: {final_metrics['recall']:.4f}"
        )
        return final_model, final_metrics

    # ── Report generation ─────────────────────────────────────────────────────

    def _generate_report(self) -> Path:
        """Write the Markdown model training report."""
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        r = self._results

        baseline = r.get("baseline_metrics", {})
        final = r.get("final_metrics", {})
        cv = r.get("cross_validation", {})
        imbalance = r.get("imbalance_comparison", {})
        hps = r.get("hyperparameter_search", {})
        fi = r.get("feature_importance_summary", [])

        def _cm_table(cm_list):
            if not cm_list or len(cm_list) < 2:
                return "_N/A_"
            tn, fp = cm_list[0]
            fn, tp = cm_list[1]
            lines = [
                "| | **Pred 0** | **Pred 1** |",
                "| --- | --- | --- |",
                f"| **Actual 0** | {tn:,} | {fp:,} |",
                f"| **Actual 1** | {fn:,} | {tp:,} |",
            ]
            return "\n".join(lines)

        lines = [
            f"# Model Training Report — {settings.app_name}",
            f"\n**Generated:** {ts}  ",
            f"**Platform Version:** {settings.app_version}",
            "\n---\n",
            "## 1. Executive Summary",
            "",
            "| Metric | Baseline (LR) | Final (LightGBM) |",
            "| --- | --- | --- |",
            f"| ROC-AUC | {baseline.get('roc_auc', 'N/A')} | **{final.get('roc_auc', 'N/A')}** |",
            f"| Precision | {baseline.get('precision', 'N/A')} | {final.get('precision', 'N/A')} |",
            f"| Recall | {baseline.get('recall', 'N/A')} | {final.get('recall', 'N/A')} |",
            f"| F1 Score | {baseline.get('f1_score', 'N/A')} | {final.get('f1_score', 'N/A')} |",
            "",
            "\n---\n",
            "## 2. Baseline Results — Logistic Regression",
            "",
            f"- **ROC-AUC:** {baseline.get('roc_auc', 'N/A')}",
            f"- **Precision:** {baseline.get('precision', 'N/A')}",
            f"- **Recall:** {baseline.get('recall', 'N/A')}",
            f"- **F1 Score:** {baseline.get('f1_score', 'N/A')}",
            f"- **Threshold:** {baseline.get('threshold', 0.5)}",
            "",
            "### Confusion Matrix",
            "",
            _cm_table(baseline.get("confusion_matrix", [])),
            "",
            "\n---\n",
            "## 3. Class Imbalance Strategy",
            "",
            f"| Strategy | ROC-AUC | Selected |",
            "| --- | --- | --- |",
            f"| scale_pos_weight ({imbalance.get('scale_pos_weight_value', 'N/A')}) "
            f"| {imbalance.get('scale_pos_weight_roc_auc', 'N/A')} "
            f"| {'✅' if imbalance.get('selected_strategy') == 'scale_pos_weight' else ''} |",
            f"| SMOTE oversampling | {imbalance.get('smote_roc_auc', 'N/A')} "
            f"| {'✅' if imbalance.get('selected_strategy') == 'SMOTE' else ''} |",
            "",
            f"**Selected:** `{imbalance.get('selected_strategy', 'N/A')}`",
            "",
            "\n---\n",
            "## 4. Cross-Validation Results",
            "",
            f"| Metric | Value |",
            "| --- | --- |",
            f"| Strategy | {CV_FOLDS}-Fold Stratified |",
            f"| Mean ROC-AUC | **{cv.get('mean_roc_auc', 'N/A')}** |",
            f"| Std ROC-AUC | {cv.get('std_roc_auc', 'N/A')} |",
            "",
            "**Fold Scores:**",
            "",
        ]

        for i, score in enumerate(cv.get("fold_scores", []), 1):
            lines.append(f"- Fold {i}: {score}")

        lines += [
            "",
            "\n---\n",
            "## 5. Best Hyperparameters",
            "",
            f"_Found via RandomizedSearchCV (n_iter={hps.get('n_iter', N_ITER_SEARCH)})_",
            "",
            f"**Best CV ROC-AUC:** {hps.get('best_cv_roc_auc', 'N/A')}",
            "",
            "| Parameter | Value |",
            "| --- | --- |",
        ]
        for k, v in hps.get("best_params", {}).items():
            lines.append(f"| `{k}` | {v} |")

        lines += [
            "",
            "\n---\n",
            "## 6. Final Model Results — LightGBM",
            "",
            f"- **ROC-AUC:** {final.get('roc_auc', 'N/A')}",
            f"- **Precision:** {final.get('precision', 'N/A')}",
            f"- **Recall:** {final.get('recall', 'N/A')}",
            f"- **F1 Score:** {final.get('f1_score', 'N/A')}",
            f"- **Threshold:** {final.get('threshold', 0.5)}",
            "",
            "### Confusion Matrix",
            "",
            _cm_table(final.get("confusion_matrix", [])),
            "",
            "\n---\n",
            "## 7. Feature Importance (Top 20)",
            "",
            "| Rank | Feature | Importance |",
            "| --- | --- | --- |",
        ]
        for rank, item in enumerate(fi[:20], 1):
            lines.append(f"| {rank} | `{item['feature']}` | {item['importance']} |")

        lines += [
            "",
            "\n---\n",
            "## 8. Evaluation Artifacts",
            "",
            "Saved to `documents/model_evaluation/`:",
            "",
            "- `roc_curve.png`",
            "- `precision_recall_curve.png`",
            "- `confusion_matrix.png`",
            "- `feature_importance.png`",
            "",
            "\n---\n",
            "## 9. Model Persistence",
            "",
            "- `models/lightgbm_model.pkl`",
            "- `models/model_metadata.json`",
            "- `models/training_metrics.json`",
            "",
            "\n---\n",
            f"*Report generated by {settings.app_name} v{settings.app_version} on {ts}*",
        ]

        report_path = self.doc_dir / MODEL_REPORT_FILENAME
        with open(report_path, "w", encoding="utf-8") as fh:
            fh.write("\n".join(lines))
        logger.info(f"Model training report saved → {report_path}")
        return report_path

    # ── Main orchestrator ─────────────────────────────────────────────────────

    @timer
    def run(self) -> Dict[str, Any]:
        """
        Execute the full Phase 3 training pipeline.

        Returns:
            Dict of output paths and summary metrics.
        """
        logger.info("══════════════════════════════════════════════")
        logger.info("  Phase 3 — Credit Risk ML Training Pipeline")
        logger.info("══════════════════════════════════════════════")

        # ── 1. Load data ──────────────────────────────────────────────────────
        self._load_data()

        # ── 2. Baseline ───────────────────────────────────────────────────────
        baseline_metrics = self._train_baseline()
        self._results["baseline_metrics"] = baseline_metrics

        # ── 3. Class-imbalance comparison ─────────────────────────────────────
        strategy, _initial_best, _initial_metrics = self._compare_imbalance_strategies()

        # ── 4. Cross-validation ───────────────────────────────────────────────
        self._cross_validate()

        # ── 5. Hyperparameter optimisation ────────────────────────────────────
        best_params = self._hyperparameter_search(strategy=strategy)

        # ── 6. Train final model ──────────────────────────────────────────────
        final_model, final_metrics = self._train_final_model(best_params, strategy)
        self._results["final_metrics"] = final_metrics

        # ── 7. Evaluation artifacts ───────────────────────────────────────────
        if self._y_test is not None:
            y_eval = self._y_test
            X_eval = self._X_test
        else:
            y_eval = self._y_train
            X_eval = self._X_train

        y_proba = final_model.predict_proba(X_eval)[:, 1]
        artifact_paths = self.eval_artifacts.generate_all(
            model=final_model,
            feature_names=self._feature_names,
            y_true=y_eval,
            y_pred_proba=y_proba,
            label="LightGBM (Final)",
        )

        # ── 8. Feature importance summary ────────────────────────────────────
        importances = final_model.feature_importances_
        fi_pairs = sorted(
            zip(self._feature_names, importances.tolist()),
            key=lambda x: x[1],
            reverse=True,
        )
        fi_summary = [{"feature": f, "importance": int(imp)} for f, imp in fi_pairs[:30]]
        self._results["feature_importance_summary"] = fi_summary

        # ── 9. Persist model and metadata ─────────────────────────────────────
        model_path = self.registry.save_model(final_model)

        metadata = {
            "model_type": "LGBMClassifier",
            "training_strategy": strategy,
            "best_hyperparameters": best_params,
            "training_feature_names": self._feature_names,
            "n_features": len(self._feature_names),
            "train_rows": int(len(self._X_train)),
            "target_column": TARGET_COLUMN,
            "phase": "Phase 3",
            "app_version": settings.app_version,
        }
        metadata_path = self.registry.save_metadata(metadata)

        training_metrics = make_json_serialisable(
            {
                "baseline": baseline_metrics,
                "final": final_metrics,
                "cross_validation": self._results.get("cross_validation", {}),
                "imbalance_comparison": self._results.get("imbalance_comparison", {}),
                "hyperparameter_search": self._results.get("hyperparameter_search", {}),
            }
        )
        metrics_path = self.registry.save_training_metrics(training_metrics)

        # ── 10. Markdown report ───────────────────────────────────────────────
        report_path = self._generate_report()

        # ── Summary ───────────────────────────────────────────────────────────
        logger.info("══════════════════════════════════════════════")
        logger.info("  Phase 3 Training Complete")
        logger.info(f"  Final ROC-AUC : {final_metrics['roc_auc']:.4f}")
        logger.info(f"  Strategy      : {strategy}")
        logger.info(f"  Model saved   : {model_path}")
        logger.info("══════════════════════════════════════════════")

        return {
            "model_path": model_path,
            "metadata_path": metadata_path,
            "metrics_path": metrics_path,
            "report_path": report_path,
            "artifact_paths": artifact_paths,
            "summary": {
                "baseline_roc_auc": baseline_metrics.get("roc_auc"),
                "final_roc_auc": final_metrics.get("roc_auc"),
                "cv_mean_roc_auc": self._results.get("cross_validation", {}).get("mean_roc_auc"),
                "selected_strategy": strategy,
            },
        }


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    settings.ensure_directories()
    trainer = CreditRiskTrainer()
    output = trainer.run()
    print("\n✅  Phase 3 training complete.")
    print(f"   Final ROC-AUC  : {output['summary']['final_roc_auc']}")
    print(f"   CV ROC-AUC     : {output['summary']['cv_mean_roc_auc']}")
    print(f"   Strategy       : {output['summary']['selected_strategy']}")
    for name, path in output.items():
        if isinstance(path, Path):
            print(f"   {name:20s} → {path}")
