"""
app/ml/evaluate.py — Model evaluation utilities.

Provides:
  - compute_classification_metrics()  : ROC-AUC, Precision, Recall, F1, Confusion Matrix
  - plot_roc_curve()                  : ROC Curve figure
  - plot_precision_recall_curve()     : Precision-Recall Curve figure
  - plot_confusion_matrix()           : Heatmap confusion matrix
  - plot_feature_importance()         : Top-N feature importances for LightGBM
  - EvaluationArtifacts               : Orchestrates all plotting and saves figures

All figures are saved to ``documents/model_evaluation/``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # non-interactive backend — safe in CI / headless envs

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    auc,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from app.utils.logger import get_logger
from config import settings

logger = get_logger(__name__)

# ── Output directory ──────────────────────────────────────────────────────────
MODEL_EVAL_DIR: Path = settings.documents_dir / "model_evaluation"

# ── Plot style ────────────────────────────────────────────────────────────────
FIGURE_DPI: int = 150
PALETTE = sns.color_palette("husl", 8)


def _ensure_eval_dir(output_dir: Optional[Path] = None) -> Path:
    """Create and return the model evaluation output directory."""
    d = output_dir or MODEL_EVAL_DIR
    d.mkdir(parents=True, exist_ok=True)
    return d


# ─────────────────────────────────────────────────────────────────────────────
# Metric computation
# ─────────────────────────────────────────────────────────────────────────────

def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    threshold: float = 0.5,
    label: str = "model",
) -> Dict[str, Any]:
    """
    Compute standard binary classification metrics.

    Args:
        y_true:        Ground-truth labels (0 / 1).
        y_pred_proba:  Predicted probabilities for the positive class.
        threshold:     Decision threshold (default 0.5).
        label:         Identifier included in the returned dict.

    Returns:
        Dict with keys: label, roc_auc, precision, recall, f1, threshold,
        confusion_matrix (as nested list).
    """
    y_pred = (y_pred_proba >= threshold).astype(int)

    roc_auc = float(roc_auc_score(y_true, y_pred_proba))
    precision = float(precision_score(y_true, y_pred, zero_division=0))
    recall = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    cm = confusion_matrix(y_true, y_pred).tolist()

    metrics = {
        "label": label,
        "roc_auc": round(roc_auc, 6),
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1_score": round(f1, 6),
        "threshold": threshold,
        "confusion_matrix": cm,
    }

    logger.info(
        f"[{label}] ROC-AUC={roc_auc:.4f} | "
        f"Precision={precision:.4f} | Recall={recall:.4f} | F1={f1:.4f}"
    )
    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# Plotting helpers
# ─────────────────────────────────────────────────────────────────────────────

def plot_roc_curve(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    label: str = "LightGBM",
    output_dir: Optional[Path] = None,
    filename: str = "roc_curve.png",
) -> Path:
    """
    Plot and save a ROC Curve.

    Returns:
        Path to saved figure.
    """
    d = _ensure_eval_dir(output_dir)
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(8, 6), dpi=FIGURE_DPI)
    ax.plot(fpr, tpr, color=PALETTE[0], lw=2, label=f"{label} (AUC = {roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], color="grey", lw=1, linestyle="--", label="Random Classifier")
    ax.fill_between(fpr, tpr, alpha=0.08, color=PALETTE[0])
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curve — Credit Default Prediction", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(alpha=0.3)
    sns.despine()

    path = d / filename
    fig.savefig(path, bbox_inches="tight", dpi=FIGURE_DPI)
    plt.close(fig)
    logger.info(f"ROC Curve saved → {path}")
    return path


def plot_precision_recall_curve(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    label: str = "LightGBM",
    output_dir: Optional[Path] = None,
    filename: str = "precision_recall_curve.png",
) -> Path:
    """
    Plot and save a Precision-Recall Curve.

    Returns:
        Path to saved figure.
    """
    d = _ensure_eval_dir(output_dir)
    precision_vals, recall_vals, _ = precision_recall_curve(y_true, y_pred_proba)
    ap = average_precision_score(y_true, y_pred_proba)
    baseline = float(y_true.mean())

    fig, ax = plt.subplots(figsize=(8, 6), dpi=FIGURE_DPI)
    ax.plot(
        recall_vals, precision_vals,
        color=PALETTE[1], lw=2,
        label=f"{label} (AP = {ap:.4f})",
    )
    ax.axhline(y=baseline, color="grey", lw=1, linestyle="--", label=f"Baseline (prevalence={baseline:.3f})")
    ax.fill_between(recall_vals, precision_vals, alpha=0.08, color=PALETTE[1])
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title("Precision-Recall Curve — Credit Default Prediction", fontsize=14, fontweight="bold")
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(alpha=0.3)
    sns.despine()

    path = d / filename
    fig.savefig(path, bbox_inches="tight", dpi=FIGURE_DPI)
    plt.close(fig)
    logger.info(f"Precision-Recall Curve saved → {path}")
    return path


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    threshold: float = 0.5,
    label: str = "LightGBM",
    output_dir: Optional[Path] = None,
    filename: str = "confusion_matrix.png",
) -> Path:
    """
    Plot and save a confusion matrix heatmap.

    Returns:
        Path to saved figure.
    """
    d = _ensure_eval_dir(output_dir)
    y_pred = (y_pred_proba >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(7, 5), dpi=FIGURE_DPI)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["No Default (0)", "Default (1)"],
    )
    disp.plot(
        ax=ax,
        cmap="Blues",
        colorbar=False,
        values_format="d",
    )
    ax.set_title(
        f"Confusion Matrix — {label} (threshold={threshold})",
        fontsize=13, fontweight="bold", pad=15,
    )
    # Annotate percentages
    total = cm.sum()
    for text_obj, val in zip(disp.text_.ravel(), cm.ravel()):
        pct = val / total * 100
        text_obj.set_text(f"{val:,}\n({pct:.1f}%)")

    path = d / filename
    fig.savefig(path, bbox_inches="tight", dpi=FIGURE_DPI)
    plt.close(fig)
    logger.info(f"Confusion Matrix saved → {path}")
    return path


def plot_feature_importance(
    model: Any,
    feature_names: List[str],
    top_n: int = 30,
    output_dir: Optional[Path] = None,
    filename: str = "feature_importance.png",
) -> Path:
    """
    Plot and save a horizontal bar chart of top-N feature importances.

    Args:
        model:         Fitted LightGBM classifier (must have ``feature_importances_``).
        feature_names: List of feature names matching model's training columns.
        top_n:         Number of top features to display.
        output_dir:    Directory to save the figure (defaults to MODEL_EVAL_DIR).
        filename:      Output file name.

    Returns:
        Path to saved figure.
    """
    d = _ensure_eval_dir(output_dir)
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]

    top_names = [feature_names[i] for i in indices]
    top_vals = importances[indices]
    # Normalise to percentages
    total = top_vals.sum() or 1
    top_pct = top_vals / total * 100

    fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.35)), dpi=FIGURE_DPI)
    colors = sns.color_palette("viridis", len(top_names))
    bars = ax.barh(range(len(top_names)), top_pct[::-1], color=colors[::-1], edgecolor="none")
    ax.set_yticks(range(len(top_names)))
    ax.set_yticklabels(top_names[::-1], fontsize=9)
    ax.set_xlabel("Relative Importance (%)", fontsize=11)
    ax.set_title(f"Top {top_n} Feature Importances (LightGBM)", fontsize=13, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    sns.despine()

    # Value labels
    for bar, pct in zip(bars, top_pct[::-1]):
        ax.text(
            bar.get_width() + 0.2, bar.get_y() + bar.get_height() / 2,
            f"{pct:.1f}%", va="center", fontsize=7,
        )

    path = d / filename
    fig.savefig(path, bbox_inches="tight", dpi=FIGURE_DPI)
    plt.close(fig)
    logger.info(f"Feature Importance plot saved → {path}")
    return path


# ─────────────────────────────────────────────────────────────────────────────
# Orchestrator
# ─────────────────────────────────────────────────────────────────────────────

class EvaluationArtifacts:
    """
    Orchestrates generating and saving all evaluation figures.

    Example::

        ea = EvaluationArtifacts()
        paths = ea.generate_all(model, feature_names, y_true, y_pred_proba)
    """

    def __init__(self, output_dir: Optional[Path] = None) -> None:
        self.output_dir = _ensure_eval_dir(output_dir)

    def generate_all(
        self,
        model: Any,
        feature_names: List[str],
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        threshold: float = 0.5,
        label: str = "LightGBM",
    ) -> Dict[str, Path]:
        """
        Generate all evaluation figures and return a dict of paths.

        Returns:
            Dict mapping figure name → file path.
        """
        logger.info("Generating evaluation artifacts …")
        paths: Dict[str, Path] = {}

        paths["roc_curve"] = plot_roc_curve(
            y_true, y_pred_proba, label=label, output_dir=self.output_dir
        )
        paths["precision_recall_curve"] = plot_precision_recall_curve(
            y_true, y_pred_proba, label=label, output_dir=self.output_dir
        )
        paths["confusion_matrix"] = plot_confusion_matrix(
            y_true, y_pred_proba, threshold=threshold,
            label=label, output_dir=self.output_dir
        )
        paths["feature_importance"] = plot_feature_importance(
            model, feature_names, output_dir=self.output_dir
        )

        logger.info(f"All evaluation artifacts saved to {self.output_dir}")
        return paths
