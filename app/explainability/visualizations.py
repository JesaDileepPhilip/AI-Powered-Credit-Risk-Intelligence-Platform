"""
app/explainability/visualizations.py — SHAP plot generators.

Generates and saves:
  Global:
    - shap_summary.png          (beeswarm / dot summary plot)
    - shap_feature_importance.png (bar plot of mean |SHAP|)
    - global_feature_importance.csv

  Local (per-customer):
    - waterfall_{customer_id}.png
    - force_{customer_id}.png

All files are saved to:
  documents/explainability/           (global)
  documents/explainability/local/     (local)

Design:
  - Uses matplotlib Agg backend (headless-safe, no display required)
  - All plots are self-contained and closed after saving
  - CSV outputs use pandas for portability
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from app.utils.logger import get_logger
from config import settings

logger = get_logger(__name__)

# ── Output directories ────────────────────────────────────────────────────────
EXPLAINABILITY_DIR: Path = settings.documents_dir / "explainability"
LOCAL_EXPLAINABILITY_DIR: Path = EXPLAINABILITY_DIR / "local"

FIGURE_DPI: int = 150
MAX_DISPLAY_FEATURES: int = 20


def _ensure_dirs(*dirs: Path) -> None:
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Global visualizations
# ─────────────────────────────────────────────────────────────────────────────

def plot_shap_summary(
    shap_values: np.ndarray,
    X: np.ndarray,
    feature_names: List[str],
    output_dir: Optional[Path] = None,
    filename: str = "shap_summary.png",
    max_display: int = MAX_DISPLAY_FEATURES,
) -> Path:
    """
    Generate and save a SHAP beeswarm summary plot.

    The beeswarm plot shows every sample's SHAP value for the top features,
    colour-coded by the actual feature value (blue=low, red=high).

    Args:
        shap_values:   (n_samples, n_features) SHAP value array.
        X:             (n_samples, n_features) raw feature matrix.
        feature_names: List of feature name strings.
        output_dir:    Directory to save the plot (defaults to EXPLAINABILITY_DIR).
        filename:      Output file name.
        max_display:   Maximum number of features shown.

    Returns:
        Path to the saved figure.
    """
    import shap

    d = output_dir or EXPLAINABILITY_DIR
    _ensure_dirs(d)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fig, ax = plt.subplots(figsize=(12, max(6, max_display * 0.45)), dpi=FIGURE_DPI)
        shap.summary_plot(
            shap_values,
            X,
            feature_names=feature_names,
            max_display=max_display,
            show=False,
            plot_type="dot",
            plot_size=None,
        )
        plt.title(
            "SHAP Summary Plot — Credit Default Risk Drivers",
            fontsize=14, fontweight="bold", pad=12,
        )
        plt.tight_layout()

    path = d / filename
    plt.savefig(path, bbox_inches="tight", dpi=FIGURE_DPI)
    plt.close("all")
    logger.info(f"SHAP Summary Plot saved → {path}")
    return path


def plot_shap_feature_importance(
    mean_abs_shap: np.ndarray,
    feature_names: List[str],
    output_dir: Optional[Path] = None,
    filename: str = "shap_feature_importance.png",
    top_n: int = MAX_DISPLAY_FEATURES,
) -> Path:
    """
    Generate and save a horizontal bar chart of mean |SHAP| feature importance.

    Args:
        mean_abs_shap: (n_features,) array of mean absolute SHAP values.
        feature_names: List of feature names.
        output_dir:    Directory to save the plot.
        filename:      Output file name.
        top_n:         Number of top features to display.

    Returns:
        Path to the saved figure.
    """
    import shap

    d = output_dir or EXPLAINABILITY_DIR
    _ensure_dirs(d)

    order = np.argsort(mean_abs_shap)[::-1][:top_n]
    names = [feature_names[i] for i in order]
    vals = mean_abs_shap[order]

    fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.38)), dpi=FIGURE_DPI)
    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.9, top_n))

    bars = ax.barh(range(top_n), vals[::-1], color=colors[::-1], edgecolor="none")
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(names[::-1], fontsize=9)
    ax.set_xlabel("Mean |SHAP Value| (impact on model output)", fontsize=11)
    ax.set_title(
        f"SHAP Feature Importance — Top {top_n} Features",
        fontsize=14, fontweight="bold",
    )
    ax.grid(axis="x", alpha=0.3)

    # Annotate values
    for bar, val in zip(bars, vals[::-1]):
        ax.text(
            bar.get_width() + max(vals) * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.4f}",
            va="center", fontsize=7.5,
        )

    plt.tight_layout()
    path = d / filename
    fig.savefig(path, bbox_inches="tight", dpi=FIGURE_DPI)
    plt.close(fig)
    logger.info(f"SHAP Feature Importance Plot saved → {path}")
    return path


def save_global_feature_importance_csv(
    mean_abs_shap: np.ndarray,
    feature_names: List[str],
    output_dir: Optional[Path] = None,
    filename: str = "global_feature_importance.csv",
) -> Path:
    """
    Save a CSV of global SHAP feature importances ranked by mean |SHAP|.

    Columns: rank, feature, mean_abs_shap_value, cumulative_importance_pct

    Returns:
        Path to the saved CSV.
    """
    d = output_dir or EXPLAINABILITY_DIR
    _ensure_dirs(d)

    order = np.argsort(mean_abs_shap)[::-1]
    total = mean_abs_shap.sum() or 1.0
    cumulative = 0.0
    rows = []
    for rank, i in enumerate(order, 1):
        cumulative += mean_abs_shap[i]
        rows.append({
            "rank": rank,
            "feature": feature_names[i],
            "mean_abs_shap_value": round(float(mean_abs_shap[i]), 6),
            "cumulative_importance_pct": round(cumulative / total * 100, 2),
        })

    df = pd.DataFrame(rows)
    path = d / filename
    df.to_csv(path, index=False)
    logger.info(f"Global Feature Importance CSV saved → {path} ({len(df)} features)")
    return path


# ─────────────────────────────────────────────────────────────────────────────
# Local visualizations
# ─────────────────────────────────────────────────────────────────────────────

def plot_waterfall(
    shap_explainer: Any,
    shap_values: np.ndarray,
    feature_names: List[str],
    feature_values: List[float],
    expected_value: float,
    customer_id: str = "customer",
    output_dir: Optional[Path] = None,
    max_display: int = 15,
) -> Path:
    """
    Generate and save a SHAP waterfall plot for a single customer.

    A waterfall chart shows how each feature pushes the prediction up or down
    from the base value (expected_value) to the final prediction.

    Args:
        shap_explainer:  The fitted shap.TreeExplainer (for Explanation object).
        shap_values:     (n_features,) SHAP value array for this customer.
        feature_names:   Feature name list.
        feature_values:  Actual feature value list for this customer.
        expected_value:  Base value / model expected output.
        customer_id:     String identifier used in the filename.
        output_dir:      Directory to save the plot.
        max_display:     Maximum number of features to show.

    Returns:
        Path to the saved figure.
    """
    import shap

    d = output_dir or LOCAL_EXPLAINABILITY_DIR
    _ensure_dirs(d)

    explanation = shap.Explanation(
        values=shap_values,
        base_values=expected_value,
        data=np.array(feature_values),
        feature_names=feature_names,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fig, ax = plt.subplots(figsize=(12, max(6, max_display * 0.5)), dpi=FIGURE_DPI)
        shap.plots.waterfall(
            explanation,
            max_display=max_display,
            show=False,
        )
        plt.title(
            f"SHAP Waterfall Plot — Customer {customer_id}",
            fontsize=13, fontweight="bold", pad=12,
        )
        plt.tight_layout()

    filename = f"waterfall_{customer_id}.png"
    path = d / filename
    plt.savefig(path, bbox_inches="tight", dpi=FIGURE_DPI)
    plt.close("all")
    logger.info(f"Waterfall Plot saved → {path}")
    return path


def plot_force(
    shap_explainer: Any,
    shap_values: np.ndarray,
    feature_names: List[str],
    feature_values: List[float],
    expected_value: float,
    customer_id: str = "customer",
    output_dir: Optional[Path] = None,
) -> Path:
    """
    Generate and save a SHAP force plot for a single customer (saved as PNG).

    The force plot shows the cumulative effect of all features pushing the
    prediction left (decrease) or right (increase) from the base value.

    Returns:
        Path to the saved figure.
    """
    import shap

    d = output_dir or LOCAL_EXPLAINABILITY_DIR
    _ensure_dirs(d)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        shap.initjs()
        force_plot = shap.force_plot(
            base_value=expected_value,
            shap_values=shap_values,
            features=np.array(feature_values),
            feature_names=feature_names,
            matplotlib=True,
            show=False,
            figsize=(20, 3),
        )

    filename = f"force_{customer_id}.png"
    path = d / filename
    plt.savefig(path, bbox_inches="tight", dpi=FIGURE_DPI)
    plt.close("all")
    logger.info(f"Force Plot saved → {path}")
    return path


def save_feature_contributions_csv(
    shap_values: np.ndarray,
    feature_names: List[str],
    feature_values: List[float],
    output_dir: Optional[Path] = None,
    filename: str = "feature_contributions.csv",
) -> Path:
    """
    Save a per-feature SHAP contribution table for a single customer.

    Columns:
      - feature
      - feature_value
      - shap_value
      - impact_direction  (Increases Risk | Decreases Risk | Neutral)
      - impact_strength   (Strong | Moderate | Weak)

    Args:
        shap_values:   (n_features,) array of SHAP values.
        feature_names: List of feature name strings.
        feature_values: List of raw feature values for the customer.
        output_dir:    Directory to save the CSV.
        filename:      Output file name.

    Returns:
        Path to the saved CSV.
    """
    d = output_dir or LOCAL_EXPLAINABILITY_DIR
    _ensure_dirs(d)

    abs_vals = np.abs(shap_values)
    max_abs = abs_vals.max() or 1.0

    def _direction(sv: float) -> str:
        if sv > 1e-6:
            return "Increases Risk"
        if sv < -1e-6:
            return "Decreases Risk"
        return "Neutral"

    def _strength(sv: float, max_v: float) -> str:
        ratio = abs(sv) / max_v
        if ratio >= 0.5:
            return "Strong"
        if ratio >= 0.2:
            return "Moderate"
        return "Weak"

    order = np.argsort(abs_vals)[::-1]
    rows = [
        {
            "feature": feature_names[i],
            "feature_value": round(float(feature_values[i]), 6),
            "shap_value": round(float(shap_values[i]), 6),
            "impact_direction": _direction(shap_values[i]),
            "impact_strength": _strength(shap_values[i], max_abs),
        }
        for i in order
    ]

    df = pd.DataFrame(rows)
    path = d / filename
    df.to_csv(path, index=False)
    logger.info(f"Feature Contributions CSV saved → {path}")
    return path
