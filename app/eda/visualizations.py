"""
app/eda/visualizations.py — EDA visualisation module.

Generates and saves all EDA plots as PNG files using Matplotlib and Seaborn.
Every method returns the absolute ``Path`` of the saved file.

Plots generated:
  1. Default distribution (bar + pie)
  2. Income vs Default (histogram + boxplot)
  3. Credit Amount vs Default (histogram + scatter)
  4. Age vs Default (histogram + default rate by age group)
  5. Employment Length vs Default (histogram + default rate by tenure)
  6. Correlation Heatmap (top N features by |corr with TARGET|)

All plots use a consistent dark theme for professional presentation.
The output directory is created automatically.

Usage:
    from app.eda.visualizations import EDAVisualizer

    viz    = EDAVisualizer()
    paths  = viz.generate_all_plots(df)
    # paths is a dict: plot_name -> Path
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
# Set the Agg (non-interactive) backend only when no backend has been configured yet.
# This prevents overriding the backend set by Streamlit or other GUI frameworks
# that may import this module in later phases.
try:
    import matplotlib.pyplot  # noqa: F401 — probe whether a backend is already loaded
except Exception:
    pass  # backend not yet configured; Agg will be set below

if matplotlib.get_backend().lower() in ("", "agg", "module://matplotlib_inline.backend_inline"):
    # No interactive backend is active; use the headless Agg backend
    matplotlib.use("Agg")
# If an interactive backend (e.g., TkAgg, WebAgg from Streamlit) is already
# loaded, we leave it intact.

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

from app.utils.logger import get_logger
from config import settings

logger = get_logger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Colour constants
# ─────────────────────────────────────────────────────────────────────────────

_C_NON_DEFAULT = "#2ECC71"   # green — no default
_C_DEFAULT     = "#E74C3C"   # red   — default
_C_ACCENT      = "#FFD700"   # gold  — reference lines
_C_BG          = "#0D1117"   # near-black background
_C_PANEL       = "#161B22"   # slightly lighter panel background
_C_GRID        = "#21262D"   # subtle grid lines
_C_TEXT        = "#E6EDF3"   # primary text colour


# ─────────────────────────────────────────────────────────────────────────────
# Theme helpers
# ─────────────────────────────────────────────────────────────────────────────

def _apply_dark_theme() -> None:
    """Apply the project-wide dark matplotlib theme."""
    plt.rcParams.update(
        {
            "figure.facecolor":    _C_BG,
            "axes.facecolor":      _C_PANEL,
            "axes.edgecolor":      _C_GRID,
            "axes.labelcolor":     _C_TEXT,
            "axes.titlecolor":     _C_TEXT,
            "text.color":          _C_TEXT,
            "xtick.color":         _C_TEXT,
            "ytick.color":         _C_TEXT,
            "grid.color":          _C_GRID,
            "grid.alpha":          0.6,
            "grid.linestyle":      "--",
            "legend.facecolor":    _C_PANEL,
            "legend.edgecolor":    _C_GRID,
            "legend.labelcolor":   _C_TEXT,
            "font.family":         "DejaVu Sans",
            "font.size":           10,
            "axes.titlesize":      13,
            "axes.labelsize":      11,
        }
    )


def _usd_formatter(x: float, _pos: int) -> str:
    """Format axis tick values as abbreviated USD amounts."""
    if abs(x) >= 1_000_000:
        return f"${x/1_000_000:.1f}M"
    if abs(x) >= 1_000:
        return f"${x/1_000:.0f}K"
    return f"${x:.0f}"


def _pct_formatter(x: float, _pos: int) -> str:
    return f"{x:.1f}%"


# ─────────────────────────────────────────────────────────────────────────────
# EDAVisualizer
# ─────────────────────────────────────────────────────────────────────────────


class EDAVisualizer:
    """
    Generates and saves EDA visualisations using Matplotlib and Seaborn.

    Args:
        output_dir: Directory where PNG files are saved.
                    Defaults to ``settings.eda_output_dir``.

    Example::

        viz   = EDAVisualizer()
        paths = viz.generate_all_plots(df)
    """

    def __init__(self, output_dir: Optional[Path] = None) -> None:
        self.output_dir: Path = output_dir or settings.eda_output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.dpi: int = settings.eda_figure_dpi
        self._saved_plots: List[Path] = []

        sns.set_theme(style=settings.eda_figure_style)
        _apply_dark_theme()

        logger.info(f"EDAVisualizer initialised. Output: {self.output_dir}")

    # ── Private helpers ───────────────────────────────────────────────────────

    def _save_fig(self, fig: plt.Figure, filename: str) -> Path:
        """Save *fig* to *filename* inside output_dir and close it."""
        filepath = self.output_dir / filename
        fig.savefig(
            filepath,
            dpi=self.dpi,
            bbox_inches="tight",
            facecolor=fig.get_facecolor(),
        )
        plt.close(fig)
        self._saved_plots.append(filepath)
        logger.info(f"Saved: {filepath}")
        return filepath

    @staticmethod
    def _style_axes(ax: plt.Axes, title: str, xlabel: str, ylabel: str) -> None:
        """Apply consistent styling to an Axes object."""
        ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
        ax.set_xlabel(xlabel, fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.tick_params(axis="both", labelsize=9)
        ax.spines[["top", "right"]].set_visible(False)

    @staticmethod
    def _require_columns(df: pd.DataFrame, *columns: str) -> None:
        """Raise ValueError if any required column is missing."""
        missing = [c for c in columns if c not in df.columns]
        if missing:
            raise ValueError(
                f"DataFrame is missing required column(s): {missing}"
            )

    @staticmethod
    def _get_age_series(df: pd.DataFrame) -> pd.Series:
        """Derive age in years from DAYS_BIRTH (always negative in source data)."""
        return (df["DAYS_BIRTH"].abs() / 365.25).round(1)

    @staticmethod
    def _get_employment_years(df: pd.DataFrame) -> pd.Series:
        """
        Derive employment duration in years from DAYS_EMPLOYED.

        The sentinel value 365 243 encodes unemployed / retired applicants
        and is replaced with NaN.
        """
        emp = df["DAYS_EMPLOYED"].copy().astype(float)
        emp = emp.replace(365_243, np.nan)  # return-value form; inplace on copy is deprecated
        return (emp.abs() / 365.25).round(1)

    # ── Plot 1: Default Distribution ─────────────────────────────────────────

    def plot_default_distribution(self, df: pd.DataFrame) -> Path:
        """
        Side-by-side bar chart and pie chart of TARGET class distribution.

        Args:
            df: DataFrame containing ``TARGET`` column.

        Returns:
            Path to the saved PNG file.
        """
        self._require_columns(df, "TARGET")

        target = df["TARGET"].dropna()
        counts = target.value_counts().sort_index()
        labels = ["Non-Default (0)", "Default (1)"]
        colors = [_C_NON_DEFAULT, _C_DEFAULT]
        default_rate = float(target.mean() * 100)

        fig, (ax_bar, ax_pie) = plt.subplots(1, 2, figsize=(14, 6))
        fig.patch.set_facecolor(_C_BG)
        fig.suptitle(
            "Loan Default Class Distribution",
            fontsize=16, fontweight="bold", color=_C_TEXT, y=1.01,
        )

        # ── Bar chart ──
        bars = ax_bar.bar(
            labels, counts.values,
            color=colors, edgecolor=_C_GRID, linewidth=1.0, width=0.45,
        )
        for bar, val in zip(bars, counts.values):
            ax_bar.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + counts.max() * 0.01,
                f"{val:,}",
                ha="center", va="bottom", fontsize=11,
                color=_C_TEXT, fontweight="bold",
            )
        self._style_axes(ax_bar, "Count by Class", "Loan Status", "Number of Applications")
        ax_bar.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
        ax_bar.set_ylim(0, counts.max() * 1.12)

        # ── Pie chart ──
        wedges, texts, autotexts = ax_pie.pie(
            counts.values,
            labels=labels,
            colors=colors,
            autopct="%1.2f%%",
            startangle=140,
            wedgeprops={"edgecolor": _C_BG, "linewidth": 2},
            textprops={"color": _C_TEXT, "fontsize": 10},
        )
        for at in autotexts:
            at.set_fontsize(12)
            at.set_fontweight("bold")
        ax_pie.set_title(
            f"Class Proportion  (Default Rate: {default_rate:.2f}%)",
            fontsize=13, fontweight="bold", color=_C_TEXT, pad=10,
        )

        fig.tight_layout()
        return self._save_fig(fig, "01_default_distribution.png")

    # ── Plot 2: Income vs Default ─────────────────────────────────────────────

    def plot_income_vs_default(self, df: pd.DataFrame) -> Path:
        """
        Overlapping histograms and box plots of AMT_INCOME_TOTAL by TARGET.

        Income is capped at the 99th percentile to suppress extreme outliers.

        Returns:
            Path to the saved PNG file.
        """
        self._require_columns(df, "AMT_INCOME_TOTAL", "TARGET")

        plot_df = df[["AMT_INCOME_TOTAL", "TARGET"]].dropna().copy()
        cap = float(plot_df["AMT_INCOME_TOTAL"].quantile(
            settings.eda_outlier_cap_percentile
        ))
        plot_df = plot_df[plot_df["AMT_INCOME_TOTAL"] <= cap].copy()
        plot_df["Default Status"] = plot_df["TARGET"].map(
            {0: "Non-Default", 1: "Default"}
        )

        fig, (ax_hist, ax_box) = plt.subplots(1, 2, figsize=(16, 6))
        fig.patch.set_facecolor(_C_BG)
        fig.suptitle(
            "Annual Income vs. Loan Default",
            fontsize=16, fontweight="bold", color=_C_TEXT, y=1.01,
        )

        # ── Histogram (density) ──
        for tgt, color, label in [
            (0, _C_NON_DEFAULT, "Non-Default"),
            (1, _C_DEFAULT,     "Default"),
        ]:
            ax_hist.hist(
                plot_df.loc[plot_df["TARGET"] == tgt, "AMT_INCOME_TOTAL"],
                bins=70, alpha=0.65, color=color, label=label,
                density=True, edgecolor="none",
            )
        self._style_axes(ax_hist, "Income Distribution", "Annual Income (USD)", "Density")
        ax_hist.xaxis.set_major_formatter(mticker.FuncFormatter(_usd_formatter))
        ax_hist.legend(fontsize=10)

        # ── Box plot ──
        palette = {"Non-Default": _C_NON_DEFAULT, "Default": _C_DEFAULT}
        sns.boxplot(
            data=plot_df, x="Default Status", y="AMT_INCOME_TOTAL",
            palette=palette, ax=ax_box,
            width=0.45, linewidth=1.2, flierprops={"markersize": 3, "alpha": 0.4},
        )
        self._style_axes(ax_box, "Income Spread by Class", "Default Status", "Annual Income (USD)")
        ax_box.yaxis.set_major_formatter(mticker.FuncFormatter(_usd_formatter))

        # Annotate medians
        for i, tgt in enumerate([0, 1]):
            median = plot_df.loc[plot_df["TARGET"] == tgt, "AMT_INCOME_TOTAL"].median()
            ax_box.text(
                i, median * 1.02,
                f"Median: {_usd_formatter(median, 0)}",
                ha="center", va="bottom", fontsize=9, color=_C_ACCENT, fontweight="bold",
            )

        fig.tight_layout()
        return self._save_fig(fig, "02_income_vs_default.png")

    # ── Plot 3: Credit Amount vs Default ──────────────────────────────────────

    def plot_credit_amount_vs_default(self, df: pd.DataFrame) -> Path:
        """
        Histogram of AMT_CREDIT and an AMT_CREDIT vs AMT_ANNUITY scatter
        (sampled for performance), both coloured by TARGET class.

        Returns:
            Path to the saved PNG file.
        """
        self._require_columns(df, "AMT_CREDIT", "TARGET")

        plot_df = df[["AMT_CREDIT", "AMT_ANNUITY", "TARGET"]].dropna().copy()

        fig, (ax_hist, ax_scatter) = plt.subplots(1, 2, figsize=(16, 6))
        fig.patch.set_facecolor(_C_BG)
        fig.suptitle(
            "Credit Amount vs. Loan Default",
            fontsize=16, fontweight="bold", color=_C_TEXT, y=1.01,
        )

        # ── Histogram ──
        for tgt, color, label in [
            (0, _C_NON_DEFAULT, "Non-Default"),
            (1, _C_DEFAULT,     "Default"),
        ]:
            ax_hist.hist(
                plot_df.loc[plot_df["TARGET"] == tgt, "AMT_CREDIT"],
                bins=70, alpha=0.65, color=color, label=label,
                density=True, edgecolor="none",
            )
        self._style_axes(ax_hist, "Credit Amount Distribution", "Credit Amount (USD)", "Density")
        ax_hist.xaxis.set_major_formatter(mticker.FuncFormatter(_usd_formatter))
        ax_hist.legend(fontsize=10)

        # ── Scatter (AMT_CREDIT vs AMT_ANNUITY, sampled) ──
        sample_n = min(5_000, len(plot_df))
        sample_df = plot_df.sample(n=sample_n, random_state=settings.random_seed)

        for tgt, color, label in [
            (0, _C_NON_DEFAULT, "Non-Default"),
            (1, _C_DEFAULT,     "Default"),
        ]:
            subset = sample_df[sample_df["TARGET"] == tgt]
            ax_scatter.scatter(
                subset["AMT_CREDIT"], subset["AMT_ANNUITY"],
                alpha=0.25, color=color, label=label, s=10, linewidths=0,
            )
        self._style_axes(
            ax_scatter,
            f"Credit vs. Annuity (n={sample_n:,} sample)",
            "Credit Amount (USD)", "Annuity Amount (USD)",
        )
        ax_scatter.xaxis.set_major_formatter(mticker.FuncFormatter(_usd_formatter))
        ax_scatter.yaxis.set_major_formatter(mticker.FuncFormatter(_usd_formatter))
        ax_scatter.legend(fontsize=10)

        fig.tight_layout()
        return self._save_fig(fig, "03_credit_amount_vs_default.png")

    # ── Plot 4: Age vs Default ────────────────────────────────────────────────

    def plot_age_vs_default(self, df: pd.DataFrame) -> Path:
        """
        Age distribution overlay + default rate bar chart by 5-year age bins.

        Age is derived from DAYS_BIRTH.

        Returns:
            Path to the saved PNG file.
        """
        self._require_columns(df, "DAYS_BIRTH", "TARGET")

        age = self._get_age_series(df)
        plot_df = pd.DataFrame(
            {"Age": age, "TARGET": df["TARGET"].values}
        ).dropna()
        plot_df["Age_Bin"] = pd.cut(
            plot_df["Age"],
            bins=list(range(20, 75, 5)),
            right=False,
        )

        fig, (ax_hist, ax_bar) = plt.subplots(1, 2, figsize=(16, 6))
        fig.patch.set_facecolor(_C_BG)
        fig.suptitle(
            "Applicant Age vs. Loan Default",
            fontsize=16, fontweight="bold", color=_C_TEXT, y=1.01,
        )

        # ── Density histogram ──
        for tgt, color, label in [
            (0, _C_NON_DEFAULT, "Non-Default"),
            (1, _C_DEFAULT,     "Default"),
        ]:
            ax_hist.hist(
                plot_df.loc[plot_df["TARGET"] == tgt, "Age"],
                bins=40, alpha=0.65, color=color, label=label,
                density=True, edgecolor="none",
            )
        self._style_axes(ax_hist, "Age Distribution", "Age (years)", "Density")
        ax_hist.legend(fontsize=10)

        # ── Default rate by age bin ──
        bin_stats = (
            plot_df.groupby("Age_Bin", observed=True)["TARGET"]
            .agg(default_rate="mean", count="count")
            .reset_index()
        )
        bin_stats["default_rate_pct"] = bin_stats["default_rate"] * 100
        bin_labels = [str(b) for b in bin_stats["Age_Bin"]]
        overall_rate = float(plot_df["TARGET"].mean() * 100)

        bar_colors = sns.color_palette("coolwarm_r", len(bin_stats))
        bars = ax_bar.bar(
            bin_labels, bin_stats["default_rate_pct"],
            color=bar_colors, edgecolor=_C_GRID, linewidth=0.8,
        )
        ax_bar.axhline(
            y=overall_rate, color=_C_ACCENT,
            linestyle="--", linewidth=1.5,
            label=f"Overall ({overall_rate:.2f}%)",
        )
        self._style_axes(ax_bar, "Default Rate by Age Group", "Age Group (years)", "Default Rate (%)")
        ax_bar.yaxis.set_major_formatter(mticker.FuncFormatter(_pct_formatter))
        ax_bar.tick_params(axis="x", rotation=45)
        ax_bar.legend(fontsize=10)

        fig.tight_layout()
        return self._save_fig(fig, "04_age_vs_default.png")

    # ── Plot 5: Employment Length vs Default ──────────────────────────────────

    def plot_employment_vs_default(self, df: pd.DataFrame) -> Path:
        """
        Employment duration distribution overlay + default rate by tenure bands.

        Handles the DAYS_EMPLOYED sentinel (365 243) by converting it to NaN.

        Returns:
            Path to the saved PNG file.
        """
        self._require_columns(df, "DAYS_EMPLOYED", "TARGET")

        emp_years = self._get_employment_years(df)
        cap = float(emp_years.quantile(0.98))
        plot_df = pd.DataFrame(
            {
                "Employment_Years": emp_years.clip(upper=cap).values,
                "TARGET": df["TARGET"].values,
            }
        ).dropna()
        plot_df["Emp_Bin"] = pd.cut(
            plot_df["Employment_Years"],
            bins=[0, 1, 3, 5, 10, 20, float("inf")],
            labels=["<1yr", "1-3yr", "3-5yr", "5-10yr", "10-20yr", "20+yr"],
            right=False,
        )

        fig, (ax_hist, ax_bar) = plt.subplots(1, 2, figsize=(16, 6))
        fig.patch.set_facecolor(_C_BG)
        fig.suptitle(
            "Employment Duration vs. Loan Default",
            fontsize=16, fontweight="bold", color=_C_TEXT, y=1.01,
        )

        # ── Density histogram ──
        for tgt, color, label in [
            (0, _C_NON_DEFAULT, "Non-Default"),
            (1, _C_DEFAULT,     "Default"),
        ]:
            ax_hist.hist(
                plot_df.loc[plot_df["TARGET"] == tgt, "Employment_Years"],
                bins=50, alpha=0.65, color=color, label=label,
                density=True, edgecolor="none",
            )
        self._style_axes(ax_hist, "Employment Duration Distribution", "Years Employed", "Density")
        ax_hist.legend(fontsize=10)

        # ── Default rate by tenure band ──
        bin_stats = (
            plot_df.groupby("Emp_Bin", observed=True)["TARGET"]
            .agg(default_rate="mean", count="count")
            .reset_index()
        )
        bin_stats["default_rate_pct"] = bin_stats["default_rate"] * 100
        overall_rate = float(plot_df["TARGET"].mean() * 100)

        bar_colors = sns.color_palette("RdYlGn_r", len(bin_stats))
        ax_bar.bar(
            bin_stats["Emp_Bin"].astype(str),
            bin_stats["default_rate_pct"],
            color=bar_colors, edgecolor=_C_GRID, linewidth=0.8,
        )
        ax_bar.axhline(
            y=overall_rate, color=_C_ACCENT,
            linestyle="--", linewidth=1.5,
            label=f"Overall ({overall_rate:.2f}%)",
        )
        self._style_axes(
            ax_bar, "Default Rate by Employment Length",
            "Employment Duration", "Default Rate (%)",
        )
        ax_bar.yaxis.set_major_formatter(mticker.FuncFormatter(_pct_formatter))
        ax_bar.tick_params(axis="x", rotation=30)
        ax_bar.legend(fontsize=10)

        fig.tight_layout()
        return self._save_fig(fig, "05_employment_vs_default.png")

    # ── Plot 6: Correlation Heatmap ───────────────────────────────────────────

    def plot_correlation_heatmap(
        self,
        df: pd.DataFrame,
        top_n: Optional[int] = None,
    ) -> Path:
        """
        Lower-triangle Pearson correlation heatmap for the top-N numeric features.

        Features are ranked by |Pearson correlation with TARGET| so the most
        predictive features appear.  TARGET is always included.

        Args:
            df:    DataFrame (must contain ``TARGET``).
            top_n: Override for number of features.
                   Defaults to ``settings.eda_correlation_top_n``.

        Returns:
            Path to the saved PNG file.
        """
        self._require_columns(df, "TARGET")

        _top_n = top_n if top_n is not None else settings.eda_correlation_top_n

        numeric_df = df.select_dtypes(include=[np.number]).drop(
            columns=[c for c in ("SK_ID_CURR",) if c in df.columns],
            errors="ignore",
        )

        # Rank features by |correlation with TARGET|
        if "TARGET" in numeric_df.columns:
            corr_with_target = (
                numeric_df.corr()["TARGET"]
                .drop("TARGET", errors="ignore")
                .abs()
            )
            top_features = corr_with_target.nlargest(_top_n - 1).index.tolist()
            top_features = ["TARGET"] + top_features
        else:
            top_features = numeric_df.columns[:_top_n].tolist()

        corr_matrix = numeric_df[top_features].corr()

        # Lower triangle mask
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)

        fig_size = max(16, len(top_features) * 0.55)
        fig, ax = plt.subplots(figsize=(fig_size, fig_size * 0.9))
        fig.patch.set_facecolor(_C_BG)

        sns.heatmap(
            corr_matrix,
            mask=mask,
            annot=False,
            cmap="coolwarm",
            center=0, vmin=-1, vmax=1,
            ax=ax,
            linewidths=0.4,
            linecolor=_C_BG,
            cbar_kws={
                "shrink": 0.75,
                "label": "Pearson Correlation",
                "orientation": "vertical",
            },
            square=True,
        )

        ax.set_title(
            f"Feature Correlation Heatmap  "
            f"(Top {len(top_features)} features ranked by |corr with TARGET|)",
            fontsize=14, fontweight="bold", color=_C_TEXT, pad=12,
        )
        ax.tick_params(axis="both", labelsize=7)

        fig.tight_layout()
        return self._save_fig(fig, "06_correlation_heatmap.png")

    # ── Batch generator ───────────────────────────────────────────────────────

    def generate_all_plots(self, df: pd.DataFrame) -> Dict[str, Path]:
        """
        Generate all six EDA plots in sequence.

        Each plot is attempted independently; a failed plot logs an error and
        is excluded from the returned dict so the pipeline continues.

        Args:
            df: Primary application training DataFrame.

        Returns:
            Dict mapping plot key → absolute Path of saved PNG.
        """
        plot_methods = [
            ("default_distribution",  self.plot_default_distribution),
            ("income_vs_default",      self.plot_income_vs_default),
            ("credit_vs_default",      self.plot_credit_amount_vs_default),
            ("age_vs_default",         self.plot_age_vs_default),
            ("employment_vs_default",  self.plot_employment_vs_default),
            ("correlation_heatmap",    self.plot_correlation_heatmap),
        ]

        results: Dict[str, Path] = {}

        for plot_key, method in plot_methods:
            try:
                logger.info(f"Generating plot: {plot_key} ...")
                path = method(df)
                results[plot_key] = path
            except Exception as exc:
                logger.error(
                    f"Failed to generate plot '{plot_key}': {exc}",
                    exc_info=True,
                )

        logger.info(
            f"Plot generation complete: {len(results)} / {len(plot_methods)} succeeded."
        )
        return results

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def saved_plots(self) -> List[Path]:
        """List of all file paths saved during this session."""
        return list(self._saved_plots)

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"EDAVisualizer(output_dir={self.output_dir!r}, "
            f"plots_saved={len(self._saved_plots)})"
        )
