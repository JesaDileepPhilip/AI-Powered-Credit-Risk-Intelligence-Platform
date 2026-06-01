"""
app/eda/report_generator.py — EDA report orchestrator.

Orchestrates all EDA components (DataLoader, DataProfiler,
EDAVisualizer, BusinessInsightGenerator) into a single end-to-end
pipeline and produces two output artefacts:

  1. ``documents/eda/eda_report.md``   — Human-readable Markdown report.
  2. ``documents/eda/eda_profile.json`` — Machine-readable profile dict.

Usage:
    python -m app.eda.report_generator

    — or —

    from app.eda.report_generator import EDAReportGenerator

    generator = EDAReportGenerator()
    report_path, json_path = generator.run()
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from app.utils.helpers import make_json_serialisable, timer
from app.utils.logger import get_logger
from config import settings

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# EDAReportGenerator
# ─────────────────────────────────────────────────────────────────────────────


class EDAReportGenerator:
    """
    Orchestrates the full EDA pipeline and writes output files.

    Args:
        output_dir: Directory where ``eda_report.md`` and ``eda_profile.json``
                    are written.  Defaults to ``settings.eda_output_dir``.

    Example::

        gen = EDAReportGenerator()
        report_md, profile_json = gen.run()
        print("Report:", report_md)
    """

    def __init__(self, output_dir: Optional[Path] = None) -> None:
        self.output_dir = output_dir or settings.eda_output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"EDAReportGenerator initialised.  Output: {self.output_dir}")

    # ── Markdown section builders ─────────────────────────────────────────────

    @staticmethod
    def _fmt_num(value: Any) -> str:
        """Format a number for display in the Markdown report."""
        if isinstance(value, float):
            return f"{value:,.4f}" if abs(value) < 1 else f"{value:,.2f}"
        if isinstance(value, int):
            return f"{value:,}"
        return str(value)

    def _section_dimensions(self, dims: Dict[str, int], dup: Dict[str, Any]) -> str:
        return (
            "| Metric | Value |\n"
            "| --- | --- |\n"
            f"| Rows | {dims.get('rows', 0):,} |\n"
            f"| Columns | {dims.get('columns', 0)} |\n"
            f"| Total Cells | {dims.get('total_cells', 0):,} |\n"
            f"| Duplicate Rows | {dup.get('duplicate_rows', 0):,} "
            f"({dup.get('duplicate_pct', 0):.4f}%) |\n"
        )

    def _section_column_types(self, col_types: Dict[str, Dict]) -> str:
        type_counts: Dict[str, int] = {}
        for meta in col_types.values():
            kind = meta.get("kind", "unknown")
            type_counts[kind] = type_counts.get(kind, 0) + 1

        lines = ["| Column Kind | Count |", "| --- | --- |"]
        for kind, count in sorted(type_counts.items()):
            lines.append(f"| {kind} | {count} |")
        return "\n".join(lines)

    def _section_target(self, target_info: Optional[Dict]) -> str:
        if not target_info:
            return "_TARGET column not available._\n"
        return (
            f"- **Default Rate:** {target_info['default_rate']:.4f}%\n"
            f"- **Non-Default (0):** {target_info['counts'].get(0, 0):,} "
            f"({target_info['percentages'].get(0, 0):.2f}%)\n"
            f"- **Default (1):** {target_info['counts'].get(1, 0):,} "
            f"({target_info['percentages'].get(1, 0):.2f}%)\n"
            f"- **Class Imbalance Ratio:** {target_info['imbalance_ratio']:.1f}:1\n"
        )

    def _section_missing_values(self, missing_records: List[Dict]) -> str:
        if not missing_records:
            return "✅ **No missing values detected.**\n"

        lines = [
            "| # | Column | Missing Count | Missing % |",
            "| --- | --- | --- | --- |",
        ]
        for i, rec in enumerate(missing_records[:25], 1):
            lines.append(
                f"| {i} | `{rec['column']}` | "
                f"{rec['missing_count']:,} | {rec['missing_pct']:.2f}% |"
            )
        if len(missing_records) > 25:
            lines.append(
                f"| … | *{len(missing_records) - 25} more columns* | | |"
            )
        return "\n".join(lines)

    def _section_feature_categories(self, categories: Dict[str, List[str]]) -> str:
        lines: List[str] = []
        for cat, features in categories.items():
            if not features:
                continue
            label = cat.replace("_", " ").title()
            feature_str = ", ".join(f"`{f}`" for f in features[:12])
            more = f" *…+{len(features) - 12} more*" if len(features) > 12 else ""
            lines.append(f"\n**{label}** ({len(features)} features)  \n{feature_str}{more}")
        return "\n".join(lines)

    def _section_top_correlations(
        self, correlations: Optional[List[Dict]]
    ) -> str:
        if not correlations:
            return "_Correlation data not available._\n"

        lines = [
            "| Rank | Feature | Correlation | |Corr| | p-value | Significant? |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
        for i, row in enumerate(correlations[:15], 1):
            sig = "✓" if row.get("significant") else "✗"
            lines.append(
                f"| {i} | `{row['feature']}` | {row['correlation']:+.4f} | "
                f"{row['abs_correlation']:.4f} | {row['p_value']:.6f} | {sig} |"
            )
        return "\n".join(lines)

    def _section_insights(self, insights: List[Dict]) -> str:
        icons = {"info": "ℹ️", "warning": "⚠️", "critical": "🚨"}
        lines: List[str] = []
        for i, insight in enumerate(insights, 1):
            icon = icons.get(insight["severity"], "ℹ️")
            lines += [
                f"\n### {i}. {icon} {insight['title']}",
                f"**Category:** {insight['category']}  "
                f"| **Severity:** `{insight['severity']}`",
                f"\n**Finding:**  \n{insight['finding']}",
                f"\n**Recommendation:**  \n> {insight['recommendation']}",
            ]
        return "\n".join(lines)

    def _section_plots(self, plot_paths: Dict[str, Path]) -> str:
        descriptions = {
            "default_distribution":  "Loan Default Class Distribution (counts + pie chart)",
            "income_vs_default":     "Annual Income vs. Default Status (distribution + boxplot)",
            "credit_vs_default":     "Credit Amount vs. Default Status (distribution + scatter)",
            "age_vs_default":        "Applicant Age vs. Default (distribution + rate by age group)",
            "employment_vs_default": "Employment Duration vs. Default (distribution + rate by tenure)",
            "correlation_heatmap":   "Feature Correlation Heatmap (top features by |corr with TARGET|)",
        }
        lines: List[str] = []
        for key, path in plot_paths.items():
            desc = descriptions.get(key, key)
            lines += [f"\n#### {desc}", f"![{desc}]({path.name})"]
        return "\n".join(lines)

    # ── Markdown report ───────────────────────────────────────────────────────

    def generate_markdown_report(
        self,
        profile: Dict[str, Any],
        insights: List[Dict[str, Any]],
        plot_paths: Dict[str, Path],
        datasets_loaded: List[str],
    ) -> Path:
        """
        Write a comprehensive Markdown EDA report to disk.

        Args:
            profile:         Profile dict from ``DataProfiler.run()``.
            insights:        Insights list from ``BusinessInsightGenerator.run()``.
            plot_paths:      Dict of plot_key → Path from ``EDAVisualizer``.
            datasets_loaded: List of dataset names successfully loaded.

        Returns:
            Path to the written Markdown file.
        """
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        dims           = profile.get("dimensions", {})
        dup_stats      = profile.get("duplicate_stats", {})
        target_info    = profile.get("target_distribution")
        col_types      = profile.get("column_types", {})
        missing_recs   = profile.get("missing_value_stats", [])
        categories     = profile.get("feature_categories", {})
        correlations   = profile.get("target_correlations")
        numeric_df_raw = profile.get("numeric_summary", {})

        severity_counts: Dict[str, int] = {}
        for ins in insights:
            s = ins.get("severity", "info")
            severity_counts[s] = severity_counts.get(s, 0) + 1

        lines: List[str] = [
            f"# EDA Report — {settings.app_name}",
            f"\n**Generated:** {ts}  ",
            f"**Platform Version:** {settings.app_version}  ",
            f"**Datasets analysed:** {', '.join(datasets_loaded)}",

            "\n---\n",
            "## 1. Executive Summary",
            f"\n| KPI | Value |",
            "| --- | --- |",
            f"| Primary dataset shape | {dims.get('rows', 0):,} rows × {dims.get('columns', 0)} cols |",
            f"| Total cells | {dims.get('total_cells', 0):,} |",
            f"| Duplicate rows | {dup_stats.get('duplicate_rows', 0):,} ({dup_stats.get('duplicate_pct', 0):.4f}%) |",
            f"| Columns with missing values | {len(missing_recs)} |",
            f"| Datasets loaded | {len(datasets_loaded)} ({', '.join(datasets_loaded)}) |",
            f"| Business insights generated | {len(insights)} |",
            f"| Visualisations saved | {len(plot_paths)} |",
            f"| Critical / Warning / Info insights | "
            f"{severity_counts.get('critical', 0)} / "
            f"{severity_counts.get('warning', 0)} / "
            f"{severity_counts.get('info', 0)} |",

            "\n---\n",
            "## 2. Dataset Overview",
            "\n### 2.1 Dimensions\n",
            self._section_dimensions(dims, dup_stats),
            "\n### 2.2 Column Type Distribution\n",
            self._section_column_types(col_types),

            "\n---\n",
            "## 3. Target Variable Analysis\n",
            self._section_target(target_info),

            "\n---\n",
            "## 4. Missing Values\n",
            self._section_missing_values(missing_recs),

            "\n---\n",
            "## 5. Feature Categories\n",
            self._section_feature_categories(categories),

            "\n---\n",
            "## 6. Top Feature Correlations with TARGET\n",
            self._section_top_correlations(correlations),

            "\n---\n",
            "## 7. Business Insights\n",
            self._section_insights(insights),

            "\n---\n",
            "## 8. Visualisations\n",
            self._section_plots(plot_paths),

            "\n---\n",
            "## 9. Recommendations for Next Phases",
            "\n### Feature Engineering (ML Phase 2)",
            "```python",
            "df['CREDIT_TO_INCOME_RATIO']  = df['AMT_CREDIT']  / df['AMT_INCOME_TOTAL']",
            "df['ANNUITY_TO_INCOME_RATIO'] = df['AMT_ANNUITY'] / df['AMT_INCOME_TOTAL']",
            "df['CREDIT_TERM']             = df['AMT_CREDIT']  / df['AMT_ANNUITY']",
            "df['AGE_YEARS']               = df['DAYS_BIRTH'].abs() / 365.25",
            "df['EMPLOYMENT_YEARS']        = df['DAYS_EMPLOYED'].replace(365243, np.nan).abs() / 365.25",
            "df['IS_EMPLOYED_ANOMALY']     = (df['DAYS_EMPLOYED'] == 365243).astype(int)",
            "df['EXT_SOURCE_MEAN']         = df[['EXT_SOURCE_1','EXT_SOURCE_2','EXT_SOURCE_3']].mean(axis=1)",
            "df['DOC_SUBMISSION_COUNT']    = df[[c for c in df.columns if 'FLAG_DOCUMENT' in c]].sum(axis=1)",
            "```",
            "\n### Preprocessing Checklist",
            "- Drop features with >80% missing values",
            "- Median imputation for numeric; mode for categorical",
            "- Cap income outliers at 99th percentile",
            "- Label-encode categoricals (LightGBM native support)",
            "- Apply `scale_pos_weight` or SMOTE for class imbalance",
            "\n### Model Selection",
            "- **Primary:** LightGBM (StratifiedKFold × 5, scale_pos_weight)",
            "- **Metrics:** AUC-ROC, PR-AUC, KS Statistic, Gini Coefficient",
            "- **Explainability:** TreeSHAP via `shap.TreeExplainer`",

            "\n---\n",
            f"*Report generated by {settings.app_name} v{settings.app_version} on {ts}*",
        ]

        report_content = "\n".join(lines)
        report_path = self.output_dir / settings.eda_report_filename

        with open(report_path, "w", encoding="utf-8") as fh:
            fh.write(report_content)

        logger.info(f"Markdown report saved: {report_path}")
        return report_path

    # ── JSON profile ──────────────────────────────────────────────────────────

    def save_profile_json(self, profile: Dict[str, Any]) -> Path:
        """
        Serialise the profile dict to JSON for downstream consumption.

        Handles all numpy/pandas types via ``make_json_serialisable``.

        Returns:
            Path to the written JSON file.
        """
        json_path = self.output_dir / settings.eda_profile_json_filename
        serialisable = make_json_serialisable(profile)

        with open(json_path, "w", encoding="utf-8") as fh:
            json.dump(serialisable, fh, indent=2, default=str)

        logger.info(f"Profile JSON saved: {json_path}")
        return json_path

    # ── Full pipeline ─────────────────────────────────────────────────────────

    @timer
    def run(self) -> Tuple[Path, Path]:
        """
        Execute the complete EDA pipeline end-to-end.

        Pipeline steps:
          1. Load datasets (DataLoader)
          2. Profile application_train (DataProfiler)
          3. Generate visualisations (EDAVisualizer)
          4. Generate business insights (BusinessInsightGenerator)
          5. Write Markdown report
          6. Write JSON profile

        Returns:
            Tuple of (report_md_path, profile_json_path).
        """
        # Deferred imports to avoid circular dependencies at module level
        from app.eda.data_loader  import DataLoader
        from app.eda.insights     import BusinessInsightGenerator
        from app.eda.profiler     import DataProfiler
        from app.eda.visualizations import EDAVisualizer

        logger.info("=" * 60)
        logger.info("Starting EDA pipeline ...")
        logger.info("=" * 60)

        # ── Step 1: Load data ──────────────────────────────────────────────
        loader   = DataLoader()
        datasets = loader.load_all()
        app_df   = datasets["application_train"]
        bureau   = datasets.get("bureau")

        # Optional: sample for faster EDA when EDA_SAMPLE_SIZE is set
        sample_size = settings.eda_sample_size
        if sample_size and len(app_df) > sample_size:
            logger.info(
                f"Sampling {sample_size:,} rows from {len(app_df):,} for EDA "
                f"(set EDA_SAMPLE_SIZE= to disable sampling)."
            )
            app_df = app_df.sample(n=sample_size, random_state=settings.random_seed)

        # ── Step 2: Profile ────────────────────────────────────────────────
        logger.info("Profiling dataset ...")
        profiler = DataProfiler(app_df, dataset_name="application_train")
        profile  = profiler.run()

        # ── Step 3: Visualise ──────────────────────────────────────────────
        logger.info("Generating visualisations ...")
        viz        = EDAVisualizer(output_dir=self.output_dir)
        plot_paths = viz.generate_all_plots(app_df)

        # ── Step 4: Insights ───────────────────────────────────────────────
        logger.info("Generating business insights ...")
        gen      = BusinessInsightGenerator(app_df, bureau_df=bureau)
        insights = gen.run()

        # ── Step 5 & 6: Write artefacts ───────────────────────────────────
        report_path = self.generate_markdown_report(
            profile         = profile,
            insights        = insights,
            plot_paths      = plot_paths,
            datasets_loaded = loader.get_loaded_names(),
        )
        json_path = self.save_profile_json(profile)

        logger.info("=" * 60)
        logger.info(f"EDA pipeline complete.")
        logger.info(f"  Report  : {report_path}")
        logger.info(f"  Profile : {json_path}")
        logger.info(f"  Plots   : {self.output_dir}")
        logger.info("=" * 60)

        return report_path, json_path


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry-point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    settings.ensure_directories()
    generator = EDAReportGenerator()
    report, profile = generator.run()
    print(f"\n✅  EDA complete.")
    print(f"    Report  → {report}")
    print(f"    Profile → {profile}")
