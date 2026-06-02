"""
app/explainability/report_generator.py — Explainability Markdown report orchestrator.

Generates:
  documents/explainability_report.md

Report sections:
  1. Executive Summary
  2. Global Feature Importance (top 20 by mean |SHAP|)
  3. Top Risk Drivers
  4. Sample Customer Explanation
  5. Interpretation Guidance
  6. Artifact Index

Usage::

    from app.explainability.report_generator import ExplainabilityReportGenerator

    gen = ExplainabilityReportGenerator()
    gen.run(X_background, feature_names, model, sample_record)
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from app.explainability.shap_explainer import SHAPExplainer
from app.explainability.visualizations import (
    EXPLAINABILITY_DIR,
    LOCAL_EXPLAINABILITY_DIR,
    plot_shap_feature_importance,
    plot_shap_summary,
    save_feature_contributions_csv,
    save_global_feature_importance_csv,
    plot_waterfall,
    plot_force,
)
from app.utils.logger import get_logger
from config import settings

logger = get_logger(__name__)

REPORT_FILENAME = "explainability_report.md"


class ExplainabilityReportGenerator:
    """
    Orchestrates global + local SHAP analysis and produces a Markdown report.

    Args:
        explainer:      A fitted SHAPExplainer instance.
        output_dir:     Directory for the Markdown report (defaults to documents/).
        eval_dir:       Directory for global plots (defaults to documents/explainability/).
        local_dir:      Directory for local plots (defaults to documents/explainability/local/).
    """

    def __init__(
        self,
        explainer: SHAPExplainer,
        output_dir: Optional[Path] = None,
        eval_dir: Optional[Path] = None,
        local_dir: Optional[Path] = None,
    ) -> None:
        self.explainer = explainer
        self.output_dir = output_dir or settings.documents_dir
        self.eval_dir = eval_dir or EXPLAINABILITY_DIR
        self.local_dir = local_dir or LOCAL_EXPLAINABILITY_DIR

        for d in (self.output_dir, self.eval_dir, self.local_dir):
            d.mkdir(parents=True, exist_ok=True)

    # ── Global analysis ───────────────────────────────────────────────────────

    def run_global(
        self,
        X_background: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Run global SHAP analysis: compute values, generate plots, save CSV.

        Args:
            X_background: 2-D feature matrix used as the background dataset.

        Returns:
            Dict with ``global_result``, ``csv_path``, ``summary_path``,
            ``importance_path``.
        """
        logger.info("Running global SHAP analysis …")
        global_result = self.explainer.explain_global(X_background)

        shap_values = global_result["shap_values"]
        mean_abs = global_result["mean_abs_shap"]
        feature_names = global_result["feature_names"]

        summary_path = plot_shap_summary(
            shap_values, X_background, feature_names,
            output_dir=self.eval_dir,
        )
        importance_path = plot_shap_feature_importance(
            mean_abs, feature_names,
            output_dir=self.eval_dir,
        )
        csv_path = save_global_feature_importance_csv(
            mean_abs, feature_names,
            output_dir=self.eval_dir,
        )

        logger.info("Global SHAP analysis complete.")
        return {
            "global_result": global_result,
            "summary_path": summary_path,
            "importance_path": importance_path,
            "csv_path": csv_path,
        }

    # ── Local analysis ────────────────────────────────────────────────────────

    def run_local(
        self,
        x_record: np.ndarray,
        customer_id: str = "sample",
    ) -> Dict[str, Any]:
        """
        Run local SHAP analysis for a single customer record.

        Args:
            x_record:    1-D or (1, n_features) feature array.
            customer_id: Identifier used in filenames.

        Returns:
            Dict with ``local_result``, ``waterfall_path``, ``force_path``,
            ``contributions_path``.
        """
        logger.info(f"Running local SHAP analysis for customer_id={customer_id!r}")

        if x_record.ndim == 2:
            x_1d = x_record[0]
        else:
            x_1d = x_record

        local_result = self.explainer.explain_local(x_1d)

        shap_exp = self.explainer.get_shap_explainer()

        waterfall_path = plot_waterfall(
            shap_explainer=shap_exp,
            shap_values=local_result["shap_values"],
            feature_names=local_result["feature_names"],
            feature_values=local_result["feature_values"],
            expected_value=local_result["expected_value"],
            customer_id=customer_id,
            output_dir=self.local_dir,
        )

        force_path = plot_force(
            shap_explainer=shap_exp,
            shap_values=local_result["shap_values"],
            feature_names=local_result["feature_names"],
            feature_values=local_result["feature_values"],
            expected_value=local_result["expected_value"],
            customer_id=customer_id,
            output_dir=self.local_dir,
        )

        contributions_path = save_feature_contributions_csv(
            shap_values=local_result["shap_values"],
            feature_names=local_result["feature_names"],
            feature_values=local_result["feature_values"],
            output_dir=self.local_dir,
            filename=f"feature_contributions_{customer_id}.csv",
        )

        logger.info("Local SHAP analysis complete.")
        return {
            "local_result": local_result,
            "waterfall_path": waterfall_path,
            "force_path": force_path,
            "contributions_path": contributions_path,
        }

    # ── Report generation ─────────────────────────────────────────────────────

    def generate_report(
        self,
        global_result: Dict[str, Any],
        local_result: Dict[str, Any],
        sample_narrative: str = "",
        artifact_paths: Optional[Dict[str, Path]] = None,
    ) -> Path:
        """
        Write the Markdown explainability report.

        Args:
            global_result:    Output from run_global()["global_result"].
            local_result:     Output from run_local()["local_result"].
            sample_narrative: Business narrative for the sample customer.
            artifact_paths:   Dict of named path objects to include in the index.

        Returns:
            Path to the generated report file.
        """
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        top_features = global_result.get("top_features", [])
        expected_value = global_result.get("expected_value", 0.0)
        pos_drivers = local_result.get("positive_drivers", [])
        neg_drivers = local_result.get("negative_drivers", [])
        sample_prob = local_result.get("predicted_probability", 0.0)

        lines = [
            f"# Explainability Report — {settings.app_name}",
            f"\n**Generated:** {ts}  ",
            f"**Platform Version:** {settings.app_version}  ",
            f"**Method:** SHAP TreeExplainer (LightGBM)",
            "\n---\n",

            "## 1. Executive Summary",
            "",
            f"| Metric | Value |",
            "| --- | --- |",
            f"| Base Rate (Expected Value) | {expected_value:.4f} |",
            f"| Top Global Risk Driver | "
            f"`{top_features[0]['feature'] if top_features else 'N/A'}` |",
            f"| Features Explained | {len(global_result.get('feature_names', []))} |",
            f"| Sample Customer Predicted Prob | {sample_prob:.4f} |",
            "",
            "\n---\n",

            "## 2. Global Feature Importance (Top 20 by Mean |SHAP|)",
            "",
            "| Rank | Feature | Mean |SHAP| |",
            "| --- | --- | --- |",
        ]

        for i, feat in enumerate(top_features[:20], 1):
            lines.append(
                f"| {i} | `{feat['feature']}` | {feat['importance']:.5f} |"
            )

        lines += [
            "",
            "\n---\n",

            "## 3. Top Risk Drivers",
            "",
            "### 3.1 Risk-Increasing Features (Positive SHAP — push toward default)",
            "",
            "| Feature | SHAP Value | Feature Value |",
            "| --- | --- | --- |",
        ]

        for d in pos_drivers[:10]:
            lines.append(
                f"| `{d['feature']}` | +{d['shap_value']:.5f} | {d['value']:.4f} |"
            )

        lines += [
            "",
            "### 3.2 Risk-Reducing Features (Negative SHAP — push away from default)",
            "",
            "| Feature | SHAP Value | Feature Value |",
            "| --- | --- | --- |",
        ]

        for d in neg_drivers[:10]:
            lines.append(
                f"| `{d['feature']}` | {d['shap_value']:.5f} | {d['value']:.4f} |"
            )

        lines += [
            "",
            "\n---\n",

            "## 4. Sample Customer Explanation",
            "",
            f"**Predicted Default Probability:** {sample_prob:.4f}  ",
            f"**Base Rate:** {expected_value:.4f}",
            "",
            "### Business Narrative",
            "",
            f"> {sample_narrative}" if sample_narrative else "> _No narrative generated._",
            "",
            "\n---\n",

            "## 5. Interpretation Guidance",
            "",
            "### What are SHAP values?",
            "SHAP (SHapley Additive exPlanations) values measure the contribution "
            "of each feature to the model's prediction for a specific applicant. "
            "A positive SHAP value increases the predicted default probability; "
            "a negative value decreases it.",
            "",
            "### Risk Band Thresholds",
            "",
            "| Band | Probability Range | Risk Score |",
            "| --- | --- | --- |",
            "| 🟢 Low Risk | 0.00 – 0.20 | 0 – 200 |",
            "| 🟡 Medium Risk | 0.20 – 0.50 | 200 – 500 |",
            "| 🔴 High Risk | 0.50 – 1.00 | 500 – 1000 |",
            "",
            "### How to read the waterfall plot",
            "The waterfall chart starts at the model's base rate (expected value) "
            "and shows how each feature moves the prediction upward (red, risk-increasing) "
            "or downward (blue, risk-reducing) to reach the final predicted probability.",
            "",
            "### How to read the force plot",
            "The force plot compresses the same information into a single horizontal bar, "
            "showing all features simultaneously. Red features push the prediction right "
            "(toward default); blue features push it left.",
            "",
            "### Audit Note",
            "> [!NOTE]",
            "> SHAP explanations reflect the model's learned patterns from historical data. "
            "They should be used as decision-support tools and reviewed alongside manual "
            "underwriting judgement, policy rules, and regulatory requirements.",
            "",
            "\n---\n",

            "## 6. Artifact Index",
            "",
            "| Artifact | Path |",
            "| --- | --- |",
            "| Global Feature Importance CSV | `documents/explainability/global_feature_importance.csv` |",
            "| SHAP Summary Plot | `documents/explainability/shap_summary.png` |",
            "| SHAP Feature Importance Plot | `documents/explainability/shap_feature_importance.png` |",
            "| Sample Waterfall Plot | `documents/explainability/local/waterfall_sample.png` |",
            "| Sample Force Plot | `documents/explainability/local/force_sample.png` |",
            "| Sample Feature Contributions | `documents/explainability/local/feature_contributions_sample.csv` |",
            "",
            "\n---\n",
            f"*Report generated by {settings.app_name} v{settings.app_version} on {ts}*",
        ]

        report_path = self.output_dir / REPORT_FILENAME
        with open(report_path, "w", encoding="utf-8") as fh:
            fh.write("\n".join(lines))

        logger.info(f"Explainability report saved → {report_path}")
        return report_path

    # ── Full orchestration ────────────────────────────────────────────────────

    def run(
        self,
        X_background: np.ndarray,
        sample_record: np.ndarray,
        business_narrative: str = "",
        customer_id: str = "sample",
    ) -> Dict[str, Path]:
        """
        Full explainability pipeline: global → local → report.

        Args:
            X_background:     Background dataset (2-D array).
            sample_record:    Single customer record (1-D or 2-D array).
            business_narrative: Plain-English narrative for the sample.
            customer_id:      Identifier for local artifact filenames.

        Returns:
            Dict of named output paths.
        """
        global_artifacts = self.run_global(X_background)
        local_artifacts = self.run_local(sample_record, customer_id=customer_id)

        report_path = self.generate_report(
            global_result=global_artifacts["global_result"],
            local_result=local_artifacts["local_result"],
            sample_narrative=business_narrative,
        )

        return {
            "report": report_path,
            "global_csv": global_artifacts["csv_path"],
            "shap_summary": global_artifacts["summary_path"],
            "shap_importance": global_artifacts["importance_path"],
            "waterfall": local_artifacts["waterfall_path"],
            "force": local_artifacts["force_path"],
            "contributions_csv": local_artifacts["contributions_path"],
        }
