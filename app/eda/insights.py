"""
app/eda/insights.py — Automated business insight generator.

Analyses the Home Credit Default Risk dataset and surfaces actionable
findings with severity levels (info | warning | critical) and concrete
recommendations for downstream modelling phases.

Each ``analyze_*`` method:
  - Is independently callable (no required call order).
  - Appends structured dicts to ``self._insights``.
  - Returns a raw metrics dict for callers who need numeric values.
  - Skips gracefully when required columns are absent.

Usage:
    from app.eda.insights import BusinessInsightGenerator

    gen      = BusinessInsightGenerator(app_df, bureau_df=bureau_df)
    insights = gen.run()          # list[dict]
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from app.utils.logger import get_logger

logger = get_logger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Severity thresholds (configurable via class-level constants)
# ─────────────────────────────────────────────────────────────────────────────

_IMBALANCE_WARNING_RATIO  = 5.0   # non-default : default ratio above this → warning
_IMBALANCE_CRITICAL_RATIO = 10.0  # above this  → critical
_HIGH_MISSING_THRESHOLD   = 50.0  # % missing above which a feature is "high-missing"
_MOD_MISSING_THRESHOLD    = 10.0  # % missing above which a feature is "moderate-missing"
_YOUNG_AGE_CUTOFF         = 30    # years
_SENIOR_AGE_CUTOFF        = 55    # years
_SHORT_TENURE_CUTOFF      = 1     # years

_DAYS_EMPLOYED_SENTINEL   = 365_243  # value encoding unemployed/retired


# ─────────────────────────────────────────────────────────────────────────────
# BusinessInsightGenerator
# ─────────────────────────────────────────────────────────────────────────────


class BusinessInsightGenerator:
    """
    Derives domain-specific insights from the Home Credit Default Risk dataset.

    Args:
        df:         Primary application_train DataFrame.
        bureau_df:  Optional bureau DataFrame for credit history insights.

    Example::

        gen      = BusinessInsightGenerator(df, bureau_df=bureau_df)
        insights = gen.run()
        for item in insights:
            print(item["severity"], item["title"])
    """

    def __init__(
        self,
        df: pd.DataFrame,
        bureau_df: Optional[pd.DataFrame] = None,
    ) -> None:
        self.df = df
        self.bureau_df = bureau_df
        self._insights: List[Dict[str, Any]] = []
        logger.info(
            f"BusinessInsightGenerator initialised. "
            f"Bureau data: {'present' if bureau_df is not None else 'absent'}."
        )

    # ── Private helpers ───────────────────────────────────────────────────────

    def _record(
        self,
        category: str,
        title: str,
        finding: str,
        recommendation: str,
        severity: str = "info",
    ) -> None:
        """Append a structured insight record to the internal list."""
        if severity not in {"info", "warning", "critical"}:
            raise ValueError(
                f"Invalid severity: '{severity}'. Must be one of 'info', 'warning', or 'critical'."
            )
        self._insights.append(
            {
                "category":       category,
                "title":          title,
                "finding":        finding,
                "recommendation": recommendation,
                "severity":       severity,
            }
        )

    def _has_columns(self, *cols: str) -> bool:
        """Return True if *all* given columns exist in ``self.df``."""
        return all(c in self.df.columns for c in cols)

    # ── Analysis methods ──────────────────────────────────────────────────────

    def analyze_class_imbalance(self) -> Dict[str, Any]:
        """
        Detect and quantify class imbalance in the TARGET variable.

        Returns:
            Metrics dict with ``default_rate`` and ``imbalance_ratio``.
        """
        if not self._has_columns("TARGET"):
            return {}

        target = self.df["TARGET"].dropna()
        n_default     = int((target == 1).sum())
        n_non_default = int((target == 0).sum())
        default_rate  = float(target.mean() * 100)
        ratio         = n_non_default / max(n_default, 1)

        severity = (
            "critical" if ratio > _IMBALANCE_CRITICAL_RATIO
            else "warning" if ratio > _IMBALANCE_WARNING_RATIO
            else "info"
        )

        self._record(
            category="Target Variable",
            title="Class Imbalance Detected",
            finding=(
                f"Default rate: {default_rate:.2f}%.  "
                f"Non-default: {n_non_default:,} | Default: {n_default:,}  "
                f"(imbalance ratio {ratio:.1f}:1)."
            ),
            recommendation=(
                "Use scale_pos_weight in LightGBM (set to imbalance_ratio) or apply SMOTE "
                "oversampling from imbalanced-learn.  Evaluate with AUC-ROC, PR-AUC, and "
                "KS statistic rather than raw accuracy."
            ),
            severity=severity,
        )
        logger.info(
            f"Class imbalance — default rate: {default_rate:.2f}%, ratio: {ratio:.1f}:1, severity: {severity}"
        )
        return {"default_rate": default_rate, "imbalance_ratio": ratio}

    def analyze_income_patterns(self) -> Dict[str, Any]:
        """
        Compare income distributions across default / non-default groups.

        Returns:
            Metrics dict with median income values and gap percentage.
        """
        if not self._has_columns("AMT_INCOME_TOTAL", "TARGET"):
            return {}

        median_default     = float(self.df.loc[self.df["TARGET"] == 1, "AMT_INCOME_TOTAL"].median())
        median_non_default = float(self.df.loc[self.df["TARGET"] == 0, "AMT_INCOME_TOTAL"].median())

        # Guard against all-null income slices (produces nan from .median())
        if np.isnan(median_default) or np.isnan(median_non_default):
            logger.warning("Income median is NaN for one or both TARGET classes — skipping income insight.")
            return {}

        gap_pct = (median_non_default - median_default) / max(median_non_default, 1) * 100

        p99 = float(self.df["AMT_INCOME_TOTAL"].quantile(0.99))
        outlier_count = int((self.df["AMT_INCOME_TOTAL"] > p99).sum())

        self._record(
            category="Financial",
            title="Income Disparity Between Default Groups",
            finding=(
                f"Median income — Non-Default: ${median_non_default:,.0f}, "
                f"Default: ${median_default:,.0f} "
                f"(non-defaulters earn {gap_pct:.1f}% more at the median).  "
                f"{outlier_count:,} applicants ({outlier_count / len(self.df) * 100:.1f}%) "
                f"have income above the 99th percentile (${p99:,.0f})."
            ),
            recommendation=(
                "Engineer CREDIT_TO_INCOME_RATIO and ANNUITY_TO_INCOME_RATIO as features.  "
                "Cap income at the 99th percentile during preprocessing to prevent distortion."
            ),
            severity="info",
        )
        return {
            "median_income_default":     median_default,
            "median_income_non_default": median_non_default,
            "income_gap_pct":            gap_pct,
            "outlier_count":             outlier_count,
        }

    def analyze_credit_patterns(self) -> Dict[str, Any]:
        """
        Analyse credit amount and debt-to-income patterns by default status.

        Returns:
            Metrics dict with mean credit amounts and high-ratio count.
        """
        if not self._has_columns("AMT_CREDIT", "TARGET"):
            return {}

        mean_default     = float(self.df.loc[self.df["TARGET"] == 1, "AMT_CREDIT"].mean())
        mean_non_default = float(self.df.loc[self.df["TARGET"] == 0, "AMT_CREDIT"].mean())
        direction        = "larger" if mean_default > mean_non_default else "smaller"

        metrics: Dict[str, Any] = {
            "mean_credit_default":     mean_default,
            "mean_credit_non_default": mean_non_default,
        }

        high_ratio_count = 0
        if self._has_columns("AMT_INCOME_TOTAL"):
            ratio = self.df["AMT_CREDIT"] / self.df["AMT_INCOME_TOTAL"].replace(0, np.nan)
            high_ratio_count = int((ratio > 3).sum())
            metrics["high_credit_income_ratio_count"] = high_ratio_count

        self._record(
            category="Financial",
            title="Credit Amount Risk Profile",
            finding=(
                f"Defaulters take {direction} loans on average "
                f"(Default avg: ${mean_default:,.0f} vs Non-Default: ${mean_non_default:,.0f}).  "
                f"{high_ratio_count:,} applicants have credit > 3× annual income."
            ),
            recommendation=(
                "Add CREDIT_TO_INCOME_RATIO as a feature.  "
                "Applicants with credit > 3× income should trigger a 'high debt burden' policy flag."
            ),
            severity="warning" if mean_default > mean_non_default else "info",
        )
        return metrics

    def analyze_age_patterns(self) -> Dict[str, Any]:
        """
        Identify age-based default risk patterns.

        Returns:
            Metrics dict with default rates by age cohort.
        """
        if not self._has_columns("DAYS_BIRTH", "TARGET"):
            return {}

        age = (self.df["DAYS_BIRTH"].abs() / 365.25)
        overall_rate = float(self.df["TARGET"].mean() * 100)

        young_mask        = age < _YOUNG_AGE_CUTOFF
        senior_mask       = age > _SENIOR_AGE_CUTOFF
        young_rate        = float(self.df.loc[young_mask,  "TARGET"].mean() * 100)
        senior_rate       = float(self.df.loc[senior_mask, "TARGET"].mean() * 100)

        # Find the 5-year age bin with the highest default rate
        age_bins = pd.cut(age, bins=list(range(20, 75, 5)), right=False)
        bin_rates = self.df.groupby(age_bins, observed=True)["TARGET"].mean() * 100
        valid_bins = bin_rates.dropna()
        peak_bin  = str(valid_bins.idxmax()) if not valid_bins.empty else "N/A"

        severity = "warning" if young_rate > overall_rate * 1.25 else "info"

        self._record(
            category="Demographic",
            title="Age-Based Default Risk",
            finding=(
                f"Applicants under {_YOUNG_AGE_CUTOFF} have a {young_rate:.1f}% default rate "
                f"vs {overall_rate:.1f}% overall.  "
                f"Applicants over {_SENIOR_AGE_CUTOFF} have a {senior_rate:.1f}% default rate.  "
                f"Highest-risk age group: {peak_bin}."
            ),
            recommendation=(
                "Derive AGE_YEARS = DAYS_BIRTH.abs() / 365.25 as a model feature.  "
                "Consider age-stratified risk tiers in the business rules engine."
            ),
            severity=severity,
        )
        return {
            "young_default_rate":   young_rate,
            "senior_default_rate":  senior_rate,
            "overall_default_rate": overall_rate,
            "peak_age_bin":         peak_bin,
        }

    def analyze_employment_patterns(self) -> Dict[str, Any]:
        """
        Detect the DAYS_EMPLOYED sentinel anomaly and assess tenure-based risk.

        Returns:
            Metrics dict with anomaly count and short-tenure default rate.
        """
        if not self._has_columns("DAYS_EMPLOYED", "TARGET"):
            return {}

        emp = self.df["DAYS_EMPLOYED"].copy()
        anomaly_count = int((emp == _DAYS_EMPLOYED_SENTINEL).sum())
        anomaly_pct   = anomaly_count / len(emp) * 100

        emp_clean = emp.replace(_DAYS_EMPLOYED_SENTINEL, np.nan)
        emp_years = emp_clean.abs() / 365.25

        short_tenure_mask = emp_years < _SHORT_TENURE_CUTOFF
        short_tenure_rate = float(self.df.loc[short_tenure_mask, "TARGET"].mean() * 100)
        overall_rate      = float(self.df["TARGET"].mean() * 100)

        self._record(
            category="Financial",
            title="Employment Duration — Anomaly & Risk Analysis",
            finding=(
                f"DAYS_EMPLOYED contains {anomaly_count:,} rows ({anomaly_pct:.1f}%) "
                f"with the sentinel value {_DAYS_EMPLOYED_SENTINEL} (unemployed/retired).  "
                f"Applicants with < {_SHORT_TENURE_CUTOFF} yr employment have a "
                f"{short_tenure_rate:.1f}% default rate vs {overall_rate:.1f}% overall."
            ),
            recommendation=(
                f"Replace DAYS_EMPLOYED == {_DAYS_EMPLOYED_SENTINEL} with NaN and create "
                "'IS_EMPLOYED_ANOMALY' binary flag.  "
                "Derive EMPLOYMENT_YEARS = DAYS_EMPLOYED.abs() / 365.25 after cleaning."
            ),
            severity="warning",
        )
        return {
            "anomaly_count":          anomaly_count,
            "anomaly_pct":            anomaly_pct,
            "short_tenure_def_rate":  short_tenure_rate,
            "overall_default_rate":   overall_rate,
        }

    def analyze_external_sources(self) -> Dict[str, Any]:
        """
        Assess predictive power of external credit score features (EXT_SOURCE_*).

        Computes Pearson correlation between each EXT_SOURCE column and TARGET.
        Note: TARGET is binary (0/1), making Pearson equivalent to the
        point-biserial correlation in this case.

        Returns:
            Dict mapping EXT_SOURCE column name → Pearson correlation with TARGET.
        """
        ext_cols = [
            c for c in ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]
            if c in self.df.columns and "TARGET" in self.df.columns
        ]
        if not ext_cols:
            return {}

        correlations: Dict[str, float] = {}
        null_pcts:    Dict[str, float] = {}

        for col in ext_cols:
            corr = float(self.df[col].corr(self.df["TARGET"]))
            correlations[col]  = round(corr, 4)
            null_pcts[col]     = round(self.df[col].isnull().mean() * 100, 2)

        best = min(correlations, key=lambda k: correlations[k])  # strongest negative

        finding_parts = ", ".join(
            f"{c}: {v:.4f} ({null_pcts[c]:.1f}% missing)"
            for c, v in correlations.items()
        )

        self._record(
            category="Credit History",
            title="External Credit Scores — Top Predictors",
            finding=(
                f"Pearson correlation with TARGET — {finding_parts}.  "
                f"{best} shows the strongest negative association "
                "(higher external score ↔ lower default risk)."
            ),
            recommendation=(
                "EXT_SOURCE_1/2/3 are typically the most predictive features.  "
                "Create EXT_SOURCE_MEAN = mean(EXT_SOURCE_1, 2, 3) to retain "
                "signal for applicants missing individual scores."
            ),
            severity="info",
        )
        return correlations

    def analyze_missing_values(self) -> Dict[str, Any]:
        """
        Classify features by missing value severity and surface the worst offenders.

        Returns:
            Metrics dict with counts of high-missing and moderate-missing features.
        """
        missing_pct = self.df.isnull().mean() * 100
        high_missing     = missing_pct[missing_pct > _HIGH_MISSING_THRESHOLD]
        moderate_missing = missing_pct[
            (missing_pct > _MOD_MISSING_THRESHOLD) & (missing_pct <= _HIGH_MISSING_THRESHOLD)
        ]

        top5 = high_missing.nlargest(5)
        top5_str = ", ".join(
            f"`{col}` ({pct:.1f}%)" for col, pct in top5.items()
        ) if len(top5) else "none"

        severity = "warning" if len(high_missing) > 0 else "info"

        self._record(
            category="Data Quality",
            title="Missing Value Severity Assessment",
            finding=(
                f"{len(high_missing)} features have >{_HIGH_MISSING_THRESHOLD:.0f}% missing values.  "
                f"{len(moderate_missing)} features have "
                f"{_MOD_MISSING_THRESHOLD:.0f}–{_HIGH_MISSING_THRESHOLD:.0f}% missing.  "
                f"Worst offenders: {top5_str}."
            ),
            recommendation=(
                "Drop features with >80% missing values.  "
                "Apply median imputation for numeric and mode imputation for categorical features.  "
                "Create binary IS_MISSING_<col> indicator flags for features "
                f"with >{_MOD_MISSING_THRESHOLD:.0f}% missing — the pattern of missingness is itself predictive."
            ),
            severity=severity,
        )
        return {
            "high_missing_count":     int(len(high_missing)),
            "moderate_missing_count": int(len(moderate_missing)),
        }

    def analyze_bureau_integration(self) -> Dict[str, Any]:
        """
        Summarise the bureau dataset and recommend aggregation strategy.

        Returns:
            Metrics dict with record counts and overdue statistics.
        """
        if self.bureau_df is None:
            logger.info("Bureau data absent — skipping bureau insight.")
            return {}

        n_records        = int(len(self.bureau_df))
        n_unique_clients = int(self.bureau_df["SK_ID_CURR"].nunique())
        avg_per_client   = round(n_records / max(n_unique_clients, 1), 1)

        overdue_count = 0
        if "CREDIT_DAY_OVERDUE" in self.bureau_df.columns:
            overdue_count = int((self.bureau_df["CREDIT_DAY_OVERDUE"] > 0).sum())

        credit_type_dist = (
            self.bureau_df["CREDIT_TYPE"].value_counts().head(5).to_dict()
            if "CREDIT_TYPE" in self.bureau_df.columns else {}
        )

        self._record(
            category="Credit History",
            title="Bureau Credit History — Overview",
            finding=(
                f"Bureau contains {n_records:,} records for {n_unique_clients:,} unique applicants "
                f"(avg {avg_per_client} bureau entries per applicant).  "
                f"{overdue_count:,} records show overdue credits.  "
                + (f"Top credit types: {credit_type_dist}." if credit_type_dist else "")
            ),
            recommendation=(
                "Aggregate bureau per SK_ID_CURR:  "
                "count active / closed loans, average & max CREDIT_DAY_OVERDUE, "
                "total credit sum and debt.  Merge with application_train before training."
            ),
            severity="info",
        )
        return {
            "total_bureau_records":    n_records,
            "unique_clients":          n_unique_clients,
            "avg_entries_per_client":  avg_per_client,
            "overdue_record_count":    overdue_count,
        }

    def analyze_document_flags(self) -> Dict[str, Any]:
        """
        Assess predictive signal in document submission flags (FLAG_DOCUMENT_*).

        Returns:
            Dict mapping flag column → default rate for flagged applicants.
        """
        if "TARGET" not in self.df.columns:
            return {}

        flag_cols = [c for c in self.df.columns if c.startswith("FLAG_DOCUMENT_")]
        if not flag_cols:
            return {}

        rates: Dict[str, float] = {}
        for col in flag_cols:
            flagged_mask = self.df[col] == 1
            if flagged_mask.sum() < 10:
                continue
            rate = float(self.df.loc[flagged_mask, "TARGET"].mean() * 100)
            rates[col] = round(rate, 2)

        overall_rate = float(self.df["TARGET"].mean() * 100)
        high_risk_docs = {c: r for c, r in rates.items() if r > overall_rate * 1.2}

        self._record(
            category="Behavioural",
            title="Document Submission Flags — Risk Signal",
            finding=(
                f"{len(flag_cols)} document flags found.  "
                f"Overall default rate: {overall_rate:.2f}%.  "
                + (
                    f"Elevated risk documents (>1.2× overall rate): "
                    + ", ".join(f"{c} ({r:.1f}%)" for c, r in list(high_risk_docs.items())[:5])
                    if high_risk_docs else "No individual flag shows elevated risk."
                )
            ),
            recommendation=(
                "Consider creating a DOC_SUBMISSION_COUNT = sum of all FLAG_DOCUMENT_* columns "
                "as a single aggregated feature capturing documentation completeness."
            ),
            severity="info",
        )
        return rates

    # ── Full pipeline ─────────────────────────────────────────────────────────

    def run(self) -> List[Dict[str, Any]]:
        """
        Execute all insight analysis methods in sequence.

        Individual failures are logged and do not abort the pipeline.

        Returns:
            List of insight dicts, each with keys:
            ``category``, ``title``, ``finding``, ``recommendation``, ``severity``.
        """
        logger.info("Generating business insights ...")

        analysis_methods = [
            self.analyze_class_imbalance,
            self.analyze_income_patterns,
            self.analyze_credit_patterns,
            self.analyze_age_patterns,
            self.analyze_employment_patterns,
            self.analyze_external_sources,
            self.analyze_missing_values,
            self.analyze_bureau_integration,
            self.analyze_document_flags,
        ]

        for method in analysis_methods:
            try:
                method()
            except Exception as exc:
                logger.error(
                    f"Insight generation failed for {method.__name__}: {exc}",
                    exc_info=True,
                )

        logger.info(f"Insight generation complete: {len(self._insights)} insights produced.")
        return self._insights

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def insights(self) -> List[Dict[str, Any]]:
        """The list of generated insights (populated after ``run()``)."""
        return list(self._insights)

    def to_dataframe(self) -> pd.DataFrame:
        """Return insights as a flat DataFrame for tabular rendering."""
        return pd.DataFrame(self._insights)

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"BusinessInsightGenerator("
            f"rows={len(self.df):,}, "
            f"bureau={'yes' if self.bureau_df is not None else 'no'}, "
            f"insights={len(self._insights)})"
        )
