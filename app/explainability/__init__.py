"""
app/explainability/__init__.py — Phase 4 Explainable AI package.

Modules:
  - shap_explainer:       TreeExplainer wrapper for global and local SHAP values
  - visualizations:       SHAP plot generators (summary, waterfall, force, importance)
  - explanation_service:  Reusable end-to-end explain(record) service
  - business_explainer:   Human-readable, non-technical narrative generator
  - report_generator:     Markdown explainability report orchestrator
"""

__all__ = [
    "ExplanationService",
    "SHAPExplainer",
    "BusinessExplainer",
]


def __getattr__(name: str):
    if name == "ExplanationService":
        from app.explainability.explanation_service import ExplanationService
        return ExplanationService
    if name == "SHAPExplainer":
        from app.explainability.shap_explainer import SHAPExplainer
        return SHAPExplainer
    if name == "BusinessExplainer":
        from app.explainability.business_explainer import BusinessExplainer
        return BusinessExplainer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
