"""
app/eda/__init__.py — EDA package public API.

Exports the five core EDA classes and a convenience ``run_eda_pipeline()``
function so callers only need a single import.

Example::

    from app.eda import run_eda_pipeline
    report_path, json_path = run_eda_pipeline()

    # — or use individual classes —
    from app.eda import DataLoader, DataProfiler, EDAVisualizer
"""

from app.eda.data_loader       import DataLoader
from app.eda.insights          import BusinessInsightGenerator
from app.eda.profiler          import DataProfiler
from app.eda.report_generator  import EDAReportGenerator
from app.eda.visualizations    import EDAVisualizer

__all__ = [
    "DataLoader",
    "DataProfiler",
    "EDAVisualizer",
    "BusinessInsightGenerator",
    "EDAReportGenerator",
    "run_eda_pipeline",
]


def run_eda_pipeline() -> tuple:
    """
    Convenience wrapper — runs the full EDA pipeline with default settings.

    Returns:
        ``(report_md_path, profile_json_path)`` as returned by
        :meth:`EDAReportGenerator.run`.
    """
    generator = EDAReportGenerator()
    return generator.run()
