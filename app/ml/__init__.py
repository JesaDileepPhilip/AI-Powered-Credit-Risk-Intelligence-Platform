"""
app/ml — Feature engineering, preprocessing, and ML training pipeline.

Phase 2 Modules:
  - data_validator:     Schema, dtype, and target validation
  - feature_engineering: Derived ratio features
  - feature_selector:   High-missing and zero-variance removal
  - preprocessing:      Imputation, encoding, sklearn Pipeline orchestration
  - data_splitter:      Stratified train/test split
  - artifact_manager:   Persistence and report generation

Phase 3 Modules:
  - train:              Full training orchestrator (LR baseline + LightGBM)
  - evaluate:           Metrics computation and evaluation plot generators
  - predict:            Batch and single-record prediction utilities
  - risk_scoring:       Default probability → risk band + risk score mapping
  - model_registry:     Model and metadata persistence layer
  - inference_service:  Reusable, schema-validating inference service
"""

__all__ = [
    "PreprocessingPipeline",
    "CreditRiskTrainer",
    "InferenceService",
    "ModelRegistry",
    "score_single",
    "score_batch",
]


def __getattr__(name: str):
    if name == "PreprocessingPipeline":
        from app.ml.preprocessing import PreprocessingPipeline
        return PreprocessingPipeline
    if name == "CreditRiskTrainer":
        from app.ml.train import CreditRiskTrainer
        return CreditRiskTrainer
    if name == "InferenceService":
        from app.ml.inference_service import InferenceService
        return InferenceService
    if name == "ModelRegistry":
        from app.ml.model_registry import ModelRegistry
        return ModelRegistry
    if name == "score_single":
        from app.ml.risk_scoring import score_single
        return score_single
    if name == "score_batch":
        from app.ml.risk_scoring import score_batch
        return score_batch
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

