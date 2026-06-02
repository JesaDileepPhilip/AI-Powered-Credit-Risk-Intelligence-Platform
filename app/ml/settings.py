"""
app/ml/settings.py — Phase 2 configuration (extends root config without modifying it).

All tuneable preprocessing parameters live here.  Shared paths and seeds
are read from ``config.settings``.
"""

from pathlib import Path

from config import settings

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_RAW_DIR: Path = settings.data_raw_dir
DATA_PROCESSED_DIR: Path = settings.data_processed_dir
MODELS_DIR: Path = settings.models_dir
DOCUMENTS_DIR: Path = settings.documents_dir

INPUT_FILENAME: str = settings.app_train_filename
TRAIN_FILENAME: str = "train.csv"
TEST_FILENAME: str = "test.csv"
PIPELINE_FILENAME: str = "preprocessing_pipeline.pkl"
FEATURE_METADATA_FILENAME: str = "feature_metadata.json"
FE_REPORT_FILENAME: str = "feature_engineering_report.md"

# ── Split ─────────────────────────────────────────────────────────────────────
RANDOM_SEED: int = settings.random_seed
TEST_SIZE: float = settings.test_size

# ── Validation ────────────────────────────────────────────────────────────────
REQUIRED_COLUMNS: frozenset[str] = frozenset({"SK_ID_CURR", "TARGET"})
VALID_TARGET_VALUES: frozenset[int] = frozenset({0, 1})
ID_COLUMNS: frozenset[str] = frozenset({"SK_ID_CURR"})
TARGET_COLUMN: str = "TARGET"

# ── Feature engineering ─────────────────────────────────────────────────────────
DAYS_EMPLOYED_SENTINEL: float = 365_243.0
ENGINEERED_FEATURES: tuple[str, ...] = (
    "debt_to_income_ratio",
    "credit_to_income_ratio",
    "annuity_to_income_ratio",
    "employment_age_ratio",
)

# ── Feature selection ───────────────────────────────────────────────────────────
HIGH_MISSING_THRESHOLD: float = 0.80
BINARY_CATEGORY_MAX_UNIQUE: int = 2

# ── Encoding ────────────────────────────────────────────────────────────────────
ENCODING_STRATEGY: dict[str, str] = {
    "binary": "label_encoding",
    "multi_category": "one_hot_encoding",
}

# ── Imputation ──────────────────────────────────────────────────────────────────
NUMERIC_IMPUTATION_STRATEGY: str = "median"
CATEGORICAL_IMPUTATION_STRATEGY: str = "most_frequent"

# ── Encoding (inference) ────────────────────────────────────────────────────────
LABEL_ENCODER_UNKNOWN_VALUE: int = -1
