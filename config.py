"""
config.py — Central application configuration.

All settings are loaded from environment variables or .env file via pydantic-settings.
No values are hardcoded; all tuneable parameters live here.

Usage:
    from config import settings
    print(settings.data_raw_dir)
"""

from pathlib import Path
from typing import List, Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Resolve the project root directory (directory containing this file)
BASE_DIR: Path = Path(__file__).resolve().parent


class Settings(BaseSettings):
    """Application-wide configuration with environment variable support."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ── Application ──────────────────────────────────────────────────────────
    app_name: str = Field(default="Credit Risk Intelligence Platform")
    app_version: str = Field(default="1.0.0")

    # ── Directory Paths ───────────────────────────────────────────────────────
    data_raw_dir: Path = Field(default=BASE_DIR / "data" / "raw")
    data_processed_dir: Path = Field(default=BASE_DIR / "data" / "processed")
    eda_output_dir: Path = Field(default=BASE_DIR / "documents" / "eda")
    models_dir: Path = Field(default=BASE_DIR / "models")
    notebooks_dir: Path = Field(default=BASE_DIR / "notebooks")
    documents_dir: Path = Field(default=BASE_DIR / "documents")

    # ── Dataset Filenames ─────────────────────────────────────────────────────
    app_train_filename: str = Field(default="application_train.csv")
    bureau_filename: str = Field(default="bureau.csv")
    previous_app_filename: str = Field(default="previous_application.csv")
    bureau_balance_filename: str = Field(default="bureau_balance.csv")
    pos_cash_filename: str = Field(default="POS_CASH_balance.csv")
    installments_filename: str = Field(default="installments_payments.csv")
    credit_card_filename: str = Field(default="credit_card_balance.csv")

    # ── EDA Settings ──────────────────────────────────────────────────────────
    eda_sample_size: Optional[int] = Field(
        default=None,
        description="Row limit for EDA sampling. None = use full dataset.",
    )
    eda_correlation_top_n: int = Field(
        default=30,
        description="Number of top features to include in correlation heatmap.",
    )
    eda_figure_dpi: int = Field(default=150, description="DPI for saved plot images.")
    eda_figure_style: str = Field(
        default="darkgrid",
        description="Seaborn figure style (darkgrid | whitegrid | dark | white | ticks).",
    )
    eda_outlier_cap_percentile: float = Field(
        default=0.99,
        description="Percentile used to cap outliers in income/credit plots.",
    )
    eda_min_correlation_threshold: float = Field(
        default=0.01,
        description="Minimum absolute correlation to include a feature in reports.",
    )
    eda_report_filename: str = Field(default="eda_report.md")
    eda_profile_json_filename: str = Field(default="eda_profile.json")

    # ── ML Settings (reserved for future phases) ──────────────────────────────
    random_seed: int = Field(default=42)
    test_size: float = Field(default=0.2)
    cv_folds: int = Field(default=5)
    model_filename: str = Field(default="lgbm_credit_risk.pkl")
    preprocessor_filename: str = Field(default="preprocessor.pkl")

    # ── LLM / Chatbot Settings (reserved for future phases) ───────────────────
    openai_api_key: Optional[str] = Field(default=None, alias="OPENAI_API_KEY")
    openai_base_url: Optional[str] = Field(
        default="https://api.openai.com/v1", alias="OPENAI_BASE_URL"
    )
    openai_model: str = Field(default="gpt-4o-mini", alias="OPENAI_MODEL")
    openai_temperature: float = Field(default=0.0, alias="OPENAI_TEMPERATURE")
    openai_max_tokens: int = Field(default=1024, alias="OPENAI_MAX_TOKENS")
    chatbot_max_history_turns: int = Field(default=10)
    sql_max_result_rows: int = Field(default=500)

    # ── Database Settings (reserved for future phases) ────────────────────────
    db_path: Path = Field(default=BASE_DIR / "data" / "credit_risk.db")
    db_echo_sql: bool = Field(default=False)

    # ── Logging ───────────────────────────────────────────────────────────────
    log_level: str = Field(default="INFO")
    log_format: str = Field(
        default="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        description="Python logging format string.",
    )
    log_date_format: str = Field(default="%Y-%m-%d %H:%M:%S")

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        upper = v.upper()
        if upper not in allowed:
            raise ValueError(f"log_level must be one of {allowed}, got '{v}'")
        return upper

    @field_validator("test_size")
    @classmethod
    def validate_test_size(cls, v: float) -> float:
        if not 0.05 <= v <= 0.5:
            raise ValueError(f"test_size must be between 0.05 and 0.5, got {v}")
        return v

    @field_validator("eda_outlier_cap_percentile")
    @classmethod
    def validate_percentile(cls, v: float) -> float:
        if not 0.5 <= v <= 1.0:
            raise ValueError(f"eda_outlier_cap_percentile must be between 0.5 and 1.0")
        return v

    def ensure_directories(self) -> None:
        """Create all required output directories if they do not exist."""
        dirs = [
            self.data_raw_dir,
            self.data_processed_dir,
            self.eda_output_dir,
            self.models_dir,
            self.notebooks_dir,
            self.documents_dir,
            self.db_path.parent,
        ]
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)


# Singleton instance — import this throughout the project
settings = Settings()
