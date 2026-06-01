"""
tests/test_data_loader.py — Unit tests for app.eda.data_loader.DataLoader.

Tests cover:
  - Successful CSV loading with schema validation
  - Memory reduction behaviour
  - Graceful handling of optional missing files
  - Schema validation errors
  - TARGET integrity checks
"""

from __future__ import annotations

import io
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def valid_app_train_df() -> pd.DataFrame:
    """Minimal valid application_train DataFrame."""
    return pd.DataFrame(
        {
            "SK_ID_CURR":      [100001, 100002, 100003, 100004, 100005],
            "TARGET":          [0, 1, 0, 0, 1],
            "AMT_INCOME_TOTAL":[135000.0, 99000.0, 202500.0, 450000.0, 135000.0],
            "AMT_CREDIT":      [406597.5, 1293502.5, 135000.0, 312682.5, 513000.0],
            "AMT_ANNUITY":     [24700.5, 35698.5, 6750.0, 15625.5, 25396.0],
            "DAYS_BIRTH":      [-9461, -16765, -19046, -19005, -11111],
            "DAYS_EMPLOYED":   [-637, -1188, 365243, -3227, -508],
            "CODE_GENDER":     ["M", "F", "M", "F", "M"],
            "EXT_SOURCE_1":    [0.083, 0.311, np.nan, 0.502, 0.721],
            "EXT_SOURCE_2":    [0.263, 0.622, 0.555, 0.729, 0.339],
            "EXT_SOURCE_3":    [0.139, np.nan, 0.729, 0.561, 0.802],
        }
    )


@pytest.fixture()
def data_loader_with_mock(valid_app_train_df, tmp_path):
    """
    DataLoader configured to read from a temp directory,
    with application_train.csv pre-written.
    """
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    valid_app_train_df.to_csv(raw_dir / "application_train.csv", index=False)

    # Patch settings to use tmp_path
    with patch("app.eda.data_loader.settings") as mock_settings:
        mock_settings.data_raw_dir = raw_dir
        mock_settings.app_train_filename = "application_train.csv"
        mock_settings.bureau_filename = "bureau.csv"
        mock_settings.previous_app_filename = "previous_application.csv"
        mock_settings.random_seed = 42
        mock_settings.log_level = "INFO"
        mock_settings.log_format = "%(message)s"
        mock_settings.log_date_format = "%H:%M:%S"

        from app.eda.data_loader import DataLoader
        yield DataLoader(), raw_dir


# ---------------------------------------------------------------------------
# DataLoader tests
# ---------------------------------------------------------------------------


class TestDataLoaderLoad:
    """Tests for successful file loading."""

    def test_load_application_train_returns_dataframe(self, data_loader_with_mock):
        loader, _ = data_loader_with_mock
        df = loader.load_application_train()
        assert isinstance(df, pd.DataFrame)

    def test_load_application_train_correct_shape(self, data_loader_with_mock):
        loader, _ = data_loader_with_mock
        df = loader.load_application_train()
        assert df.shape == (5, 11)

    def test_target_column_values_are_binary(self, data_loader_with_mock):
        loader, _ = data_loader_with_mock
        df = loader.load_application_train()
        assert set(df["TARGET"].dropna().unique()).issubset({0, 1})

    def test_sk_id_curr_is_present(self, data_loader_with_mock):
        loader, _ = data_loader_with_mock
        df = loader.load_application_train()
        assert "SK_ID_CURR" in df.columns

    def test_load_all_returns_dict_with_app_train(self, data_loader_with_mock):
        loader, _ = data_loader_with_mock
        datasets = loader.load_all()
        assert "application_train" in datasets
        assert isinstance(datasets["application_train"], pd.DataFrame)

    def test_optional_bureau_absent_returns_none(self, data_loader_with_mock):
        loader, _ = data_loader_with_mock
        result = loader.load_bureau()
        assert result is None

    def test_optional_previous_app_absent_returns_none(self, data_loader_with_mock):
        loader, _ = data_loader_with_mock
        result = loader.load_previous_application()
        assert result is None

    def test_get_loaded_names_after_load_all(self, data_loader_with_mock):
        loader, _ = data_loader_with_mock
        loader.load_all()
        names = loader.get_loaded_names()
        assert "application_train" in names

    def test_get_dataset_returns_frame(self, data_loader_with_mock):
        loader, _ = data_loader_with_mock
        loader.load_application_train()
        df = loader.get_dataset("application_train")
        assert df is not None
        assert isinstance(df, pd.DataFrame)

    def test_get_dataset_unknown_name_returns_none(self, data_loader_with_mock):
        loader, _ = data_loader_with_mock
        assert loader.get_dataset("nonexistent") is None


class TestDataLoaderOptionalFiles:
    """Tests for optional file loading when files exist."""

    def test_load_bureau_when_present(self, valid_app_train_df, tmp_path):
        """Bureau loading works when the file exists in raw_dir."""
        raw_dir = tmp_path / "raw2"
        raw_dir.mkdir()
        valid_app_train_df.to_csv(raw_dir / "application_train.csv", index=False)

        bureau_df = pd.DataFrame(
            {
                "SK_ID_CURR":          [100001, 100001, 100002],
                "SK_ID_BUREAU":        [5000001, 5000002, 5000003],
                "CREDIT_ACTIVE":       ["Active", "Closed", "Active"],
                "CREDIT_DAY_OVERDUE":  [0, 0, 5],
                "AMT_CREDIT_SUM":      [135000.0, 45000.0, 270000.0],
                "CREDIT_TYPE":         ["Consumer credit", "Car loan", "Consumer credit"],
            }
        )
        bureau_df.to_csv(raw_dir / "bureau.csv", index=False)

        from app.eda.data_loader import DataLoader

        with patch("app.eda.data_loader.settings") as mock_settings:
            mock_settings.data_raw_dir = raw_dir
            mock_settings.app_train_filename = "application_train.csv"
            mock_settings.bureau_filename = "bureau.csv"
            mock_settings.previous_app_filename = "previous_application.csv"
            mock_settings.random_seed = 42
            mock_settings.log_level = "INFO"
            mock_settings.log_format = "%(message)s"
            mock_settings.log_date_format = "%H:%M:%S"

            loader = DataLoader()
            result = loader.load_bureau()

        assert result is not None
        assert "SK_ID_BUREAU" in result.columns


class TestDataLoaderValidation:
    """Tests for schema validation on application_train."""

    def test_missing_required_column_raises_value_error(self, tmp_path):
        raw_dir = tmp_path / "raw3"
        raw_dir.mkdir()
        # Write CSV without TARGET
        bad_df = pd.DataFrame({"SK_ID_CURR": [1, 2], "AMT_CREDIT": [100.0, 200.0]})
        bad_df.to_csv(raw_dir / "application_train.csv", index=False)

        from app.eda.data_loader import DataLoader

        with patch("app.eda.data_loader.settings") as mock_settings:
            mock_settings.data_raw_dir = raw_dir
            mock_settings.app_train_filename = "application_train.csv"
            mock_settings.bureau_filename = "bureau.csv"
            mock_settings.previous_app_filename = "previous_application.csv"
            mock_settings.log_level = "INFO"
            mock_settings.log_format = "%(message)s"
            mock_settings.log_date_format = "%H:%M:%S"

            loader = DataLoader()
            with pytest.raises(ValueError, match="missing required column"):
                loader.load_application_train()

    def test_file_not_found_raises_error(self, tmp_path):
        raw_dir = tmp_path / "raw"
        raw_dir.mkdir()  # empty dir

        with patch("app.eda.data_loader.settings") as mock_settings:
            mock_settings.data_raw_dir = raw_dir
            mock_settings.app_train_filename = "application_train.csv"
            mock_settings.log_level = "INFO"
            mock_settings.log_format = "%(message)s"
            mock_settings.log_date_format = "%H:%M:%S"

            from importlib import reload
            import app.eda.data_loader as dl_module
            reload(dl_module)

            loader = dl_module.DataLoader()
            with pytest.raises(FileNotFoundError):
                loader.load_application_train()


class TestMemoryReduction:
    """Tests for the reduce_memory_usage utility function."""

    def test_memory_is_reduced(self):
        from app.utils.helpers import reduce_memory_usage

        df = pd.DataFrame(
            {
                "int_col":   pd.array([1, 2, 3, 4, 5], dtype=np.int64),
                "float_col": pd.array([1.1, 2.2, 3.3, 4.4, 5.5], dtype=np.float64),
            }
        )
        before = df.memory_usage(deep=True).sum()
        df_reduced = reduce_memory_usage(df.copy(), verbose=False)
        after = df_reduced.memory_usage(deep=True).sum()
        assert after <= before

    def test_output_shape_unchanged(self):
        from app.utils.helpers import reduce_memory_usage

        df = pd.DataFrame(np.random.randn(100, 10))
        result = reduce_memory_usage(df.copy(), verbose=False)
        assert result.shape == df.shape

    def test_object_columns_unchanged(self):
        from app.utils.helpers import reduce_memory_usage

        df = pd.DataFrame({"text": ["a", "b", "c"], "num": [1.0, 2.0, 3.0]})
        result = reduce_memory_usage(df.copy(), verbose=False)
        # In pandas 3.0+, string columns have StringDtype; either way, not numeric
        assert not pd.api.types.is_numeric_dtype(result["text"].dtype)
