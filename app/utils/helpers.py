"""
app/utils/helpers.py — Shared utility functions used across all modules.

Contains:
  - DataFrame memory reduction
  - Execution timer decorator
  - Safe JSON serialisation helper
  - Feature name sanitiser (for LightGBM compatibility)
"""

import functools
import json
import re
import time
from datetime import datetime
from typing import Any, Callable, Dict, TypeVar

import numpy as np
import pandas as pd

from app.utils.logger import get_logger

logger = get_logger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


# ─── DataFrame Utilities ──────────────────────────────────────────────────────

def reduce_memory_usage(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Downcast numeric columns to the smallest appropriate dtype.

    Reduces DataFrame RAM footprint, typically by 50-70% on wide datasets
    such as application_train.csv.  Object columns are left unchanged.

    Args:
        df:      Input DataFrame.
        verbose: Log memory before/after when True.

    Returns:
        The same DataFrame with dtypes lowered in-place.
    """
    start_mem_mb = df.memory_usage(deep=True).sum() / 1024 ** 2

    for col in df.columns:
        col_type = df[col].dtype

        # Skip non-numeric columns (object, string, category, bool, etc.)
        if not pd.api.types.is_numeric_dtype(col_type):
            continue

        c_min = df[col].min()
        c_max = df[col].max()

        if pd.api.types.is_integer_dtype(col_type):
            for int_type in (np.int8, np.int16, np.int32, np.int64):
                info = np.iinfo(int_type)
                if c_min >= info.min and c_max <= info.max:
                    df[col] = df[col].astype(int_type)
                    break
        elif pd.api.types.is_float_dtype(col_type):
            # float16 has limited precision; use float32 as smallest safe type
            if c_min >= np.finfo(np.float32).min and c_max <= np.finfo(np.float32).max:
                df[col] = df[col].astype(np.float32)

    if verbose:
        end_mem_mb = df.memory_usage(deep=True).sum() / 1024 ** 2
        reduction_pct = 100 * (start_mem_mb - end_mem_mb) / max(start_mem_mb, 1e-9)
        logger.info(
            f"Memory reduced: {start_mem_mb:.2f} MB → {end_mem_mb:.2f} MB "
            f"({reduction_pct:.1f}% reduction)"
        )

    return df


# ─── Decorator ────────────────────────────────────────────────────────────────

def timer(func: F) -> F:
    """
    Decorator that logs the wall-clock execution time of a function.

    Example:
        @timer
        def heavy_computation():
            ...
    """

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        logger.info(f"{func.__qualname__} completed in {elapsed:.3f}s")
        return result

    return wrapper  # type: ignore[return-value]


# ─── Serialisation ────────────────────────────────────────────────────────────

def make_json_serialisable(obj: Any) -> Any:
    """
    Recursively convert an object to a JSON-serialisable form.

    Handles numpy scalars, numpy arrays, pandas Timestamps, datetimes,
    and nested dicts/lists.  Unknown types are cast to str.

    Args:
        obj: Any Python object.

    Returns:
        A JSON-serialisable equivalent.
    """
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (pd.Timestamp, datetime)):
        return obj.isoformat()
    if isinstance(obj, dict):
        return {str(k): make_json_serialisable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [make_json_serialisable(i) for i in obj]
    if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
        return None
    return obj


def safe_json_dumps(obj: Any, **kwargs: Any) -> str:
    """Serialise *obj* to a JSON string, handling numpy/pandas types."""
    return json.dumps(make_json_serialisable(obj), **kwargs)


# ─── Feature Name Utilities ───────────────────────────────────────────────────

def sanitise_feature_names(columns: list) -> list:
    """
    Replace special characters in column names with underscores.

    LightGBM rejects feature names containing characters like ``[``, ``]``,
    ``<``, ``>``, ``{``, ``}``.  This function produces safe names while
    preserving readability.

    Args:
        columns: List of column name strings.

    Returns:
        List of sanitised column name strings.
    """
    pattern = re.compile(r"[^A-Za-z0-9_]")
    seen: Dict[str, int] = {}
    result = []

    for col in columns:
        safe = pattern.sub("_", col).strip("_")
        safe = re.sub(r"_+", "_", safe)  # collapse consecutive underscores

        # Resolve duplicates introduced by sanitisation
        if safe in seen:
            seen[safe] += 1
            safe = f"{safe}_{seen[safe]}"
        else:
            seen[safe] = 0

        result.append(safe)

    return result
