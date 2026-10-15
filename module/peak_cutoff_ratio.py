"""Shared model-ID matching and numerical peak-to-cutoff ratio calculation"""
from __future__ import annotations

import math
import numpy as np
import pandas as pd
from numpy.typing import NDArray

from module.models import ColumnNames, ModelParameters
from module.mydataclasses import QuantityGrid

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
RATIO_COLUMN = "peak_cutoff_ratio"


def require_series(df: pd.DataFrame, name: str) -> pd.Series:
    """Require a single unambiguous column before inspecting external scalars"""
    if not df.columns.is_unique:
        raise ValueError("Duplicate column labels in ratio table")
    if name not in df:
        raise ValueError(f"Missing ratio input column: {name}")
    column: object = df[name]
    if not isinstance(column, pd.Series):
        raise TypeError(f"Expected one Series for {name}")
    return column


def real_scalar(value: object, label: str) -> float:
    """Narrow a finite real scalar before converting it to Python float"""
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{label} must be a real scalar")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def model_id(value: object) -> int:
    """Validate a nonnegative integer identifier without truncation"""
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, bool):
        raise TypeError("Boolean model indices are invalid")
    if isinstance(value, int):
        result = value
    elif isinstance(value, float) and math.isfinite(value) and value.is_integer():
        result = int(value)
    else:
        raise TypeError("model_index must contain integer identifiers")
    if not 0 <= result < 2**63:
        raise ValueError("model_index is outside the supported int64 range")
    return result


def column_ids(df: pd.DataFrame) -> IntArray:
    """Read unique model identifiers from an external peak table"""
    values = np.asarray([model_id(value) for value in require_series(df, ColumnNames.MODEL_INDEX).tolist()], dtype=np.int64)
    if values.size == 0 or len(set(values.tolist())) != values.size:
        raise ValueError("Model indices must be nonempty and unique")
    return values


def model_ids(value: object, label: str) -> IntArray:
    """Validate an integer grid while preserving its shape and order"""
    if not isinstance(value, np.ndarray) or value.dtype.kind not in "iu" or value.size == 0:
        raise TypeError(f"{label} must be a nonempty integer array")
    values = np.asarray([model_id(item) for item in value.ravel()], dtype=np.int64).reshape(value.shape)
    if len(set(values.ravel().tolist())) != values.size:
        raise ValueError(f"Duplicate identifiers in {label}")
    return values


def numeric_grid(value: object, shape: tuple[int, ...], label: str) -> FloatArray:
    """Validate finite real model values against the identifier-grid shape"""
    if not isinstance(value, np.ndarray) or value.dtype.kind not in "fiu":
        raise TypeError(f"{label} must be a real numeric array")
    if value.shape != shape:
        raise ValueError(f"{label} shape differs from model identifiers")
    result = np.asarray(value, dtype=np.float64)
    if np.any(~np.isfinite(result)) or np.any(result <= 0):
        raise ValueError(f"{label} must be positive and finite")
    return result


def build_ratio_table(df_peak: pd.DataFrame, model_params: ModelParameters) -> pd.DataFrame:
    """Match every model by ID and retain the legacy peak_cutoff_ratio column"""
    peak_ids = column_ids(df_peak)
    peak = np.asarray([real_scalar(value, "xm_peak") for value in require_series(df_peak, ColumnNames.XM).tolist()], dtype=np.float64)
    if np.any(peak <= 0):
        raise ValueError("xm_peak must be positive")
    ids = model_ids(model_params.model_indices, "model indices")
    if set(peak_ids.tolist()) != set(ids.ravel().tolist()):
        raise ValueError("Peak and parameter tables must contain exactly the same model indices")
    cutoff = numeric_grid(model_params.xm_cutoff, ids.shape, "xm_cutoff").ravel()
    phi = numeric_grid(model_params.phi_th, ids.shape, "phi_th").ravel()
    theta0 = numeric_grid(model_params.theta0, ids.shape, "theta0").ravel()
    peak_by_id = {int(index): real_scalar(value, "xm_peak") for index, value in zip(peak_ids, peak)}
    aligned_peak = np.asarray([peak_by_id[int(index)] for index in ids.ravel()], dtype=np.float64)
    ratio = aligned_peak / cutoff
    if np.any(~np.isfinite(ratio)) or np.any(ratio <= 0):
        raise ValueError("xm_peak/xm_cutoff must be positive and finite")
    table = pd.DataFrame({
        ColumnNames.MODEL_INDEX: ids.ravel(), "xm_peak": aligned_peak, "xm_cutoff": cutoff,
        ColumnNames.PHI_TH: phi, ColumnNames.THETA0: theta0, RATIO_COLUMN: ratio,
    })
    return table.sort_values(ColumnNames.MODEL_INDEX).reset_index(drop=True)


def build_ratio_quantity_grid(df_peak: pd.DataFrame, model_params: ModelParameters,
                              target_model_indices: IntArray) -> QuantityGrid:
    """Reorder ratios to an explicit Chevalier grid without free-parameter pivoting"""
    table = build_ratio_table(df_peak, model_params)
    target = model_ids(target_model_indices, "target peak-grid indices")
    if target.ndim != 2:
        raise ValueError("Target Chevalier model indices must form a two-dimensional grid")
    indices = column_ids(table)
    if set(target.ravel().tolist()) != set(indices.tolist()):
        raise ValueError("Target Chevalier grid must contain the same model indices")
    ratios = np.asarray([real_scalar(value, RATIO_COLUMN) for value in require_series(table, RATIO_COLUMN).tolist()], dtype=np.float64)
    by_id = {int(index): real_scalar(value, RATIO_COLUMN) for index, value in zip(indices, ratios)}
    values = np.asarray([by_id[int(index)] for index in target.ravel()], dtype=np.float64).reshape(target.shape)
    return QuantityGrid(values=values, unit="")
