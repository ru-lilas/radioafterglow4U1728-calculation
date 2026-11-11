import math
import numpy as np
import pandas as pd

def real_scalar(value: object) -> float:
    """Validate a finite real scalar before conversion"""
    if isinstance(value, np.generic):
        native: object = value.item()
        if isinstance(native, np.generic):
            raise TypeError("Unsupported NumPy scalar precision")
        return real_scalar(native)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"Expected a real scalar, got {type(value).__name__}")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("Expected a finite number")
    return result

def require_series(df: pd.DataFrame, name: str) -> pd.Series:
    """Select exactly one named column"""
    column: object = df[name]
    if not isinstance(column, pd.Series):
        raise TypeError(f"Expected one Series for {name}")
    return column
