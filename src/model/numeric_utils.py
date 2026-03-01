"""Ensure DataFrame is numeric for LightGBM."""
import numpy as np
import pandas as pd


def ensure_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Convert object/string/datetime columns to numeric. In-place style; returns df."""
    out = df.copy()
    for c in list(out.columns):
        col = out[c]
        if col.dtype == object or (getattr(col.dtype, "name", None) in ("object", "string")):
            out[c] = pd.Categorical(col.fillna("__NA__")).codes.astype(np.float64)
        elif pd.api.types.is_datetime64_any_dtype(col):
            out[c] = (col - pd.Timestamp("1970-01-01")).dt.total_seconds().astype(np.float64)
    for c in list(out.columns):
        if out[c].dtype not in (np.float32, np.float64, np.int8, np.int16, np.int32, np.int64, np.bool_):
            try:
                out[c] = out[c].astype(np.float64)
            except (ValueError, TypeError):
                out[c] = pd.Categorical(out[c]).codes.astype(np.float64)
    return out
