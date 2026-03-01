"""
Inference: load model, align features to training columns, score candidates, return top-K.
Latency measured for < 300ms target.
"""
from pathlib import Path
from typing import Any

import pandas as pd
import numpy as np
import lightgbm as lgb
import yaml
import time

from src.features.config import load_feature_config
from src.features.pipeline import build_inference_matrix
from src.features.restaurant_features import build_restaurant_features, build_item_features
from src.features.user_features import build_user_features
from src.model.config import load_model_config
from src.model.numeric_utils import ensure_numeric


def load_model_and_metadata(save_dir: str | Path | None = None) -> tuple[lgb.Booster, list[str], dict]:
    """Load LightGBM booster and feature column list."""
    save_dir = Path(save_dir or Path(__file__).resolve().parent.parent.parent / "models")
    booster = lgb.Booster(model_file=str(save_dir / "csao_lgb.txt"))
    with open(save_dir / "feature_columns.yaml") as f:
        meta = yaml.safe_load(f)
    feature_columns = meta["feature_columns"]
    model_config = load_model_config()
    return booster, feature_columns, model_config


def align_features(X: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    """Ensure X has exactly the same columns as training; add 0 for missing, drop extra."""
    for c in feature_columns:
        if c not in X.columns:
            X = X.copy()
            X[c] = 0
    X = X[[c for c in feature_columns if c in X.columns]]
    # If still missing (e.g. different order), reindex
    missing = [c for c in feature_columns if c not in X.columns]
    if missing:
        for c in missing:
            X[c] = 0
    return X[feature_columns]


def predict_top_k(
    user_id: str,
    restaurant_id: str,
    cart_item_ids: list[str],
    menu_items: pd.DataFrame,
    user_features: pd.DataFrame,
    restaurant_features: pd.DataFrame,
    item_features: pd.DataFrame,
    booster: lgb.Booster,
    feature_columns: list[str],
    top_k: int = 10,
    timestamp: pd.Timestamp | str | None = None,
    city: str | None = None,
    zone: str | None = None,
    config: dict[str, Any] | None = None,
) -> tuple[list[dict], float]:
    """
    Return list of {item_id, score, rank} for top-K add-on recommendations, and latency_ms.
    """
    config = config or load_feature_config()
    inf_cfg = load_model_config().get("inference", {})
    k = top_k or inf_cfg.get("top_k", 10)

    t0 = time.perf_counter()
    X, candidate_ids = build_inference_matrix(
        user_id, restaurant_id, cart_item_ids, menu_items,
        user_features, restaurant_features, item_features,
        timestamp=timestamp, city=city, zone=zone, config=config,
    )
    if X.empty or candidate_ids.empty:
        return [], (time.perf_counter() - t0) * 1000

    X = align_features(X, feature_columns)
    X = ensure_numeric(X)
    scores = booster.predict(X)
    order = np.argsort(-scores)[:k]
    out = []
    for r, idx in enumerate(order, start=1):
        out.append({
            "item_id": candidate_ids.iloc[idx],
            "score": float(scores[idx]),
            "rank": r,
        })
    latency_ms = (time.perf_counter() - t0) * 1000
    return out, latency_ms
