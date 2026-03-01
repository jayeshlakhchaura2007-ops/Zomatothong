"""Baseline recommender: rank candidates by item popularity (order count)."""
from typing import Any

import pandas as pd

from src.features.candidates import get_candidates


def baseline_recommend(
    restaurant_id: str,
    cart_item_ids: list[str],
    menu_items: pd.DataFrame,
    item_order_counts: pd.DataFrame,
    top_k: int = 10,
    config: dict[str, Any] | None = None,
) -> list[dict]:
    """
    Return top-K candidate item_ids ranked by item_order_count (popularity).
    item_order_counts: DataFrame with columns item_id, item_order_count (or quantity sum).
    """
    candidates_df = get_candidates(restaurant_id, cart_item_ids, menu_items, config)
    if candidates_df.empty:
        return []
    merged = candidates_df.merge(
        item_order_counts[["item_id", "item_order_count"]],
        on="item_id",
        how="left",
    )
    merged["item_order_count"] = merged["item_order_count"].fillna(0)
    top = merged.nlargest(top_k, "item_order_count")
    return [
        {"item_id": row["item_id"], "score": float(row["item_order_count"]), "rank": r}
        for r, (_, row) in enumerate(top.iterrows(), start=1)
    ]
