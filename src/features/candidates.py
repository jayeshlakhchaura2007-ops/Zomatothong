"""Candidate add-on generation: same restaurant, exclude cart, optional category filter."""
from typing import Any

import pandas as pd


def get_candidates(
    restaurant_id: str,
    cart_item_ids: list[str],
    menu_items: pd.DataFrame,
    config: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """
    Return candidate items for add-on recommendation.
    Same restaurant only; exclude items already in cart; optional category filter.
    """
    config = config or {}
    cand_cfg = config.get("candidates", {})
    max_per_request = cand_cfg.get("max_per_request", 50)
    exclude_cart = cand_cfg.get("exclude_cart_items", True)
    category_filters = cand_cfg.get("category_filters") or []

    cand = menu_items[menu_items["restaurant_id"] == restaurant_id].copy()
    if cand.empty:
        return pd.DataFrame()

    if exclude_cart and cart_item_ids:
        cand = cand[~cand["item_id"].isin(cart_item_ids)]
    if category_filters:
        cand = cand[cand["category"].isin(category_filters)]
    cand = cand.head(max_per_request)
    return cand
