"""
Cold-start and diversity/fairness safeguards.
- Cold start: when user, restaurant, or item is new, fall back to popularity-based recommendations.
- Diversity: ensure top-K is not dominated by a single category (e.g. mix beverages, desserts, sides).
"""
from typing import Any

import pandas as pd


def cold_start_recommend(
    restaurant_id: str,
    cart_item_ids: list[str],
    menu_items: pd.DataFrame,
    order_items: pd.DataFrame,
    top_k: int = 10,
    config: dict[str, Any] | None = None,
    meal_bucket: str | None = None,
) -> list[dict]:
    """
    Fallback when model cannot be used (new user/restaurant/item).
    Rank by: same-restaurant item popularity (order count). Optionally bias by meal_bucket
    (e.g. breakfast -> prefer beverages; dinner -> desserts).
    """
    from src.features.candidates import get_candidates

    candidates_df = get_candidates(restaurant_id, cart_item_ids, menu_items, config)
    if candidates_df.empty:
        return []

    item_counts = (
        order_items.groupby("item_id")["quantity"]
        .sum()
        .reset_index()
        .rename(columns={"quantity": "item_order_count"})
    )
    merged = candidates_df.merge(item_counts, on="item_id", how="left")
    merged["item_order_count"] = merged["item_order_count"].fillna(0)

    # Optional: boost by category for meal time (diversity + relevance)
    if meal_bucket == "breakfast":
        merged["score"] = merged["item_order_count"] + (merged["category"] == "beverage").astype(int) * 10
    elif meal_bucket == "dinner":
        merged["score"] = merged["item_order_count"] + (merged["category"] == "dessert").astype(int) * 5
    else:
        merged["score"] = merged["item_order_count"]

    top = merged.nlargest(top_k * 2, "score")  # get more then apply diversity
    return _apply_diversity(top, top_k)


def _apply_diversity(candidates_ranked: pd.DataFrame, top_k: int) -> list[dict]:
    """
    Ensure top-K is not all same category: take at most 2-3 per category in top-K
    so that add-ons are diverse (beverages, desserts, sides).
    """
    out = []
    taken_per_cat: dict[str, int] = {}
    max_per_cat = max(2, (top_k + 2) // 3)
    for _, row in candidates_ranked.iterrows():
        if len(out) >= top_k:
            break
        cat = row.get("category", "other")
        if taken_per_cat.get(cat, 0) >= max_per_cat:
            continue
        taken_per_cat[cat] = taken_per_cat.get(cat, 0) + 1
        out.append({
            "item_id": row["item_id"],
            "score": float(row["score"]),
            "rank": len(out) + 1,
        })
    # Fill remaining if we have slack
    for _, row in candidates_ranked.iterrows():
        if len(out) >= top_k:
            break
        if row["item_id"] in [x["item_id"] for x in out]:
            continue
        out.append({
            "item_id": row["item_id"],
            "score": float(row["score"]),
            "rank": len(out) + 1,
        })
    return out[:top_k]


def is_cold_start(
    user_id: str,
    restaurant_id: str,
    user_features: pd.DataFrame,
    restaurant_features: pd.DataFrame,
    menu_items: pd.DataFrame,
) -> bool:
    """True if user or restaurant is unknown (no history)."""
    user_known = user_id in user_features["user_id"].values if user_features is not None and not user_features.empty else False
    rest_known = restaurant_id in restaurant_features["restaurant_id"].values if restaurant_features is not None and not restaurant_features.empty else False
    rest_has_items = (menu_items["restaurant_id"] == restaurant_id).any() if menu_items is not None and not menu_items.empty else False
    return not (user_known and rest_known and rest_has_items)
