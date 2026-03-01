"""User-level features for CSAO (frequency, recency, segment, preferences)."""
from typing import Any

import pandas as pd


def build_user_features(
    orders: pd.DataFrame,
    users: pd.DataFrame,
    restaurants: pd.DataFrame,
    order_items: pd.DataFrame,
    menu_items: pd.DataFrame,
    config: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Build user feature table: one row per user_id."""
    config = config or {}
    ucfg = config.get("user", {})

    # Merge orders with restaurant for cuisine (orders already have zone/city)
    ord_rest = orders.merge(
        restaurants[["restaurant_id", "cuisine"]],
        on="restaurant_id",
        how="left",
    )
    if "zone" not in ord_rest.columns:
        ord_rest["zone"] = ord_rest.get("city", "")
    ord_rest["created_at"] = pd.to_datetime(ord_rest["created_at"])

    agg = ord_rest.groupby("user_id").agg(
        order_count=("order_id", "count"),
        total_value=("total_value", "sum"),
        last_order_at=("created_at", "max"),
    ).reset_index()
    agg["order_frequency"] = agg["order_count"]
    agg["monetary_value_avg"] = (agg["total_value"] / agg["order_count"]).round(2)
    agg["recency_days"] = (
        (pd.Timestamp.now() - agg["last_order_at"]).dt.total_seconds() / 86400
    ).round(0).astype(int)

    # Segment from users if present
    if "segment" in users.columns and ucfg.get("segment", True):
        agg = agg.merge(users[["user_id", "segment"]], on="user_id", how="left")
    else:
        agg["segment"] = "occasional"

    # Preferred cuisines (top-k)
    k_cuisine = ucfg.get("preferred_cuisines_top_k", 5)
    cuisine_counts = (
        ord_rest.groupby(["user_id", "cuisine"])
        .size()
        .reset_index(name="cnt")
    )
    top_cuisines = (
        cuisine_counts.sort_values(["user_id", "cnt"], ascending=[True, False])
        .groupby("user_id")
        .head(k_cuisine)
    )
    top_cuisines["cuisine_rank"] = top_cuisines.groupby("user_id").cumcount() + 1
    cuisine_pivot = top_cuisines.pivot(
        index="user_id", columns="cuisine_rank", values="cuisine"
    ).add_prefix("preferred_cuisine_")
    agg = agg.merge(cuisine_pivot.reset_index(), on="user_id", how="left")

    # Preferred zones (top-k)
    k_zone = ucfg.get("preferred_zones_top_k", 3)
    zone_counts = ord_rest.groupby(["user_id", "zone"]).size().reset_index(name="cnt")
    top_zones = (
        zone_counts.sort_values(["user_id", "cnt"], ascending=[True, False])
        .groupby("user_id")
        .head(k_zone)
    )
    top_zones["zone_rank"] = top_zones.groupby("user_id").cumcount() + 1
    zone_pivot = top_zones.pivot(
        index="user_id", columns="zone_rank", values="zone"
    ).add_prefix("preferred_zone_")
    agg = agg.merge(zone_pivot.reset_index(), on="user_id", how="left")

    return agg
