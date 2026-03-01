"""Restaurant and item-level features for CSAO."""
from typing import Any

import pandas as pd


def build_restaurant_features(
    restaurants: pd.DataFrame,
    menu_items: pd.DataFrame,
    order_items: pd.DataFrame,
    orders: pd.DataFrame,
    config: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Restaurant-level features (one row per restaurant_id)."""
    config = config or {}
    rcfg = config.get("restaurant", {})

    rest = restaurants.copy()
    if "price_tier" not in rest.columns and "price_tier" in rcfg:
        rest["price_tier"] = "mid"

    # Item popularity: order count per item, then aggregate to restaurant
    if rcfg.get("item_popularity", True):
        item_orders = order_items.merge(
            orders[["order_id", "restaurant_id"]], on="order_id", how="left"
        )
        item_pop = (
            item_orders.groupby(["restaurant_id", "item_id"])["quantity"]
            .sum()
            .reset_index()
            .rename(columns={"quantity": "item_orders"})
        )
        rest_pop = item_pop.groupby("restaurant_id")["item_orders"].sum().reset_index()
        rest_pop.columns = ["restaurant_id", "total_item_orders"]
        rest = rest.merge(rest_pop, on="restaurant_id", how="left")
        rest["total_item_orders"] = rest["total_item_orders"].fillna(0).astype(int)
    return rest


def build_item_features(
    menu_items: pd.DataFrame,
    order_items: pd.DataFrame,
    orders: pd.DataFrame,
) -> pd.DataFrame:
    """Item-level features: price, category, veg, popularity (order count)."""
    items = menu_items.copy()
    items["price_log"] = (items["price"] + 1).apply(lambda x: round(__import__("math").log(x), 4))
    item_orders = (
        order_items.groupby("item_id")["quantity"]
        .sum()
        .reset_index()
        .rename(columns={"quantity": "item_order_count"})
    )
    items = items.merge(item_orders, on="item_id", how="left")
    items["item_order_count"] = items["item_order_count"].fillna(0).astype(int)
    return items
