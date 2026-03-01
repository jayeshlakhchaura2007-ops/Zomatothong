"""Cart context features: composition, value, incomplete meal signals."""
from typing import Any

import pandas as pd


def _meal_bucket(hour: int) -> str:
    if 6 <= hour < 11:
        return "breakfast"
    if 11 <= hour < 15:
        return "lunch"
    if 15 <= hour < 21:
        return "dinner"
    return "late_night"


def build_cart_context(
    cart_item_ids: list[str],
    menu_items: pd.DataFrame,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Build cart-level features from current cart item IDs.
    Returns a dict suitable for merging into request-level features.
    """
    config = config or {}
    ccfg = config.get("cart", {})

    if not cart_item_ids:
        return {
            "cart_item_count": 0,
            "cart_total_value": 0.0,
            "cart_veg_ratio": 0.0,
            "cart_has_main": 0,
            "cart_has_beverage": 0,
            "cart_has_dessert": 0,
            "cart_main_without_beverage": 0,
            "cart_category_main": 0,
            "cart_category_beverage": 0,
            "cart_category_dessert": 0,
            "cart_category_side": 0,
            "cart_category_addon": 0,
        }

    cart_items = menu_items[menu_items["item_id"].isin(cart_item_ids)]
    if cart_items.empty:
        cart_items = pd.DataFrame(columns=menu_items.columns)

    n = len(cart_items)
    total_value = float(cart_items["price"].sum()) if n else 0.0
    veg_ratio = float(cart_items["veg"].mean()) if n else 0.0
    categories = cart_items["category"].value_counts()

    has_main = 1 if (categories.get("main", 0) > 0) else 0
    has_beverage = 1 if (categories.get("beverage", 0) > 0) else 0
    has_dessert = 1 if (categories.get("dessert", 0) > 0) else 0
    main_without_beverage = 1 if (has_main and not has_beverage) else 0

    out = {
        "cart_item_count": n,
        "cart_total_value": round(total_value, 2),
        "cart_veg_ratio": round(veg_ratio, 4),
        "cart_has_main": has_main,
        "cart_has_beverage": has_beverage,
        "cart_has_dessert": has_dessert,
        "cart_main_without_beverage": main_without_beverage,
        "cart_category_main": int(categories.get("main", 0)),
        "cart_category_beverage": int(categories.get("beverage", 0)),
        "cart_category_dessert": int(categories.get("dessert", 0)),
        "cart_category_side": int(categories.get("side", 0)),
        "cart_category_addon": int(categories.get("addon", 0)),
    }
    return out


def build_context_features(
    timestamp: pd.Timestamp | str | None = None,
    city: str | None = None,
    zone: str | None = None,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Temporal and geographic context."""
    ts = pd.Timestamp(timestamp) if timestamp else pd.Timestamp.now()
    hour = ts.hour
    day_of_week = ts.dayofweek
    meal_bucket = _meal_bucket(hour)
    return {
        "context_hour": hour,
        "context_day_of_week": day_of_week,
        "context_meal_breakfast": 1 if meal_bucket == "breakfast" else 0,
        "context_meal_lunch": 1 if meal_bucket == "lunch" else 0,
        "context_meal_dinner": 1 if meal_bucket == "dinner" else 0,
        "context_meal_late_night": 1 if meal_bucket == "late_night" else 0,
        "context_city": str(city) if city else "",
        "context_zone": str(zone) if zone else "",
    }
