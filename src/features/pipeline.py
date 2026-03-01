"""
Full feature pipeline: build (request, candidate_item) feature matrix for training and inference.
"""
from pathlib import Path
from typing import Any

import pandas as pd

from src.features.config import load_feature_config
from src.features.user_features import build_user_features
from src.features.restaurant_features import build_restaurant_features, build_item_features
from src.features.cart_features import build_cart_context, build_context_features
from src.features.candidates import get_candidates


def _encode_categorical(df: pd.DataFrame, object_cols: list[str]) -> pd.DataFrame:
    """Label-encode object columns for model."""
    out = df.copy()
    for col in object_cols:
        if col in out.columns and out[col].dtype == object:
            out[col] = pd.Categorical(out[col]).codes
    return out


def build_training_matrix(
    orders: pd.DataFrame,
    order_items: pd.DataFrame,
    menu_items: pd.DataFrame,
    users: pd.DataFrame,
    restaurants: pd.DataFrame,
    config: dict[str, Any] | None = None,
    add_on_definition: str = "after_first",
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    Build feature matrix for training.
    add_on_definition: "after_first" = items after the first in each order are add-ons.
    Returns (X, y, order_ids) for temporal split.
    For each order we simulate cart = first k items, label = 1 for add-on items that were ordered, 0 for other candidates.
    """
    config = config or load_feature_config()
    restaurants = build_restaurant_features(
        restaurants, menu_items, order_items, orders, config
    )
    item_features = build_item_features(menu_items, order_items, orders)
    user_features = build_user_features(
        orders, users, restaurants, order_items, menu_items, config
    )

    # Temporal split uses order date; we still need one big matrix with labels
    orders = orders.copy()
    orders["created_at"] = pd.to_datetime(orders["created_at"])
    order_items = order_items.merge(
        orders[["order_id", "user_id", "restaurant_id", "created_at", "city", "zone"]],
        on="order_id",
        how="left",
    )
    order_items = order_items.merge(
        menu_items[["item_id", "category", "veg", "price"]],
        on="item_id",
        how="left",
    )

    rows = []
    for order_id, grp in order_items.groupby("order_id"):
        row0 = grp.iloc[0]
        user_id = row0["user_id"]
        restaurant_id = row0["restaurant_id"]
        created_at = row0["created_at"]
        city, zone = row0.get("city", ""), row0.get("zone", "")

        # Cart = items in order (we use full order as "cart" for training simplicity; add-on = later items)
        item_list = grp["item_id"].tolist()
        if add_on_definition == "after_first":
            cart_ids = item_list[:1]  # cart is first item only
            add_on_ids = set(item_list[1:])
        else:
            cart_ids = item_list
            add_on_ids = set()

        cart_ctx = build_cart_context(cart_ids, menu_items, config)
        ctx = build_context_features(created_at, city, zone, config)

        candidates_df = get_candidates(
            restaurant_id, cart_ids, menu_items, config
        )
        if candidates_df.empty:
            continue

        user_row = user_features[user_features["user_id"] == user_id]
        rest_row = restaurants[restaurants["restaurant_id"] == restaurant_id]
        if user_row.empty:
            user_row = pd.DataFrame([{c: 0 for c in user_features.columns}])
            user_row["user_id"] = user_id
        else:
            user_row = user_row.iloc[[0]]
        if rest_row.empty:
            rest_row = pd.DataFrame([{c: 0 for c in restaurants.columns}])
            rest_row["restaurant_id"] = restaurant_id
        else:
            rest_row = rest_row.iloc[[0]]

        for _, cand in candidates_df.iterrows():
            item_id = cand["item_id"]
            label = 1 if item_id in add_on_ids else 0
            item_row = item_features[item_features["item_id"] == item_id]
            if item_row.empty:
                item_row = pd.DataFrame([{
                    "item_id": item_id, "price": cand["price"], "category": cand["category"],
                    "veg": cand["veg"], "price_log": 0, "item_order_count": 0,
                }])
            else:
                item_row = item_row.iloc[[0]].copy()

            # Flatten into one row
            rec = {}
            for c in user_row.columns:
                if c != "user_id":
                    rec[f"user_{c}"] = user_row[c].iloc[0]
            for c in rest_row.columns:
                if c != "restaurant_id":
                    rec[f"rest_{c}"] = rest_row[c].iloc[0]
            for k, v in cart_ctx.items():
                rec[k] = v
            for k, v in ctx.items():
                rec[k] = v
            for c in item_row.columns:
                if c != "item_id":
                    rec[f"item_{c}"] = item_row[c].iloc[0]
            rec["candidate_item_id"] = item_id
            rec["order_id"] = order_id
            rec["user_id"] = user_id
            rec["restaurant_id"] = restaurant_id
            rec["label"] = label
            rows.append(rec)

    df = pd.DataFrame(rows)
    if df.empty:
        return df, pd.Series(dtype=int), pd.Series(dtype=object)
    order_ids = df["order_id"].copy()
    labels = df["label"].copy()
    df = df.drop(columns=["label", "order_id", "user_id", "restaurant_id", "candidate_item_id"], errors="ignore")
    object_cols = [c for c in df.columns if df[c].dtype == object]
    df = _encode_categorical(df, object_cols)
    # Convert datetime to numeric (seconds since epoch)
    for c in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[c]):
            df[c] = (df[c] - pd.Timestamp("1970-01-01")).dt.total_seconds().astype(float)
    # Encode any remaining object columns
    object_cols2 = [c for c in df.columns if df[c].dtype == object]
    df = _encode_categorical(df, object_cols2)
    df = df.fillna(0)
    # Ensure numeric only (LightGBM)
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = pd.Categorical(df[c]).codes
    return df, labels, order_ids


def build_inference_matrix(
    user_id: str,
    restaurant_id: str,
    cart_item_ids: list[str],
    menu_items: pd.DataFrame,
    user_features: pd.DataFrame,
    restaurant_features: pd.DataFrame,
    item_features: pd.DataFrame,
    timestamp: pd.Timestamp | str | None = None,
    city: str | None = None,
    zone: str | None = None,
    config: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Build feature matrix for one request: one row per candidate item.
    Returns (feature_df, candidate_item_ids) so caller can map scores back to item_id.
    """
    config = config or load_feature_config()
    cart_ctx = build_cart_context(cart_item_ids, menu_items, config)
    ctx = build_context_features(timestamp, city, zone, config)
    candidates_df = get_candidates(restaurant_id, cart_item_ids, menu_items, config)
    if candidates_df.empty:
        return pd.DataFrame(), pd.Series(dtype=object)

    user_row = user_features[user_features["user_id"] == user_id]
    rest_row = restaurant_features[restaurant_features["restaurant_id"] == restaurant_id]
    if user_row.empty:
        user_row = pd.DataFrame([{c: 0 for c in user_features.columns if c != "user_id"}])
        user_row["user_id"] = user_id
    else:
        user_row = user_row.iloc[[0]]
    if rest_row.empty:
        rest_row = pd.DataFrame([{c: 0 for c in restaurant_features.columns if c != "restaurant_id"}])
        rest_row["restaurant_id"] = restaurant_id
    else:
        rest_row = rest_row.iloc[[0]]

    rows = []
    candidate_ids = []
    for _, cand in candidates_df.iterrows():
        item_id = cand["item_id"]
        item_row = item_features[item_features["item_id"] == item_id]
        if item_row.empty:
            item_row = pd.DataFrame([{
                "item_id": item_id, "price": cand["price"], "category": cand["category"],
                "veg": cand["veg"], "price_log": 0, "item_order_count": 0,
            }])
        else:
            item_row = item_row.iloc[[0]].copy()

        rec = {}
        for c in user_row.columns:
            if c != "user_id":
                rec[f"user_{c}"] = user_row[c].iloc[0]
        for c in rest_row.columns:
            if c != "restaurant_id":
                rec[f"rest_{c}"] = rest_row[c].iloc[0]
        for k, v in cart_ctx.items():
            rec[k] = v
        for k, v in ctx.items():
            rec[k] = v
        for c in item_row.columns:
            if c != "item_id":
                rec[f"item_{c}"] = item_row[c].iloc[0]
        rows.append(rec)
        candidate_ids.append(item_id)

    df = pd.DataFrame(rows)
    object_cols = [c for c in df.columns if df[c].dtype == object]
    df = _encode_categorical(df, object_cols)
    for c in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[c]):
            df[c] = (df[c] - pd.Timestamp("1970-01-01")).dt.total_seconds().astype(float)
    object_cols2 = [c for c in df.columns if df[c].dtype == object]
    df = _encode_categorical(df, object_cols2)
    df = df.fillna(0)
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = pd.Categorical(df[c]).codes
    return df, pd.Series(candidate_ids)
