"""
FastAPI service for CSAO recommendations: POST /recommend, /health, /ready.
"""
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.features.config import load_feature_config
from src.features.user_features import build_user_features
from src.features.restaurant_features import build_restaurant_features, build_item_features
from src.features.cold_start import cold_start_recommend, is_cold_start
from src.model.predict import load_model_and_metadata, predict_top_k


# Global state for model and feature tables (loaded at startup)
_state: dict[str, Any] = {}


def _project_root() -> Path:
    """Project root (directory containing configs/ and data/)."""
    root = Path(__file__).resolve().parent.parent.parent
    if (root / "configs" / "features.yaml").exists():
        return root
    # Fallback when run from other cwd (e.g. tests)
    cwd = Path.cwd()
    if (cwd / "configs" / "features.yaml").exists():
        return cwd
    return root


def _load_data_and_features():
    """Load CSVs and build user/restaurant/item feature tables."""
    root = _project_root()
    data_dir = root / "data"
    orders = pd.read_csv(data_dir / "orders.csv")
    order_items = pd.read_csv(data_dir / "order_items.csv")
    menu_items = pd.read_csv(data_dir / "menu_items.csv")
    users = pd.read_csv(data_dir / "users.csv")
    restaurants_raw = pd.read_csv(data_dir / "restaurants.csv")
    config = load_feature_config()
    restaurants = build_restaurant_features(
        restaurants_raw, menu_items, order_items, orders, config
    )
    item_features = build_item_features(menu_items, order_items, orders)
    user_features = build_user_features(
        orders, users, restaurants, order_items, menu_items, config
    )
    return {
        "menu_items": menu_items,
        "order_items": order_items,
        "user_features": user_features,
        "restaurant_features": restaurants,
        "item_features": item_features,
        "config": config,
    }


@asynccontextmanager
async def lifespan(app: FastAPI):
    root = _project_root()
    models_dir = root / "models"
    if (models_dir / "csao_lgb.txt").exists():
        booster, feature_columns, model_config = load_model_and_metadata(models_dir)
        _state["booster"] = booster
        _state["feature_columns"] = feature_columns
        _state["model_config"] = model_config
    else:
        _state["booster"] = _state["feature_columns"] = _state["model_config"] = None
    try:
        data = _load_data_and_features()
        _state["menu_items"] = data["menu_items"]
        _state["order_items"] = data["order_items"]
        _state["user_features"] = data["user_features"]
        _state["restaurant_features"] = data["restaurant_features"]
        _state["item_features"] = data["item_features"]
        _state["config"] = data["config"]
    except Exception:
        _state["menu_items"] = None
        _state["order_items"] = None
        _state["user_features"] = None
        _state["restaurant_features"] = None
        _state["item_features"] = None
        _state["config"] = load_feature_config()
    yield
    _state.clear()


app = FastAPI(title="CSAO Recommendation API", lifespan=lifespan)


class RecommendRequest(BaseModel):
    user_id: str = Field(..., description="User identifier")
    restaurant_id: str = Field(..., description="Restaurant identifier")
    cart_item_ids: list[str] = Field(default_factory=list, description="Current cart item IDs")
    top_k: int | None = Field(default=10, description="Number of recommendations (default 10)")
    timestamp: str | None = Field(None, description="ISO timestamp for context")
    city: str | None = Field(None, description="Delivery city")
    zone: str | None = Field(None, description="Delivery zone")


class RecommendationItem(BaseModel):
    item_id: str
    score: float
    rank: int


class RecommendResponse(BaseModel):
    recommendations: list[RecommendationItem]
    latency_ms: float


def _meal_bucket_from_timestamp(ts: str | None) -> str | None:
    if not ts:
        return None
    try:
        from datetime import datetime
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        h = dt.hour
        if 6 <= h < 11:
            return "breakfast"
        if 11 <= h < 15:
            return "lunch"
        if 15 <= h < 21:
            return "dinner"
        return "late_night"
    except Exception:
        return None


@app.post("/recommend", response_model=RecommendResponse)
def recommend(req: RecommendRequest):
    """Return ranked add-on recommendations for the given cart and context."""
    if _state.get("menu_items") is None:
        raise HTTPException(status_code=503, detail="Data not loaded.")

    # Cold start: use popularity-based fallback for new users/restaurants
    if is_cold_start(
        req.user_id,
        req.restaurant_id,
        _state.get("user_features"),
        _state.get("restaurant_features"),
        _state["menu_items"],
    ) or _state.get("booster") is None:
        order_items = _state.get("order_items")
        if order_items is not None and not order_items.empty:
            recs = cold_start_recommend(
                req.restaurant_id,
                req.cart_item_ids,
                _state["menu_items"],
                order_items,
                top_k=req.top_k or 10,
                config=_state["config"],
                meal_bucket=_meal_bucket_from_timestamp(req.timestamp),
            )
            return RecommendResponse(
                recommendations=[RecommendationItem(item_id=r["item_id"], score=r["score"], rank=r["rank"]) for r in recs],
                latency_ms=0.0,
            )
        return RecommendResponse(recommendations=[], latency_ms=0.0)

    recs, latency_ms = predict_top_k(
        user_id=req.user_id,
        restaurant_id=req.restaurant_id,
        cart_item_ids=req.cart_item_ids,
        menu_items=_state["menu_items"],
        user_features=_state["user_features"],
        restaurant_features=_state["restaurant_features"],
        item_features=_state["item_features"],
        booster=_state["booster"],
        feature_columns=_state["feature_columns"],
        top_k=req.top_k or 10,
        timestamp=req.timestamp,
        city=req.city,
        zone=req.zone,
        config=_state["config"],
    )
    return RecommendResponse(
        recommendations=[RecommendationItem(item_id=r["item_id"], score=r["score"], rank=r["rank"]) for r in recs],
        latency_ms=round(latency_ms, 2),
    )


@app.get("/health")
def health():
    """Liveness: service is running."""
    return {"status": "ok"}


@app.get("/ready")
def ready():
    """Readiness: model and data loaded."""
    return {
        "ready": _state.get("booster") is not None and _state.get("menu_items") is not None,
        "model_loaded": _state.get("booster") is not None,
        "data_loaded": _state.get("menu_items") is not None,
    }
