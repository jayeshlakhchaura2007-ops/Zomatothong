"""
Offline evaluation: run model and baseline on test set, compute AUC/P@K/R@K/NDCG, compare.
"""
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from src.features.config import load_feature_config
from src.features.pipeline import build_training_matrix
from src.features.restaurant_features import build_item_features
from src.model.config import load_model_config
from src.model.predict import load_model_and_metadata, align_features
from src.model.numeric_utils import ensure_numeric
from src.model.train import temporal_split_orders
from src.evaluation.metrics import compute_metrics
from src.evaluation.baseline import baseline_recommend


def run_evaluation(
    data_dir: str | Path | None = None,
    models_dir: str | Path | None = None,
    k: int = 10,
) -> dict[str, Any]:
    """
    Load test orders, generate (cart, add-on) pairs with labels.
    Score with model and baseline; compute metrics per (request) and aggregate.
    Returns dict with model_metrics, baseline_metrics, comparison.
    """
    data_dir = Path(data_dir or Path(__file__).resolve().parent.parent.parent / "data")
    models_dir = Path(models_dir or Path(__file__).resolve().parent.parent.parent / "models")

    orders = pd.read_csv(data_dir / "orders.csv")
    order_items = pd.read_csv(data_dir / "order_items.csv")
    menu_items = pd.read_csv(data_dir / "menu_items.csv")
    users = pd.read_csv(data_dir / "users.csv")
    restaurants = pd.read_csv(data_dir / "restaurants.csv")
    config = load_feature_config()
    from src.features.restaurant_features import build_restaurant_features
    from src.features.user_features import build_user_features
    restaurants_f = build_restaurant_features(restaurants, menu_items, order_items, orders, config)
    item_features = build_item_features(menu_items, order_items, orders)
    user_features = build_user_features(orders, users, restaurants_f, order_items, menu_items, config)
    item_order_counts = order_items.groupby("item_id")["quantity"].sum().reset_index()
    item_order_counts.columns = ["item_id", "item_order_count"]

    _, _, test_ids = temporal_split_orders(orders, 0.7, 0.15, 0.15)
    orders["created_at"] = pd.to_datetime(orders["created_at"])
    test_orders = orders[orders["order_id"].isin(test_ids)]

    # Build test matrix with labels (same as training logic)
    X_full, y_full, order_ids = build_training_matrix(
        orders, order_items, menu_items, users, restaurants, config
    )
    test_mask = order_ids.isin(test_ids)
    X_test = X_full[test_mask]
    y_test = y_full[test_mask]
    feature_cols = list(X_full.columns)

    results = {"model": {}, "baseline": {}, "comparison": {}}

    # Model metrics (pointwise on test matrix)
    if (models_dir / "csao_lgb.txt").exists():
        booster, saved_cols, _ = load_model_and_metadata(models_dir)
        X_test_aligned = align_features(X_test.copy(), saved_cols)
        X_test_aligned = ensure_numeric(X_test_aligned)
        y_pred_proba = booster.predict(X_test_aligned)
        y_pred_bin = (y_pred_proba >= 0.5).astype(int)
        results["model"] = compute_metrics(
            y_test.values, y_pred_proba, y_pred_bin, k=k
        )
    else:
        results["model"] = {"note": "Model not found; run python -m src.model.train"}

    # Baseline: for each test row we have (order, candidate_item, label).
    # Baseline would rank by item_order_count; so for same (order, candidates) we assign scores = item_order_count.
    # Then we can compute same metrics if we treat baseline score as y_score.
    order_items_test = order_items[order_items["order_id"].isin(test_ids)]
    baseline_scores = X_test.copy()
    pop_col = "item_item_order_count" if "item_item_order_count" in baseline_scores.columns else "item_order_count"
    if pop_col in baseline_scores.columns:
        baseline_y_score = baseline_scores[pop_col].values.astype(float)
    else:
        baseline_y_score = np.zeros(len(baseline_scores), dtype=float)
    baseline_y_score = np.asarray(baseline_y_score, dtype=float)
    results["baseline"] = compute_metrics(
        y_test.values, baseline_y_score, (baseline_y_score >= np.median(baseline_y_score)).astype(int), k=k
    )

    for key in ["auc", f"precision_at_{k}", f"recall_at_{k}", f"ndcg_at_{k}"]:
        if key in results["model"] and key in results["baseline"]:
            m, b = results["model"][key], results["baseline"][key]
            results["comparison"][f"{key}_lift"] = round((m - b) / b * 100, 2) if b else 0

    return results


def main():
    results = run_evaluation()
    out_path = Path(__file__).resolve().parent.parent.parent / "models" / "evaluation_report.yaml"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        yaml.dump(results, f)
    print("Evaluation report:")
    print(yaml.dump(results, default_flow_style=False))
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
