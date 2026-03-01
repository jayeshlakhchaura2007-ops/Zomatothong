"""
Training pipeline: temporal split, LightGBM binary classification, early stopping.
Saves model and feature names for inference.
"""
from pathlib import Path
from typing import Any

import pandas as pd
import lightgbm as lgb
import yaml

from src.features.config import load_feature_config
from src.features.pipeline import build_training_matrix
from src.model.config import load_model_config
from src.evaluation.metrics import compute_metrics
from src.model.numeric_utils import ensure_numeric


def temporal_split_orders(
    orders: pd.DataFrame,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Split order_ids by order created_at (temporal)."""
    orders = orders.copy()
    orders["created_at"] = pd.to_datetime(orders["created_at"])
    orders = orders.sort_values("created_at").reset_index(drop=True)
    n = len(orders)
    t1 = int(n * train_ratio)
    t2 = int(n * (train_ratio + val_ratio))
    train_ids = orders.iloc[:t1]["order_id"]
    val_ids = orders.iloc[t1:t2]["order_id"]
    test_ids = orders.iloc[t2:]["order_id"]
    return train_ids, val_ids, test_ids


def train(
    data_dir: str | Path | None = None,
    feature_config: dict[str, Any] | None = None,
    model_config: dict[str, Any] | None = None,
    save_dir: str | Path | None = None,
) -> tuple[lgb.LGBMClassifier, list[str], dict[str, Any]]:
    """
    Load data, build features, temporal split, train LightGBM.
    Returns (model, feature_columns, metrics_dict).
    """
    data_dir = Path(data_dir or Path(__file__).resolve().parent.parent.parent / "data")
    save_dir = Path(save_dir or Path(__file__).resolve().parent.parent.parent / "models")
    save_dir.mkdir(parents=True, exist_ok=True)

    feature_config = feature_config or load_feature_config()
    model_config = model_config or load_model_config()
    mcfg = model_config.get("model", {})
    tcfg = model_config.get("training", {})
    split_cfg = tcfg.get("temporal_split", {})

    orders = pd.read_csv(data_dir / "orders.csv")
    order_items = pd.read_csv(data_dir / "order_items.csv")
    menu_items = pd.read_csv(data_dir / "menu_items.csv")
    users = pd.read_csv(data_dir / "users.csv")
    restaurants = pd.read_csv(data_dir / "restaurants.csv")

    X_full, y_full, order_ids = build_training_matrix(
        orders, order_items, menu_items, users, restaurants, feature_config
    )
    if X_full.empty or len(X_full) < 50:
        raise ValueError("Training matrix too small. Generate sample data: python -m src.data.generate_sample_data")

    train_ids, val_ids, test_ids = temporal_split_orders(
        orders,
        split_cfg.get("train_ratio", 0.7),
        split_cfg.get("val_ratio", 0.15),
        split_cfg.get("test_ratio", 0.15),
    )
    train_mask = order_ids.isin(train_ids)
    val_mask = order_ids.isin(val_ids)
    test_mask = order_ids.isin(test_ids)

    X_train, X_val, X_test = X_full[train_mask], X_full[val_mask], X_full[test_mask]
    y_train, y_val, y_test = y_full[train_mask], y_full[val_mask], y_full[test_mask]
    feature_cols = list(X_full.columns)

    # Ensure all columns are numeric (LightGBM)
    X_train = ensure_numeric(X_train)
    X_val = ensure_numeric(X_val)
    X_test = ensure_numeric(X_test)

    params = {
        "objective": mcfg.get("objective", "binary"),
        "metric": mcfg.get("metric", "auc"),
        "num_leaves": mcfg.get("num_leaves", 31),
        "max_depth": mcfg.get("max_depth", 8),
        "learning_rate": mcfg.get("learning_rate", 0.05),
        "n_estimators": mcfg.get("n_estimators", 500),
        "min_child_samples": mcfg.get("min_child_samples", 20),
        "subsample": mcfg.get("subsample", 0.8),
        "colsample_bytree": mcfg.get("colsample_bytree", 0.8),
        "reg_alpha": mcfg.get("reg_alpha", 0.1),
        "reg_lambda": mcfg.get("reg_lambda", 0.1),
        "random_state": mcfg.get("random_state", 42),
        "n_jobs": mcfg.get("n_jobs", -1),
        "verbose": mcfg.get("verbose", -1),
    }
    model = lgb.LGBMClassifier(**params)
    early_stopping = tcfg.get("early_stopping_rounds", 50)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(early_stopping, verbose=False)],
    )

    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred_bin = (y_pred_proba >= 0.5).astype(int)
    metrics = compute_metrics(y_test.values, y_pred_proba, y_pred_bin, k=10)
    metrics["train_size"] = int(len(X_train))
    metrics["val_size"] = int(len(X_val))
    metrics["test_size"] = int(len(X_test))

    model.booster_.save_model(str(save_dir / "csao_lgb.txt"))
    with open(save_dir / "feature_columns.yaml", "w") as f:
        yaml.dump({"feature_columns": feature_cols}, f)
    with open(save_dir / "metrics.yaml", "w") as f:
        yaml.dump(metrics, f)
    return model, feature_cols, metrics


if __name__ == "__main__":
    model, feature_cols, metrics = train()
    print("Training complete. Test metrics:", metrics)
