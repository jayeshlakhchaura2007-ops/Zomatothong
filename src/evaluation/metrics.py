"""Offline metrics: AUC, Precision@K, Recall@K, NDCG."""
from typing import Any

import numpy as np
from sklearn.metrics import roc_auc_score


def _precision_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int) -> float:
    """Micro-averaged Precision@K: top-k by score, then precision."""
    if y_true.size == 0 or k <= 0:
        return 0.0
    order = np.argsort(-y_score)
    top_k = order[:k]
    return float(np.sum(y_true[top_k])) / min(k, len(top_k))


def _recall_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int) -> float:
    """Recall@K: of all positives, how many in top-k."""
    n_pos = np.sum(y_true)
    if n_pos == 0:
        return 0.0
    order = np.argsort(-y_score)
    top_k = order[:k]
    return float(np.sum(y_true[top_k])) / n_pos


def _ndcg_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int) -> float:
    """NDCG@K (binary relevance)."""
    if y_true.size == 0 or k <= 0:
        return 0.0
    order = np.argsort(-y_score)[:k]
    rel = y_true[order]
    dcg = np.sum(rel / np.log2(np.arange(2, len(rel) + 2)))
    ideal = np.sort(y_true)[::-1][:k]
    idcg = np.sum(ideal / np.log2(np.arange(2, len(ideal) + 2)))
    if idcg == 0:
        return 0.0
    return float(dcg / idcg)


def compute_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    y_pred_binary: np.ndarray | None = None,
    k: int = 10,
) -> dict[str, Any]:
    """Compute AUC, Precision@K, Recall@K, NDCG@K."""
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    if y_pred_binary is None:
        y_pred_binary = (y_score >= 0.5).astype(int)
    else:
        y_pred_binary = np.asarray(y_pred_binary)

    metrics = {}
    if np.unique(y_true).size > 1:
        metrics["auc"] = float(roc_auc_score(y_true, y_score))
    else:
        metrics["auc"] = 0.0
    metrics[f"precision_at_{k}"] = _precision_at_k(y_true, y_score, k)
    metrics[f"recall_at_{k}"] = _recall_at_k(y_true, y_score, k)
    metrics[f"ndcg_at_{k}"] = _ndcg_at_k(y_true, y_score, k)
    return metrics
