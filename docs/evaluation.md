# Evaluation

## Metrics (aligned with Zomathon problem statement)

| Metric | Description |
|--------|-------------|
| **AUC** | Overall model discrimination ability |
| **Precision@K** | Accuracy of top-K recommendations |
| **Recall@K** | Coverage of relevant items in top-K |
| **NDCG@K** | Ranking quality |

## Offline Evaluation

1. **Temporal split:** Train 70% / Val 15% / Test 15% by order date.
2. **Training:** `python -m src.model.train` — builds features, trains LightGBM, writes test metrics to `models/metrics.yaml`.
3. **Full evaluation + baseline:** `python -m src.evaluation.run_evaluation` — compares model vs popularity baseline on test set; writes `models/evaluation_report.yaml` with model metrics, baseline metrics, and lift.

## Baseline

- **Popularity:** Rank candidates by `item_order_count` (how often the item was ordered). This is the primary baseline for comparison.
- **Cold start:** Popularity + meal-time category bias + diversity (max 2–3 per category in top-K).

## Segment-Level Analysis

For submission, you can extend `run_evaluation.py` to group test set by segment (e.g. meal time, restaurant type, user segment) and report metrics per segment for error analysis.
