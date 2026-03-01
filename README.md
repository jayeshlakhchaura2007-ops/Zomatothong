# CSAO (Cart Super Add-On) Rail Recommendation System

Intelligent add-on recommendation for food delivery: suggests relevant items based on cart composition, context, and user behavior. Built for the Zomathon problem statement with **< 300 ms** latency target.

## Quick Start

### 1. Install

```bash
cd Zomatothon
pip install -r requirements.txt
```

### 2. Generate sample data

```bash
python -m src.data.generate_sample_data
```

This writes `data/users.csv`, `data/restaurants.csv`, `data/menu_items.csv`, `data/orders.csv`, `data/order_items.csv`.

### 3. Train model

```bash
python -m src.model.train
```

Saves model to `models/csao_lgb.txt`, feature list to `models/feature_columns.yaml`, and test metrics to `models/metrics.yaml`.

### 4. Run evaluation (optional)

```bash
python -m src.evaluation.run_evaluation
```

Writes `models/evaluation_report.yaml` with model vs baseline metrics.

### 5. Start API

```bash
uvicorn src.api.service:app --reload --host 0.0.0.0 --port 8000
```

- **POST /recommend** — body: `{"user_id": "u_0", "restaurant_id": "r_0", "cart_item_ids": ["i_1"], "top_k": 10}`. Returns `{ "recommendations": [...], "latency_ms": ... }`.
- **GET /health** — liveness.
- **GET /ready** — model and data loaded.

## Project layout

```
Zomatothon/
├── configs/          # features.yaml, model.yaml
├── data/             # Raw CSVs + schema.md; generate_sample_data.py
├── models/           # Saved model, feature_columns.yaml, metrics, evaluation_report
├── src/
│   ├── api/          # FastAPI service
│   ├── data/         # Sample data generator
│   ├── evaluation/   # Metrics, baseline, run_evaluation
│   ├── features/     # User, restaurant, cart, context, candidates, cold_start
│   └── model/        # train, predict, config
├── docs/             # architecture, evaluation, business_impact
└── README.md
```

## Configuration

- **configs/features.yaml:** Feature flags, candidate max count, latency target.
- **configs/model.yaml:** LightGBM hyperparameters, temporal split, inference top_k.

## Documentation

- [docs/architecture.md](docs/architecture.md) — system design, data flow, scalability.
- [docs/evaluation.md](docs/evaluation.md) — metrics, offline evaluation, baseline.
- [docs/business_impact.md](docs/business_impact.md) — AOV, attach rate, deployment notes.

## Submission checklist (Zomathon)

1. **Model:** Training code and saved model; feature pipeline in `src/features/`.
2. **Evaluation:** Test metrics and baseline comparison; run `run_evaluation.py`.
3. **Documentation:** Architecture, evaluation, business impact in `docs/`.
4. **API:** Production-style endpoint with latency and cold-start fallback.
