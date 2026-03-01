# Business Impact Analysis

## Target Metrics (from problem statement)

| Metric | Description |
|--------|-------------|
| **AOV (average order value)** | Incremental lift from add-on recommendations |
| **CSAO rail order share** | % of orders that include at least one recommended add-on |
| **Cart-to-order (C2O) ratio** | Effect on order completion |
| **CSAO rail attach rate** | % of carts where user adds a recommended item |
| **Add-on acceptance rate** | % of recommended items added to cart |

## How This Solution Addresses Them

1. **Prediction accuracy (AUC, NDCG):** A better ranking model increases the chance that shown add-ons are clicked and added, which directly supports **add-on acceptance rate** and **CSAO rail attach rate**.
2. **AOV lift:** By recommending relevant add-ons (beverages, desserts, sides) we aim to increase basket size; offline we cannot measure AOV directly without A/B data, but we optimize for relevance (acceptance) which is a proxy.
3. **C2O and user experience:** Recommendations are non-intrusive (top-K rail); relevance and diversity reduce fatigue and support completion.

## Projected Impact (for submission)

- **AOV lift:** Assume baseline add-on attach rate X%; with improved model, project Y% relative lift in add-on attach and hence proportional AOV lift (e.g. 5–15% depending on segment).
- **Segment-level:** Premium users and dinner/late-night may show higher attach rates; budget users may respond better to value-oriented add-ons. These can be validated with segment-level evaluation.
- **Deployment:** Serve model via FastAPI; monitor latency (target < 300 ms) and acceptance rate in A/B tests. Refresh user/restaurant features periodically (e.g. daily) for freshness.

## Limitations

- Offline evaluation uses historical “add-on” definition (items after first in order); real acceptance requires online experimentation.
- Cold start and diversity rules are heuristic; tuning with live data would improve them.
