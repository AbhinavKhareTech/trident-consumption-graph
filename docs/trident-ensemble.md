# Trident Ensemble: Why Three Prongs

## The Problem with Single-Model Approaches

DoorDash uses a single ID-GNN. It works well for structural patterns but misses two critical signal classes:

1. **Temporal drift**: A user whose ordering frequency is declining looks identical to an active user in a static graph. DGL with recency decay catches this.

2. **Tabular thresholds**: "User never orders after 10 PM" is a sharp boundary. GNN message passing smooths over these. XGBoost handles them natively.

## The Biryani-Every-Thursday Example

User 42 ordered biryani from Meghana's every Thursday for 6 weeks, then stopped 3 weeks ago.

| Prong | Score | Reasoning |
|---|---|---|
| PyG (Structural) | 0.91 | 14 edges to Meghana's, strong neighborhood connectivity |
| DGL (Temporal) | 0.43 | Edge weights decayed significantly over 3 weeks of inactivity |
| XGBoost (Tabular) | 0.35 | recency_days=21, frequency_per_week dropping, flags threshold |
| **Ensemble** | **0.38** | **Correctly resolves: user is drifting, not active** |

A single GNN would score 0.91 and waste a proactive notification. The ensemble correctly identifies the drift.

## Ablation Results

Run `notebooks/05_ensemble_ablation.ipynb` for full reproducible results.

Expected pattern:
- Ensemble AUC > best single prong AUC
- PyG + DGL > either alone (graph signals complement)
- Adding XGBoost provides incremental lift on threshold-heavy patterns
