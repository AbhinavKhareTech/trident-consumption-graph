"""Temporal utilities for Prong 2: recency decay and time features."""
from __future__ import annotations

import torch

from bgi_trident.graph.dgl.model import compute_recency_decay


def apply_temporal_features(g: dgl.DGLHeteroGraph, interaction_timestamps: dict[str, list[float]],
                            reference_time: float, half_life_days: float = 14.0) -> dgl.DGLHeteroGraph:
    """Apply recency decay weights to all edge types with temporal data."""
    for etype in g.canonical_etypes:
        etype_key = f"{etype[0]}_{etype[1]}_{etype[2]}"
        if etype_key in interaction_timestamps:
            days_since = torch.tensor(
                [(reference_time - t) / 86400 for t in interaction_timestamps[etype_key]],
                dtype=torch.float32
            )
            decay = compute_recency_decay(days_since, half_life_days)
            g.edges[etype].data["temporal_weight"] = decay
    return g
