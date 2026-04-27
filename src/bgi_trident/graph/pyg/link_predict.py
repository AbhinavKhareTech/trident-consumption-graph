"""Link prediction utilities for Prong 1."""
from __future__ import annotations

import torch

from bgi_trident.graph.pyg.model import PyGProng


def predict_topk(model: PyGProng, data: torch_geometric.data.HeteroData,
                 user_idx: int, entity_type: str, k: int = 10) -> list[tuple[int, float]]:
    """Predict top-k entities for a user by structural link probability."""
    embeddings = model.encode(data)
    user_embed = embeddings["user"][user_idx].unsqueeze(0)
    entity_embeds = embeddings[entity_type]
    scores = model.predict_link(
        user_embed.expand(entity_embeds.size(0), -1), entity_embeds
    ).squeeze()
    topk_vals, topk_idx = torch.topk(scores, min(k, len(scores)))
    return [(idx.item(), val.item()) for idx, val in zip(topk_idx, topk_vals, strict=False)]
