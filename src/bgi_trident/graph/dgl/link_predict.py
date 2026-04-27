"""Link prediction utilities for Prong 2."""
from __future__ import annotations
import torch
from bgi_trident.graph.dgl.model import DGLProng


def predict_temporal_topk(model: DGLProng, g: "dgl.DGLHeteroGraph",
                          user_idx: int, entity_type: str, k: int = 10) -> list[tuple[int, float]]:
    """Predict top-k entities using temporal-behavioral scores."""
    embeddings = model.encode(g)
    if "user" not in embeddings or entity_type not in embeddings:
        return []
    user_embed = embeddings["user"][user_idx].unsqueeze(0)
    entity_embeds = embeddings[entity_type]
    scores = model.predict_link(
        user_embed.expand(entity_embeds.size(0), -1), entity_embeds
    ).squeeze()
    topk_vals, topk_idx = torch.topk(scores, min(k, len(scores)))
    return [(idx.item(), val.item()) for idx, val in zip(topk_idx, topk_vals)]
