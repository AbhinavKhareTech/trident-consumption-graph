"""PyG embedding extraction utilities for Prong 1."""
from __future__ import annotations

import torch

from bgi_trident.graph.pyg.model import PyGProng


def extract_user_embeddings(model: PyGProng, data: torch_geometric.data.HeteroData) -> torch.Tensor:
    """Extract user structural embeddings from trained PyG model."""
    embeddings = model.encode(data)
    return embeddings.get("user", torch.zeros(0))


def extract_entity_embeddings(model: PyGProng, data: torch_geometric.data.HeteroData, entity_type: str) -> torch.Tensor:
    """Extract entity embeddings for a given node type."""
    embeddings = model.encode(data)
    return embeddings.get(entity_type, torch.zeros(0))
