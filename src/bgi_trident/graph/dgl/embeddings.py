"""DGL embedding extraction utilities for Prong 2."""
from __future__ import annotations

import torch

from bgi_trident.graph.dgl.model import DGLProng


def extract_temporal_embeddings(model: DGLProng, g: dgl.DGLHeteroGraph) -> dict[str, torch.Tensor]:
    """Extract temporal-behavioral embeddings from trained DGL model."""
    return model.encode(g)
