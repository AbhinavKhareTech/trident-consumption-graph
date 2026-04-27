"""Tests for DGL Prong 2 model."""
from bgi_trident.graph.dgl.model import compute_recency_decay
import torch


def test_recency_decay():
    days = torch.tensor([0.0, 7.0, 14.0, 28.0])
    decay = compute_recency_decay(days, half_life_days=14.0)
    assert abs(decay[0].item() - 1.0) < 0.01  # Today = full weight
    assert abs(decay[2].item() - 0.5) < 0.01  # Half-life = half weight
    assert decay[3].item() < decay[2].item()   # Further decay
