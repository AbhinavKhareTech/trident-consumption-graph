"""Tests for PyG Prong 1 model."""
import torch
from bgi_trident.graph.pyg.model import PyGProng


def test_pyg_prong_forward():
    model = PyGProng(node_feature_dims={"user": 6, "restaurant": 6}, hidden_dim=32, embed_dim=32, num_layers=1)
    from torch_geometric.data import HeteroData
    data = HeteroData()
    data["user"].x = torch.randn(10, 6)
    data["user"].num_nodes = 10
    data["restaurant"].x = torch.randn(5, 6)
    data["restaurant"].num_nodes = 5
    embeddings = model.encode(data)
    assert "user" in embeddings
    assert embeddings["user"].shape == (10, 32)
