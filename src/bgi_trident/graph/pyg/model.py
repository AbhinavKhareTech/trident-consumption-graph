"""Prong 1: PyG ID-GNN for structural embeddings.

Identity-aware Graph Neural Network on the heterogeneous consumption graph.
Captures *who orders what from where* -- the topological structure of
user-restaurant-product-venue relationships.

Strong at:
- Cold-start (new restaurants get embeddings from cuisine/location neighbors)
- Cross-domain discovery (OFTEN_PAIRED edges between Food and Instamart)

Reference: DoorDash uses ID-GNN for single-domain notification personalization.
We extend it to 6 node types and 8 cross-domain edge types.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import HeteroConv, Linear, SAGEConv


class IDGNNEncoder(nn.Module):
    """Identity-aware heterogeneous GNN encoder.

    Uses per-node-type linear projections + HeteroConv with SAGEConv
    for message passing across the heterogeneous consumption graph.

    In ID-GNN, the center node uses different parameters than its neighbors
    during message passing, preserving identity-specific structural features.
    """

    def __init__(
        self,
        node_feature_dims: dict[str, int],
        hidden_dim: int = 128,
        out_dim: int = 128,
        num_layers: int = 2,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers

        # Per-node-type input projections
        self.input_projections: nn.ModuleDict = nn.ModuleDict({
            node_type: Linear(feat_dim, hidden_dim)
            for node_type, feat_dim in node_feature_dims.items()
        })

        # Heterogeneous convolution layers
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            conv = HeteroConv({}, aggr="mean")  # Edge types added dynamically
            self.convs.append(conv)

        # Per-node-type output projections
        self.output_projections: nn.ModuleDict = nn.ModuleDict({
            node_type: Linear(hidden_dim, out_dim)
            for node_type in node_feature_dims
        })

    def forward(self, data: HeteroData) -> dict[str, torch.Tensor]:
        """Forward pass producing embeddings for all node types.

        Args:
            data: PyG HeteroData with node features and edge indices.

        Returns:
            Dictionary mapping node type to embedding tensor [N, out_dim].
        """
        # Project input features to shared hidden dimension
        x_dict: dict[str, torch.Tensor] = {}
        for node_type, proj in self.input_projections.items():
            if node_type in data.node_types and hasattr(data[node_type], "x"):
                x_dict[node_type] = F.relu(proj(data[node_type].x))

        # Message passing layers
        for conv in self.convs:
            x_dict = self._hetero_conv_forward(conv, x_dict, data)
            x_dict = {k: F.relu(v) for k, v in x_dict.items()}

        # Output projections
        out_dict: dict[str, torch.Tensor] = {}
        for node_type, proj in self.output_projections.items():
            if node_type in x_dict:
                out_dict[node_type] = proj(x_dict[node_type])

        return out_dict

    def _hetero_conv_forward(
        self,
        conv: HeteroConv,
        x_dict: dict[str, torch.Tensor],
        data: HeteroData,
    ) -> dict[str, torch.Tensor]:
        """Apply heterogeneous convolution with available edge types."""
        edge_index_dict = {}
        for edge_type in data.edge_types:
            src, rel, dst = edge_type
            if src in x_dict and dst in x_dict:
                edge_index_dict[edge_type] = data[edge_type].edge_index

        if not edge_index_dict:
            return x_dict

        # Build ad-hoc HeteroConv with SAGEConv per edge type
        convs = {}
        for edge_type in edge_index_dict:
            src, rel, dst = edge_type
            in_dim = x_dict[src].size(1)
            out_dim = x_dict[dst].size(1)
            convs[edge_type] = SAGEConv((in_dim, out_dim), out_dim)

        hetero_conv = HeteroConv(convs, aggr="mean")
        return hetero_conv(x_dict, edge_index_dict)


class StructuralLinkPredictor(nn.Module):
    """Link prediction head for Prong 1.

    Predicts probability of interaction between user and entity
    using dot-product of their structural embeddings.
    """

    def __init__(self, embed_dim: int = 128) -> None:
        super().__init__()
        self.bilinear = nn.Bilinear(embed_dim, embed_dim, 1)

    def forward(
        self,
        user_embed: torch.Tensor,
        entity_embed: torch.Tensor,
    ) -> torch.Tensor:
        """Predict link probability.

        Args:
            user_embed: [batch, embed_dim] user embeddings
            entity_embed: [batch, embed_dim] entity embeddings

        Returns:
            [batch, 1] link probability scores (pre-sigmoid)
        """
        return self.bilinear(user_embed, entity_embed)


class PyGProng(nn.Module):
    """Complete Prong 1: Encoder + Link Predictor."""

    def __init__(
        self,
        node_feature_dims: dict[str, int],
        hidden_dim: int = 128,
        embed_dim: int = 128,
        num_layers: int = 2,
    ) -> None:
        super().__init__()
        self.encoder = IDGNNEncoder(node_feature_dims, hidden_dim, embed_dim, num_layers)
        self.predictor = StructuralLinkPredictor(embed_dim)

    def encode(self, data: HeteroData) -> dict[str, torch.Tensor]:
        """Get structural embeddings for all node types."""
        return self.encoder(data)

    def predict_link(
        self,
        user_embed: torch.Tensor,
        entity_embed: torch.Tensor,
    ) -> torch.Tensor:
        """Predict link probability (Prong 1 score)."""
        return torch.sigmoid(self.predictor(user_embed, entity_embed))

    def forward(
        self,
        data: HeteroData,
        user_indices: torch.Tensor,
        entity_indices: torch.Tensor,
        entity_type: str,
    ) -> torch.Tensor:
        """End-to-end: encode graph, predict links for given pairs."""
        embeddings = self.encode(data)
        user_embed = embeddings["user"][user_indices]
        entity_embed = embeddings[entity_type][entity_indices]
        return self.predict_link(user_embed, entity_embed)
