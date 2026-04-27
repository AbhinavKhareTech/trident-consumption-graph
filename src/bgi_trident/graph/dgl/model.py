"""Prong 2: DGL R-GCN with temporal decay for behavioral embeddings.

Relational Graph Convolutional Network on a temporal variant of the
consumption graph where edge weights decay with recency.

Captures *when and how often* -- the behavioral dynamics that
PyG's static structural embeddings miss.

Key difference from Prong 1:
- A user who ordered biryani every Thursday for 6 weeks but stopped 3 weeks ago
  looks very different here (decaying edges) vs PyG (strong structural connection).
- DGL catches the drift. PyG does not.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import dgl
    import dgl.nn as dglnn
except ImportError:
    dgl = None  # type: ignore[assignment]
    dglnn = None  # type: ignore[assignment]


def compute_recency_decay(
    days_since: torch.Tensor,
    half_life_days: float = 14.0,
) -> torch.Tensor:
    """Exponential recency decay: weight = 2^(-days_since / half_life).

    A 14-day half-life means an edge from 2 weeks ago has half the weight
    of an edge from today. Captures behavioral drift that static graphs miss.
    """
    return torch.pow(2.0, -days_since / half_life_days)


class TemporalRGCNLayer(nn.Module):
    """Single R-GCN layer with recency-weighted message passing.

    Messages from edges with higher recency (more recent interactions)
    carry more weight during neighborhood aggregation.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_relations: int,
        num_bases: int = 4,
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        # Per-relation weight matrices with basis decomposition
        self.weight = nn.Parameter(torch.Tensor(num_relations, in_dim, out_dim))
        self.basis_weight = nn.Parameter(torch.Tensor(num_bases, in_dim, out_dim))
        self.basis_coeff = nn.Parameter(torch.Tensor(num_relations, num_bases))
        self.bias = nn.Parameter(torch.Tensor(out_dim))

        self._init_params()

    def _init_params(self) -> None:
        nn.init.xavier_uniform_(self.basis_weight)
        nn.init.xavier_uniform_(self.basis_coeff)
        nn.init.zeros_(self.bias)

    def forward(
        self,
        g: dgl.DGLHeteroGraph,
        x_dict: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Forward pass with recency-weighted aggregation."""
        if dgl is None:
            raise ImportError("DGL is required for Prong 2")

        out_dict: dict[str, torch.Tensor] = {}

        for ntype in g.ntypes:
            if ntype in x_dict:
                out_dict[ntype] = torch.zeros(
                    g.num_nodes(ntype), self.out_dim, device=x_dict[ntype].device
                )

        for rel_idx, canonical_etype in enumerate(g.canonical_etypes):
            src_type, etype, dst_type = canonical_etype
            if src_type not in x_dict or dst_type not in out_dict:
                continue

            sub_g = g[canonical_etype]
            if sub_g.num_edges() == 0:
                continue

            src_feat = x_dict[src_type]

            # Compute relation-specific weight via basis decomposition
            w = torch.einsum("b,bij->ij", self.basis_coeff[rel_idx % len(self.basis_coeff)], self.basis_weight)

            # Apply recency decay to edge weights if available
            msg = src_feat[sub_g.edges()[0]] @ w
            if "weight" in sub_g.edata:
                decay = sub_g.edata["weight"].unsqueeze(-1)
                msg = msg * decay

            # Aggregate (mean)
            dst_indices = sub_g.edges()[1]
            out_dict[dst_type].scatter_add_(0, dst_indices.unsqueeze(-1).expand_as(msg), msg)

        # Normalize and add bias
        for ntype in out_dict:
            degree = torch.clamp(
                torch.zeros(g.num_nodes(ntype), device=out_dict[ntype].device)
                .scatter_add_(0, torch.cat([
                    g[et].edges()[1] for et in g.canonical_etypes if et[2] == ntype
                ]) if any(et[2] == ntype for et in g.canonical_etypes) else torch.tensor([]),
                torch.ones(sum(
                    g[et].num_edges() for et in g.canonical_etypes if et[2] == ntype
                ), device=out_dict[ntype].device)),
                min=1.0,
            )
            out_dict[ntype] = out_dict[ntype] / degree.unsqueeze(-1) + self.bias

        return out_dict


class DGLProng(nn.Module):
    """Complete Prong 2: Temporal R-GCN encoder + link predictor.

    Two-layer R-GCN with recency-decayed edge weights, producing
    128-dimensional behavioral embeddings for each node type.
    """

    def __init__(
        self,
        node_feature_dims: dict[str, int],
        hidden_dim: int = 128,
        embed_dim: int = 128,
        num_relations: int = 8,
        num_layers: int = 2,
    ) -> None:
        super().__init__()

        # Input projections
        self.input_projections = nn.ModuleDict({
            ntype: nn.Linear(fdim, hidden_dim) for ntype, fdim in node_feature_dims.items()
        })

        # R-GCN layers
        self.layers = nn.ModuleList([
            TemporalRGCNLayer(hidden_dim, hidden_dim, num_relations)
            for _ in range(num_layers)
        ])

        # Output projections
        self.output_projections = nn.ModuleDict({
            ntype: nn.Linear(hidden_dim, embed_dim) for ntype in node_feature_dims
        })

        # Link prediction
        self.link_predictor = nn.Bilinear(embed_dim, embed_dim, 1)

    def encode(self, g: dgl.DGLHeteroGraph) -> dict[str, torch.Tensor]:
        """Get temporal-behavioral embeddings for all node types."""
        x_dict = {}
        for ntype in g.ntypes:
            if "feat" in g.nodes[ntype].data and ntype in self.input_projections:
                x_dict[ntype] = F.relu(self.input_projections[ntype](g.nodes[ntype].data["feat"]))

        for layer in self.layers:
            x_dict = layer(g, x_dict)
            x_dict = {k: F.relu(v) for k, v in x_dict.items()}

        out_dict = {}
        for ntype, proj in self.output_projections.items():
            if ntype in x_dict:
                out_dict[ntype] = proj(x_dict[ntype])

        return out_dict

    def predict_link(
        self,
        user_embed: torch.Tensor,
        entity_embed: torch.Tensor,
    ) -> torch.Tensor:
        """Predict link probability (Prong 2 score)."""
        return torch.sigmoid(self.link_predictor(user_embed, entity_embed))

    def apply_recency_decay(
        self,
        g: dgl.DGLHeteroGraph,
        recency_days: dict[str, torch.Tensor],
        half_life_days: float = 14.0,
    ) -> dgl.DGLHeteroGraph:
        """Apply recency decay to edge weights before forward pass.

        This is the key differentiator from Prong 1: edges that are
        older receive exponentially less weight during message passing.
        """
        for canonical_etype in g.canonical_etypes:
            etype_str = f"{canonical_etype[0]}_{canonical_etype[1]}_{canonical_etype[2]}"
            if etype_str in recency_days:
                decay = compute_recency_decay(recency_days[etype_str], half_life_days)
                if "weight" in g.edges[canonical_etype].data:
                    g.edges[canonical_etype].data["weight"] *= decay
                else:
                    g.edges[canonical_etype].data["weight"] = decay
        return g
