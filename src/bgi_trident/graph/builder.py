"""Heterogeneous graph construction from interaction data.

Builds a single consumption graph with 6 node types and 8 edge types,
then exports to both PyG HeteroData and DGL DGLHeteroGraph formats
for Prong 1 and Prong 2 respectively.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from bgi_trident.graph.schema import (
    EDGE_REGISTRY,
    EDGE_WEIGHTS,
    NODE_FEATURES,
    EdgeType,
    NodeType,
)

logger = logging.getLogger(__name__)


@dataclass
class InteractionData:
    """Raw interaction data loaded from CSVs."""

    food_orders: pd.DataFrame  # user_id, restaurant_id, timestamp, amount, rating
    instamart_orders: pd.DataFrame  # user_id, product_id, timestamp, quantity, amount
    dineout_bookings: pd.DataFrame  # user_id, venue_id, timestamp, party_size, rating
    users: pd.DataFrame  # user_id, location, preferences
    restaurants: pd.DataFrame  # restaurant_id, cuisine, rating, price_range, lat, lng
    products: pd.DataFrame  # product_id, category, brand, price
    venues: pd.DataFrame  # venue_id, cuisine, ambiance, price_range, capacity, lat, lng


class ConsumptionGraphBuilder:
    """Builds the heterogeneous consumption graph from raw interaction data.

    Constructs node feature tensors and edge index tensors with weight signals
    for all 6 node types and 8 edge types defined in the schema.
    """

    def __init__(self, data: InteractionData) -> None:
        self.data = data
        self._node_maps: dict[NodeType, dict[int, int]] = {}
        self._node_features: dict[str, torch.Tensor] = {}
        self._edge_indices: dict[tuple[str, str, str], torch.Tensor] = {}
        self._edge_weights: dict[tuple[str, str, str], torch.Tensor] = {}

    def build(self) -> ConsumptionGraphBuilder:
        """Full build pipeline: nodes -> edges -> cross-domain edges."""
        logger.info("Building consumption graph...")
        self._build_node_maps()
        self._build_node_features()
        self._build_direct_edges()
        self._build_cross_domain_edges()
        self._build_temporal_edges()
        self._build_location_edges()
        logger.info(
            "Graph built: %d node types, %d edge types",
            len(self._node_features),
            len(self._edge_indices),
        )
        return self

    def _build_node_maps(self) -> None:
        """Create contiguous ID mappings for each node type."""
        self._node_maps[NodeType.USER] = {
            uid: idx for idx, uid in enumerate(self.data.users["user_id"].unique())
        }
        self._node_maps[NodeType.RESTAURANT] = {
            rid: idx for idx, rid in enumerate(self.data.restaurants["restaurant_id"].unique())
        }
        self._node_maps[NodeType.PRODUCT] = {
            pid: idx for idx, pid in enumerate(self.data.products["product_id"].unique())
        }
        self._node_maps[NodeType.VENUE] = {
            vid: idx for idx, vid in enumerate(self.data.venues["venue_id"].unique())
        }
        # 168 timeslots: 24 hours x 7 days
        self._node_maps[NodeType.TIMESLOT] = {i: i for i in range(168)}
        # Locations derived from user delivery addresses
        locations = self.data.users["location"].unique()
        self._node_maps[NodeType.LOCATION] = {
            loc: idx for idx, loc in enumerate(locations)
        }

    def _build_node_features(self) -> None:
        """Construct feature tensors for each node type."""
        # User features
        n_users = len(self._node_maps[NodeType.USER])
        user_feats = self._extract_user_features()
        self._node_features["user"] = user_feats

        # Restaurant features
        rest_feats = self._extract_restaurant_features()
        self._node_features["restaurant"] = rest_feats

        # Product features
        prod_feats = self._extract_product_features()
        self._node_features["product"] = prod_feats

        # Venue features
        venue_feats = self._extract_venue_features()
        self._node_features["venue"] = venue_feats

        # TimeSlot features (deterministic)
        ts_feats = self._build_timeslot_features()
        self._node_features["timeslot"] = ts_feats

        # Location features
        loc_feats = self._extract_location_features()
        self._node_features["location"] = loc_feats

    def _build_direct_edges(self) -> None:
        """Build ORDERED_FROM, PURCHASED, BOOKED_AT edges from interaction data."""
        # ORDERED_FROM: User -> Restaurant
        self._build_interaction_edge(
            interactions=self.data.food_orders,
            src_col="user_id",
            dst_col="restaurant_id",
            src_type=NodeType.USER,
            dst_type=NodeType.RESTAURANT,
            edge_type=EdgeType.ORDERED_FROM,
            weight_cols=["amount", "rating"],
        )

        # PURCHASED: User -> Product
        self._build_interaction_edge(
            interactions=self.data.instamart_orders,
            src_col="user_id",
            dst_col="product_id",
            src_type=NodeType.USER,
            dst_type=NodeType.PRODUCT,
            edge_type=EdgeType.PURCHASED,
            weight_cols=["quantity", "amount"],
        )

        # BOOKED_AT: User -> Venue
        self._build_interaction_edge(
            interactions=self.data.dineout_bookings,
            src_col="user_id",
            dst_col="venue_id",
            src_type=NodeType.USER,
            dst_type=NodeType.VENUE,
            edge_type=EdgeType.BOOKED_AT,
            weight_cols=["party_size", "rating"],
        )

    def _build_cross_domain_edges(self) -> None:
        """Build OFTEN_PAIRED and FOLLOWED_BY_DINING cross-domain edges.

        These are the edges DoorDash's single-domain graph cannot represent.
        """
        # OFTEN_PAIRED: Restaurant -> Product
        # Find co-occurrences where a food order and instamart purchase
        # happen within the same day for the same user
        food = self.data.food_orders.copy()
        food["date"] = pd.to_datetime(food["timestamp"]).dt.date
        instamart = self.data.instamart_orders.copy()
        instamart["date"] = pd.to_datetime(instamart["timestamp"]).dt.date

        paired = food.merge(instamart, on=["user_id", "date"], suffixes=("_food", "_insta"))
        if not paired.empty:
            pair_counts = (
                paired.groupby(["restaurant_id", "product_id"])
                .size()
                .reset_index(name="co_occurrence_count")
            )
            pair_counts = pair_counts[pair_counts["co_occurrence_count"] >= 2]

            src_ids = [
                self._node_maps[NodeType.RESTAURANT].get(r, -1)
                for r in pair_counts["restaurant_id"]
            ]
            dst_ids = [
                self._node_maps[NodeType.PRODUCT].get(p, -1)
                for p in pair_counts["product_id"]
            ]
            valid = [(s, d) for s, d in zip(src_ids, dst_ids) if s >= 0 and d >= 0]
            if valid:
                edge_key = ("restaurant", "often_paired", "product")
                src_t, dst_t = zip(*valid)
                self._edge_indices[edge_key] = torch.tensor([src_t, dst_t], dtype=torch.long)
                weights = pair_counts["co_occurrence_count"].values[: len(valid)]
                self._edge_weights[edge_key] = torch.tensor(weights, dtype=torch.float32)

        # FOLLOWED_BY_DINING: Restaurant -> Venue
        # Food order followed by Dineout booking within 48 hours
        food_ts = self.data.food_orders[["user_id", "restaurant_id", "timestamp"]].copy()
        food_ts["ts"] = pd.to_datetime(food_ts["timestamp"])
        dine_ts = self.data.dineout_bookings[["user_id", "venue_id", "timestamp"]].copy()
        dine_ts["ts"] = pd.to_datetime(dine_ts["timestamp"])

        sequences = food_ts.merge(dine_ts, on="user_id", suffixes=("_food", "_dine"))
        sequences["gap_hours"] = (
            (sequences["ts_dine"] - sequences["ts_food"]).dt.total_seconds() / 3600
        )
        sequences = sequences[(sequences["gap_hours"] > 0) & (sequences["gap_hours"] <= 48)]

        if not sequences.empty:
            seq_counts = (
                sequences.groupby(["restaurant_id", "venue_id"])
                .size()
                .reset_index(name="sequence_count")
            )
            seq_counts = seq_counts[seq_counts["sequence_count"] >= 2]

            src_ids = [
                self._node_maps[NodeType.RESTAURANT].get(r, -1)
                for r in seq_counts["restaurant_id"]
            ]
            dst_ids = [
                self._node_maps[NodeType.VENUE].get(v, -1) for v in seq_counts["venue_id"]
            ]
            valid = [(s, d) for s, d in zip(src_ids, dst_ids) if s >= 0 and d >= 0]
            if valid:
                edge_key = ("restaurant", "followed_by_dining", "venue")
                src_t, dst_t = zip(*valid)
                self._edge_indices[edge_key] = torch.tensor([src_t, dst_t], dtype=torch.long)

    def _build_temporal_edges(self) -> None:
        """Build PREFERS_AT_TIME edges: User -> TimeSlot."""
        food = self.data.food_orders.copy()
        food["hour"] = pd.to_datetime(food["timestamp"]).dt.hour
        food["dow"] = pd.to_datetime(food["timestamp"]).dt.dayofweek
        food["timeslot_id"] = food["hour"] + food["dow"] * 24

        slot_counts = food.groupby(["user_id", "timeslot_id"]).size().reset_index(name="count")
        slot_counts = slot_counts[slot_counts["count"] >= 2]

        src_ids = [self._node_maps[NodeType.USER].get(u, -1) for u in slot_counts["user_id"]]
        dst_ids = list(slot_counts["timeslot_id"])
        valid = [(s, d) for s, d in zip(src_ids, dst_ids) if s >= 0]

        if valid:
            edge_key = ("user", "prefers_at_time", "timeslot")
            src_t, dst_t = zip(*valid)
            self._edge_indices[edge_key] = torch.tensor([src_t, dst_t], dtype=torch.long)
            weights = slot_counts["count"].values[: len(valid)]
            self._edge_weights[edge_key] = torch.tensor(weights, dtype=torch.float32)

    def _build_location_edges(self) -> None:
        """Build LOCATED_NEAR edges: User -> Location."""
        src_ids = []
        dst_ids = []
        for _, row in self.data.users.iterrows():
            uid = self._node_maps[NodeType.USER].get(row["user_id"], -1)
            lid = self._node_maps[NodeType.LOCATION].get(row["location"], -1)
            if uid >= 0 and lid >= 0:
                src_ids.append(uid)
                dst_ids.append(lid)

        if src_ids:
            edge_key = ("user", "located_near", "location")
            self._edge_indices[edge_key] = torch.tensor([src_ids, dst_ids], dtype=torch.long)

    def _build_interaction_edge(
        self,
        interactions: pd.DataFrame,
        src_col: str,
        dst_col: str,
        src_type: NodeType,
        dst_type: NodeType,
        edge_type: EdgeType,
        weight_cols: list[str],
    ) -> None:
        """Generic edge builder from interaction DataFrame."""
        agg = interactions.groupby([src_col, dst_col]).agg(
            count=("timestamp", "size"),
            **{col: (col, "mean") for col in weight_cols if col in interactions.columns},
        ).reset_index()

        src_ids = [self._node_maps[src_type].get(v, -1) for v in agg[src_col]]
        dst_ids = [self._node_maps[dst_type].get(v, -1) for v in agg[dst_col]]
        valid_mask = [(s >= 0 and d >= 0) for s, d in zip(src_ids, dst_ids)]

        valid_src = [s for s, m in zip(src_ids, valid_mask) if m]
        valid_dst = [d for d, m in zip(dst_ids, valid_mask) if m]

        if valid_src:
            _, rel, _ = EDGE_REGISTRY[edge_type]
            edge_key = (src_type.value, rel, dst_type.value)
            self._edge_indices[edge_key] = torch.tensor(
                [valid_src, valid_dst], dtype=torch.long
            )
            counts = agg["count"].values[list(valid_mask)]
            self._edge_weights[edge_key] = torch.tensor(counts, dtype=torch.float32)

    def _extract_user_features(self) -> torch.Tensor:
        """Extract user feature tensor. Placeholder for full feature engineering."""
        n = len(self._node_maps[NodeType.USER])
        dim = len(NODE_FEATURES[NodeType.USER].feature_names)
        return torch.randn(n, dim)  # Replace with real feature extraction

    def _extract_restaurant_features(self) -> torch.Tensor:
        n = len(self._node_maps[NodeType.RESTAURANT])
        dim = len(NODE_FEATURES[NodeType.RESTAURANT].feature_names)
        return torch.randn(n, dim)

    def _extract_product_features(self) -> torch.Tensor:
        n = len(self._node_maps[NodeType.PRODUCT])
        dim = len(NODE_FEATURES[NodeType.PRODUCT].feature_names)
        return torch.randn(n, dim)

    def _extract_venue_features(self) -> torch.Tensor:
        n = len(self._node_maps[NodeType.VENUE])
        dim = len(NODE_FEATURES[NodeType.VENUE].feature_names)
        return torch.randn(n, dim)

    def _extract_location_features(self) -> torch.Tensor:
        n = len(self._node_maps[NodeType.LOCATION])
        dim = len(NODE_FEATURES[NodeType.LOCATION].feature_names)
        return torch.randn(n, dim)

    def _build_timeslot_features(self) -> torch.Tensor:
        """Deterministic timeslot features: 168 slots = 24h x 7d."""
        features = []
        for slot_id in range(168):
            hour = slot_id % 24
            dow = slot_id // 24
            is_weekend = 1.0 if dow >= 5 else 0.0
            is_holiday = 0.0  # Requires external calendar
            is_meal_hour = 1.0 if (11 <= hour <= 14 or 19 <= hour <= 22) else 0.0
            features.append([hour / 23.0, dow / 6.0, is_weekend, is_holiday, is_meal_hour])
        return torch.tensor(features, dtype=torch.float32)

    def to_pyg(self) -> "torch_geometric.data.HeteroData":
        """Export to PyG HeteroData for Prong 1 (ID-GNN)."""
        from torch_geometric.data import HeteroData

        data = HeteroData()

        for node_type_str, feat_tensor in self._node_features.items():
            data[node_type_str].x = feat_tensor
            data[node_type_str].num_nodes = feat_tensor.size(0)

        for edge_key, edge_index in self._edge_indices.items():
            data[edge_key].edge_index = edge_index
            if edge_key in self._edge_weights:
                data[edge_key].edge_weight = self._edge_weights[edge_key]

        return data

    def to_dgl(self) -> "dgl.DGLHeteroGraph":
        """Export to DGL DGLHeteroGraph for Prong 2 (R-GCN with temporal decay)."""
        import dgl

        graph_data: dict[tuple[str, str, str], tuple[torch.Tensor, torch.Tensor]] = {}
        for edge_key, edge_index in self._edge_indices.items():
            graph_data[edge_key] = (edge_index[0], edge_index[1])

        if not graph_data:
            raise ValueError("No edges found. Cannot build DGL graph.")

        g = dgl.heterograph(graph_data)

        for node_type_str, feat_tensor in self._node_features.items():
            if node_type_str in g.ntypes:
                g.nodes[node_type_str].data["feat"] = feat_tensor

        for edge_key, weight_tensor in self._edge_weights.items():
            if edge_key in g.canonical_etypes:
                g.edges[edge_key].data["weight"] = weight_tensor

        return g

    def get_xgboost_features(self) -> pd.DataFrame:
        """Export tabular features for Prong 3 (XGBoost).

        Returns a DataFrame with one row per (user, entity) pair
        and hand-engineered consumption features.
        """
        features = []

        for _, row in self.data.food_orders.groupby(["user_id", "restaurant_id"]).agg(
            order_count=("timestamp", "size"),
            avg_spend=("amount", "mean"),
            last_order=("timestamp", "max"),
            first_order=("timestamp", "min"),
        ).reset_index().iterrows():
            last_ts = pd.to_datetime(row["last_order"])
            first_ts = pd.to_datetime(row["first_order"])
            span_days = max((last_ts - first_ts).days, 1)
            features.append({
                "user_id": row["user_id"],
                "entity_id": row["restaurant_id"],
                "entity_type": "restaurant",
                "order_count": row["order_count"],
                "avg_spend": row["avg_spend"],
                "recency_days": (pd.Timestamp.now() - last_ts).days,
                "frequency_per_week": row["order_count"] / (span_days / 7),
                "span_days": span_days,
            })

        return pd.DataFrame(features)

    @property
    def node_counts(self) -> dict[str, int]:
        return {nt.value: len(nm) for nt, nm in self._node_maps.items()}

    @property
    def edge_counts(self) -> dict[str, int]:
        return {
            f"{s}-{r}->{d}": idx.size(1)
            for (s, r, d), idx in self._edge_indices.items()
        }
