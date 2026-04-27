"""Graph schema definitions for the BGI Trident consumption graph.

Defines node types, edge types, and feature vectors for the heterogeneous
behavioral graph spanning Food, Instamart, and Dineout domains.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class NodeType(str, Enum):
    """Six node types in the consumption graph."""

    USER = "user"
    RESTAURANT = "restaurant"
    PRODUCT = "product"
    VENUE = "venue"
    TIMESLOT = "timeslot"
    LOCATION = "location"


class EdgeType(str, Enum):
    """Eight cross-domain edge types.

    OFTEN_PAIRED and FOLLOWED_BY_DINING are the cross-domain edges
    that DoorDash's single-domain graph cannot represent.
    """

    # User -> Entity edges (direct interactions)
    ORDERED_FROM = "ordered_from"  # User -> Restaurant
    PURCHASED = "purchased"  # User -> Product
    BOOKED_AT = "booked_at"  # User -> Venue
    PREFERS_AT_TIME = "prefers_at_time"  # User -> TimeSlot
    LOCATED_NEAR = "located_near"  # User -> Location

    # Entity -> Entity edges (cross-domain signals)
    SERVES_CUISINE = "serves_cuisine"  # Restaurant -> Restaurant (shared cuisine)
    OFTEN_PAIRED = "often_paired"  # Restaurant -> Product (co-occurrence)
    FOLLOWED_BY_DINING = "followed_by_dining"  # Restaurant -> Venue (48h correlation)


# Canonical edge tuples for PyG HeteroData and DGL DGLHeteroGraph
EDGE_REGISTRY: dict[EdgeType, tuple[NodeType, str, NodeType]] = {
    EdgeType.ORDERED_FROM: (NodeType.USER, "ordered_from", NodeType.RESTAURANT),
    EdgeType.PURCHASED: (NodeType.USER, "purchased", NodeType.PRODUCT),
    EdgeType.BOOKED_AT: (NodeType.USER, "booked_at", NodeType.VENUE),
    EdgeType.PREFERS_AT_TIME: (NodeType.USER, "prefers_at_time", NodeType.TIMESLOT),
    EdgeType.LOCATED_NEAR: (NodeType.USER, "located_near", NodeType.LOCATION),
    EdgeType.SERVES_CUISINE: (NodeType.RESTAURANT, "serves_cuisine", NodeType.RESTAURANT),
    EdgeType.OFTEN_PAIRED: (NodeType.RESTAURANT, "often_paired", NodeType.PRODUCT),
    EdgeType.FOLLOWED_BY_DINING: (NodeType.RESTAURANT, "followed_by_dining", NodeType.VENUE),
}


@dataclass
class NodeFeatureSpec:
    """Feature vector specification for a node type."""

    node_type: NodeType
    feature_names: list[str]
    embedding_dim: int
    description: str = ""


@dataclass
class EdgeWeightSpec:
    """Weight signals carried on each edge type."""

    edge_type: EdgeType
    weight_signals: list[str]
    has_temporal_decay: bool = False
    description: str = ""


# Node feature specifications
NODE_FEATURES: dict[NodeType, NodeFeatureSpec] = {
    NodeType.USER: NodeFeatureSpec(
        node_type=NodeType.USER,
        feature_names=[
            "cuisine_pref_vector",  # Multi-hot (20 cuisines)
            "price_sensitivity",  # Float [0, 1]
            "order_frequency",  # Orders per week
            "avg_order_value",  # INR
            "preferred_hour_vector",  # 24-dim histogram
            "preferred_day_vector",  # 7-dim histogram
        ],
        embedding_dim=128,
        description="User consumption profile, partially learned from interactions",
    ),
    NodeType.RESTAURANT: NodeFeatureSpec(
        node_type=NodeType.RESTAURANT,
        feature_names=[
            "cuisine_vector",  # Multi-hot (20 cuisines)
            "rating",  # Float [1, 5]
            "price_range",  # Categorical [1, 4]
            "avg_delivery_time_min",  # Float
            "geohash_embedding",  # 8-dim location embedding
            "is_promoted",  # Boolean
        ],
        embedding_dim=128,
        description="Restaurant attributes from Food MCP + public data",
    ),
    NodeType.PRODUCT: NodeFeatureSpec(
        node_type=NodeType.PRODUCT,
        feature_names=[
            "category_vector",  # Multi-hot (50 categories)
            "brand_embedding",  # 16-dim brand embedding
            "unit_price",  # INR
            "unit_size",  # Normalized quantity
            "avg_reorder_interval_days",  # Float
        ],
        embedding_dim=64,
        description="Instamart product attributes",
    ),
    NodeType.VENUE: NodeFeatureSpec(
        node_type=NodeType.VENUE,
        feature_names=[
            "cuisine_vector",  # Multi-hot (20 cuisines)
            "ambiance_vector",  # Multi-hot (casual, fine_dining, cafe, etc.)
            "price_range",  # Categorical [1, 4]
            "avg_rating",  # Float [1, 5]
            "capacity",  # Integer
            "geohash_embedding",  # 8-dim location embedding
        ],
        embedding_dim=64,
        description="Dineout venue attributes",
    ),
    NodeType.TIMESLOT: NodeFeatureSpec(
        node_type=NodeType.TIMESLOT,
        feature_names=[
            "hour_of_day",  # Integer [0, 23]
            "day_of_week",  # Integer [0, 6]
            "is_weekend",  # Boolean
            "is_holiday",  # Boolean
            "is_meal_hour",  # Boolean (lunch 11-14, dinner 19-22)
        ],
        embedding_dim=16,
        description="Temporal context nodes (168 total = 24h x 7d)",
    ),
    NodeType.LOCATION: NodeFeatureSpec(
        node_type=NodeType.LOCATION,
        feature_names=[
            "geohash_6",  # 6-char geohash
            "area_name",  # String (Koramangala, Indiranagar, etc.)
            "lat",  # Float
            "lng",  # Float
            "restaurant_density",  # Count within 2km
        ],
        embedding_dim=16,
        description="Geolocation nodes for delivery areas",
    ),
}

# Edge weight specifications
EDGE_WEIGHTS: dict[EdgeType, EdgeWeightSpec] = {
    EdgeType.ORDERED_FROM: EdgeWeightSpec(
        edge_type=EdgeType.ORDERED_FROM,
        weight_signals=["order_count", "recency_days", "avg_spend_inr", "avg_rating_given"],
        has_temporal_decay=True,
        description="User-Restaurant order history with recency decay",
    ),
    EdgeType.PURCHASED: EdgeWeightSpec(
        edge_type=EdgeType.PURCHASED,
        weight_signals=["purchase_count", "avg_interval_days", "last_purchase_days_ago"],
        has_temporal_decay=True,
        description="User-Product purchase history for reorder prediction",
    ),
    EdgeType.BOOKED_AT: EdgeWeightSpec(
        edge_type=EdgeType.BOOKED_AT,
        weight_signals=["booking_count", "avg_party_size", "avg_rating_given"],
        has_temporal_decay=True,
        description="User-Venue dining history",
    ),
    EdgeType.PREFERS_AT_TIME: EdgeWeightSpec(
        edge_type=EdgeType.PREFERS_AT_TIME,
        weight_signals=["order_count_at_slot", "conversion_rate_at_slot"],
        has_temporal_decay=False,
        description="User temporal preference pattern",
    ),
    EdgeType.LOCATED_NEAR: EdgeWeightSpec(
        edge_type=EdgeType.LOCATED_NEAR,
        weight_signals=["delivery_count_to_location", "is_primary_address"],
        has_temporal_decay=False,
        description="User delivery address frequency",
    ),
    EdgeType.SERVES_CUISINE: EdgeWeightSpec(
        edge_type=EdgeType.SERVES_CUISINE,
        weight_signals=["cuisine_overlap_score"],
        has_temporal_decay=False,
        description="Implicit restaurant similarity via shared cuisine",
    ),
    EdgeType.OFTEN_PAIRED: EdgeWeightSpec(
        edge_type=EdgeType.OFTEN_PAIRED,
        weight_signals=["co_occurrence_count", "co_occurrence_within_hours"],
        has_temporal_decay=True,
        description="Cross-domain: Food order paired with Instamart purchase same day",
    ),
    EdgeType.FOLLOWED_BY_DINING: EdgeWeightSpec(
        edge_type=EdgeType.FOLLOWED_BY_DINING,
        weight_signals=["sequence_count", "avg_gap_hours"],
        has_temporal_decay=True,
        description="Cross-domain: Food order followed by Dineout booking within 48h",
    ),
}


def get_pyg_edge_types() -> list[tuple[str, str, str]]:
    """Return edge types as PyG-compatible string tuples."""
    return [(src.value, rel, dst.value) for src, rel, dst in EDGE_REGISTRY.values()]


def get_dgl_edge_types() -> list[tuple[str, str, str]]:
    """Return edge types as DGL-compatible canonical edge type tuples."""
    return [(src.value, rel, dst.value) for src, rel, dst in EDGE_REGISTRY.values()]


def get_node_feature_dim(node_type: NodeType) -> int:
    """Return total raw feature dimension for a node type."""
    spec = NODE_FEATURES[node_type]
    return len(spec.feature_names)
