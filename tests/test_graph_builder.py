"""Tests for heterogeneous graph construction."""
from bgi_trident.graph.builder import ConsumptionGraphBuilder, InteractionData


def test_graph_builder_constructs(sample_users, sample_restaurants, sample_products,
                                   sample_venues, sample_food_orders, sample_instamart_orders,
                                   sample_dineout_bookings):
    data = InteractionData(
        food_orders=sample_food_orders, instamart_orders=sample_instamart_orders,
        dineout_bookings=sample_dineout_bookings, users=sample_users,
        restaurants=sample_restaurants, products=sample_products, venues=sample_venues,
    )
    builder = ConsumptionGraphBuilder(data).build()
    assert builder.node_counts["user"] == 10
    assert builder.node_counts["restaurant"] == 5
    assert builder.node_counts["product"] == 5
    assert builder.node_counts["venue"] == 3
    assert builder.node_counts["timeslot"] == 168
    assert len(builder.edge_counts) > 0


def test_graph_exports_pyg(sample_users, sample_restaurants, sample_products,
                            sample_venues, sample_food_orders, sample_instamart_orders,
                            sample_dineout_bookings):
    data = InteractionData(
        food_orders=sample_food_orders, instamart_orders=sample_instamart_orders,
        dineout_bookings=sample_dineout_bookings, users=sample_users,
        restaurants=sample_restaurants, products=sample_products, venues=sample_venues,
    )
    builder = ConsumptionGraphBuilder(data).build()
    pyg_data = builder.to_pyg()
    assert "user" in pyg_data.node_types
    assert pyg_data["user"].num_nodes == 10
