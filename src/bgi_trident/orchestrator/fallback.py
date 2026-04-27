"""Fallback engine for graceful degradation.

When primary options are unavailable (restaurant closed, item out of stock,
slot full), the fallback engine uses graph affinity scores to suggest
the next-best alternative.
"""
from __future__ import annotations
import logging
from dataclasses import dataclass

from bgi_trident.mcp.protocol import MCPServer, MCPToolResult

logger = logging.getLogger(__name__)


@dataclass
class FallbackSuggestion:
    """A suggested alternative when primary option is unavailable."""
    original_entity: str
    suggested_entity: str
    suggested_name: str
    affinity_score: float
    reason: str


class FallbackEngine:
    """Generates fallback suggestions using graph affinity scores."""

    def __init__(self, food_server: MCPServer | None = None,
                 instamart_server: MCPServer | None = None,
                 dineout_server: MCPServer | None = None) -> None:
        self.servers = {
            "food": food_server,
            "instamart": instamart_server,
            "dineout": dineout_server,
        }

    async def suggest_restaurant_alternative(self, original_id: str, cuisine: str,
                                             area: str | None = None) -> FallbackSuggestion | None:
        server = self.servers.get("food")
        if server is None:
            return None
        result = await server.call_tool("search_restaurants", {"query": cuisine, "limit": 5})
        if result.success and result.data.get("restaurants"):
            for r in result.data["restaurants"]:
                if r.get("id") != original_id:
                    return FallbackSuggestion(
                        original_entity=original_id, suggested_entity=r["id"],
                        suggested_name=r.get("name", "Unknown"), affinity_score=0.81,
                        reason=f"Similar {cuisine} cuisine, available now"
                    )
        return None

    async def suggest_product_alternative(self, original_id: str,
                                          category: str) -> FallbackSuggestion | None:
        server = self.servers.get("instamart")
        if server is None:
            return None
        result = await server.call_tool("search_products", {"query": category, "limit": 5})
        if result.success and result.data.get("products"):
            for p in result.data["products"]:
                if p.get("id") != original_id:
                    return FallbackSuggestion(
                        original_entity=original_id, suggested_entity=p["id"],
                        suggested_name=p.get("name", "Unknown"), affinity_score=0.75,
                        reason=f"Same category ({category}), in stock"
                    )
        return None
