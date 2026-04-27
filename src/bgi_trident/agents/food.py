"""Food domain agent wrapping Swiggy Food MCP server."""
from __future__ import annotations
from typing import Any
from bgi_trident.agents.base import BaseAgent
from bgi_trident.mcp.protocol import MCPServer, MCPToolResult


class FoodAgent(BaseAgent):
    """Agent for Swiggy Food MCP: search, menu, cart, order, track."""

    async def search(self, query: str, **kwargs: Any) -> MCPToolResult:
        return await self.server.call_tool("search_restaurants", {"query": query, **kwargs})

    async def get_menu(self, restaurant_id: str) -> MCPToolResult:
        return await self.server.call_tool("get_restaurant_menu", {"restaurant_id": restaurant_id})

    async def add_to_cart(self, entity_id: str, **kwargs: Any) -> MCPToolResult:
        return await self.server.call_tool("update_food_cart", {
            "restaurant_id": kwargs.get("restaurant_id", entity_id),
            "items": kwargs.get("items", []), "action": "add"
        })

    async def execute_order(self, **kwargs: Any) -> MCPToolResult:
        return await self.server.call_tool("place_food_order", kwargs)

    async def track_order(self, order_id: str) -> MCPToolResult:
        return await self.server.call_tool("track_food_order", {"order_id": order_id})
