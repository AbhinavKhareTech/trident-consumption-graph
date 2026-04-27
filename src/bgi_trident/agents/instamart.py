"""Instamart domain agent wrapping Swiggy Instamart MCP server."""
from __future__ import annotations

from typing import Any

from bgi_trident.agents.base import BaseAgent
from bgi_trident.mcp.protocol import MCPToolResult


class InstamartAgent(BaseAgent):
    """Agent for Swiggy Instamart MCP: search, cart, checkout, track."""

    async def search(self, query: str, **kwargs: Any) -> MCPToolResult:
        return await self.server.call_tool("search_products", {"query": query, **kwargs})

    async def add_to_cart(self, entity_id: str, **kwargs: Any) -> MCPToolResult:
        return await self.server.call_tool("update_cart", {
            "product_id": entity_id, "quantity": kwargs.get("quantity", 1), "action": "add"
        })

    async def execute_order(self, **kwargs: Any) -> MCPToolResult:
        return await self.server.call_tool("checkout", kwargs)

    async def track_order(self, order_id: str) -> MCPToolResult:
        return await self.server.call_tool("track_instamart_order", {"order_id": order_id})
