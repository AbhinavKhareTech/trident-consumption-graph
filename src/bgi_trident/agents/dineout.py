"""Dineout domain agent wrapping Swiggy Dineout MCP server."""
from __future__ import annotations
from typing import Any
from bgi_trident.agents.base import BaseAgent
from bgi_trident.mcp.protocol import MCPToolResult


class DineoutAgent(BaseAgent):
    """Agent for Swiggy Dineout MCP: search, details, slots, book."""

    async def search(self, query: str, **kwargs: Any) -> MCPToolResult:
        return await self.server.call_tool("search_restaurants_dineout", {"query": query, **kwargs})

    async def get_details(self, venue_id: str) -> MCPToolResult:
        return await self.server.call_tool("get_restaurant_details_dineout", {"restaurant_id": venue_id})

    async def get_slots(self, venue_id: str, date: str, party_size: int = 2) -> MCPToolResult:
        return await self.server.call_tool("get_available_slots", {
            "restaurant_id": venue_id, "date": date, "party_size": party_size
        })

    async def add_to_cart(self, entity_id: str, **kwargs: Any) -> MCPToolResult:
        return await self.get_slots(entity_id, kwargs.get("date", ""), kwargs.get("party_size", 2))

    async def execute_order(self, **kwargs: Any) -> MCPToolResult:
        return await self.server.call_tool("book_table", kwargs)
