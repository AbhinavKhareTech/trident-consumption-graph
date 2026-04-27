"""Mock Swiggy Dineout MCP server for demo and testing."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from bgi_trident.mcp.protocol import MCPServer, MCPToolResult

FIXTURES = Path(__file__).parent / "fixtures"


class MockDineoutMCP(MCPServer):
    def __init__(self) -> None:
        self._venues: list[dict] = []

    async def connect(self) -> None:
        with open(FIXTURES / "venues.json") as f:
            self._venues = json.load(f)

    async def call_tool(self, tool_name: str, params: dict[str, Any]) -> MCPToolResult:
        if tool_name == "search_restaurants_dineout":
            query = params.get("query", "").lower()
            results = [v for v in self._venues if query in v["name"].lower() or query in v.get("cuisine", "").lower()]
            return MCPToolResult(tool_name=tool_name, success=True, data={"restaurants": results[:5]})
        elif tool_name == "get_restaurant_details_dineout":
            rid = params.get("restaurant_id", "")
            venue = next((v for v in self._venues if v["id"] == rid), None)
            return MCPToolResult(tool_name=tool_name, success=bool(venue), data={"restaurant": venue or {}})
        elif tool_name == "get_available_slots":
            return MCPToolResult(tool_name=tool_name, success=True, data={
                "slots": [{"time": "7:00 PM", "available": True}, {"time": "7:30 PM", "available": True},
                          {"time": "8:00 PM", "available": True}, {"time": "8:30 PM", "available": False}]
            })
        elif tool_name == "book_table":
            return MCPToolResult(tool_name=tool_name, success=True, data={
                "booking_id": "DB-MOCK-001", "status": "confirmed",
                "time": params.get("time", "8:00 PM"), "party_size": params.get("party_size", 2)
            })
        return MCPToolResult(tool_name=tool_name, success=False, data={}, error=f"Unknown tool: {tool_name}")

    async def close(self) -> None:
        pass

    def available_tools(self) -> list[str]:
        return ["search_restaurants_dineout", "get_restaurant_details_dineout", "get_available_slots", "book_table"]
