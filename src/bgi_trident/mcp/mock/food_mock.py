"""Mock Swiggy Food MCP server for demo and testing."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from bgi_trident.mcp.protocol import MCPServer, MCPToolResult

FIXTURES = Path(__file__).parent / "fixtures"


class MockFoodMCP(MCPServer):
    def __init__(self) -> None:
        self._restaurants: list[dict] = []
        self._menus: dict[str, list] = {}
        self._cart: dict[str, Any] = {"items": [], "total": 0}

    async def connect(self) -> None:
        with open(FIXTURES / "restaurants.json") as f:
            self._restaurants = json.load(f)
        with open(FIXTURES / "menus.json") as f:
            self._menus = json.load(f)

    async def call_tool(self, tool_name: str, params: dict[str, Any]) -> MCPToolResult:
        if tool_name == "search_restaurants":
            query = params.get("query", "").lower()
            results = [
                r for r in self._restaurants
                if query in r["name"].lower() or query in r.get("cuisine", "").lower()
            ]
            return MCPToolResult(
                tool_name=tool_name, success=True,
                data={"restaurants": results[:5]},
            )
        elif tool_name == "get_restaurant_menu":
            rid = params.get("restaurant_id", "")
            menu = self._menus.get(
                rid, [{"name": "Biryani", "price": 350}, {"name": "Naan", "price": 60}]
            )
            return MCPToolResult(
                tool_name=tool_name, success=True, data={"menu": menu},
            )
        elif tool_name == "update_food_cart":
            items = params.get("items", [])
            self._cart["items"].extend(items)
            self._cart["total"] = sum(
                i.get("price", 0) for i in self._cart["items"]
            )
            summary = f"{len(self._cart['items'])} items, Rs {self._cart['total']}"
            return MCPToolResult(
                tool_name=tool_name, success=True,
                data={"cart": self._cart, "summary": summary},
            )
        elif tool_name == "place_food_order":
            order_id = "FO-MOCK-001"
            result = {
                "order_id": order_id, "status": "confirmed",
                "total": self._cart["total"],
            }
            self._cart = {"items": [], "total": 0}
            return MCPToolResult(
                tool_name=tool_name, success=True, data=result,
            )
        elif tool_name == "track_food_order":
            return MCPToolResult(
                tool_name=tool_name, success=True,
                data={"status": "preparing", "eta_minutes": 30},
            )
        return MCPToolResult(
            tool_name=tool_name, success=False,
            data={}, error=f"Unknown tool: {tool_name}",
        )

    async def close(self) -> None:
        pass

    def available_tools(self) -> list[str]:
        return [
            "search_restaurants", "get_restaurant_menu",
            "update_food_cart", "place_food_order", "track_food_order",
        ]
