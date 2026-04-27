"""Mock Swiggy Instamart MCP server for demo and testing."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from bgi_trident.mcp.protocol import MCPServer, MCPToolResult

FIXTURES = Path(__file__).parent / "fixtures"


class MockInstamartMCP(MCPServer):
    def __init__(self) -> None:
        self._products: list[dict] = []
        self._cart: dict[str, Any] = {"items": [], "total": 0}

    async def connect(self) -> None:
        with open(FIXTURES / "products.json") as f:
            self._products = json.load(f)

    async def call_tool(self, tool_name: str, params: dict[str, Any]) -> MCPToolResult:
        if tool_name == "search_products":
            query = params.get("query", "").lower()
            results = [
                p for p in self._products
                if query in p["name"].lower() or query in p.get("category", "").lower()
            ]
            return MCPToolResult(
                tool_name=tool_name, success=True,
                data={"products": results[:5]},
            )
        elif tool_name == "update_cart":
            pid = params.get("product_id", "")
            qty = params.get("quantity", 1)
            product = next(
                (p for p in self._products if p["id"] == pid),
                {"name": pid, "price": 100},
            )
            item = {
                "product_id": pid, "name": product["name"],
                "price": product["price"], "quantity": qty,
            }
            self._cart["items"].append(item)
            self._cart["total"] = sum(
                i["price"] * i["quantity"] for i in self._cart["items"]
            )
            summary = f"{len(self._cart['items'])} items, Rs {self._cart['total']}"
            return MCPToolResult(
                tool_name=tool_name, success=True,
                data={"cart": self._cart, "summary": summary},
            )
        elif tool_name == "checkout":
            order_id = "IO-MOCK-001"
            result = {
                "order_id": order_id, "status": "confirmed",
                "total": self._cart["total"],
            }
            self._cart = {"items": [], "total": 0}
            return MCPToolResult(
                tool_name=tool_name, success=True, data=result,
            )
        elif tool_name == "track_instamart_order":
            return MCPToolResult(
                tool_name=tool_name, success=True,
                data={"status": "packing", "eta_minutes": 15},
            )
        return MCPToolResult(
            tool_name=tool_name, success=False,
            data={}, error=f"Unknown tool: {tool_name}",
        )

    async def close(self) -> None:
        pass

    def available_tools(self) -> list[str]:
        return [
            "search_products", "update_cart",
            "checkout", "track_instamart_order",
        ]
