"""Live Swiggy SwiggyFoodMCP client. Activated when MCP_MODE=live."""
from __future__ import annotations
from typing import Any
import httpx
from bgi_trident.mcp.protocol import MCPServer, MCPToolResult


class SwiggyFoodMCP(MCPServer):
    def __init__(self, url: str, api_key: str) -> None:
        self.url = url
        self.api_key = api_key
        self._client: httpx.AsyncClient | None = None

    async def connect(self) -> None:
        self._client = httpx.AsyncClient(
            base_url=self.url,
            headers={"Authorization": f"Bearer {self.api_key}"},
            timeout=30.0,
        )

    async def call_tool(self, tool_name: str, params: dict[str, Any]) -> MCPToolResult:
        if self._client is None:
            raise RuntimeError("Not connected")
        try:
            resp = await self._client.post(f"/tools/{tool_name}", json=params)
            resp.raise_for_status()
            return MCPToolResult(tool_name=tool_name, success=True, data=resp.json())
        except Exception as e:
            return MCPToolResult(tool_name=tool_name, success=False, data={}, error=str(e))

    async def close(self) -> None:
        if self._client:
            await self._client.aclose()

    def available_tools(self) -> list[str]:
        return []
