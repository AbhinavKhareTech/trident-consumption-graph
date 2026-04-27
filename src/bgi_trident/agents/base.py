"""Base agent protocol for domain-specific MCP agents."""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any
from bgi_trident.mcp.protocol import MCPServer, MCPToolResult


class BaseAgent(ABC):
    """Protocol for domain agents. Each agent wraps one MCP server."""

    def __init__(self, server: MCPServer) -> None:
        self.server = server

    @abstractmethod
    async def search(self, query: str, **kwargs: Any) -> MCPToolResult:
        """Search for entities in this domain."""

    @abstractmethod
    async def add_to_cart(self, entity_id: str, **kwargs: Any) -> MCPToolResult:
        """Add an item to the domain cart."""

    @abstractmethod
    async def execute_order(self, **kwargs: Any) -> MCPToolResult:
        """Execute the domain transaction (requires confirmation)."""

    async def connect(self) -> None:
        await self.server.connect()

    async def close(self) -> None:
        await self.server.close()
