"""MCP Server protocol and factory.

Defines the interface that all MCP server clients (mock and live) implement.
The mock-to-live swap is a config change, not a code change.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class MCPToolResult:
    """Result from calling an MCP tool."""

    tool_name: str
    success: bool
    data: dict[str, Any]
    error: str | None = None


class MCPServer(ABC):
    """Protocol for MCP server clients.

    Both mock and live implementations conform to this interface.
    Domain agents import MCPServer, never a concrete implementation.
    """

    @abstractmethod
    async def connect(self) -> None:
        """Establish connection to the MCP server."""

    @abstractmethod
    async def call_tool(self, tool_name: str, params: dict[str, Any]) -> MCPToolResult:
        """Call a tool on the MCP server.

        Args:
            tool_name: Name of the tool (e.g., 'search_restaurants', 'update_food_cart').
            params: Tool-specific parameters.

        Returns:
            MCPToolResult with success status and data.
        """

    @abstractmethod
    async def close(self) -> None:
        """Close connection to the MCP server."""

    @abstractmethod
    def available_tools(self) -> list[str]:
        """List available tools on this server."""
