"""Trident Orchestrator: Multi-agent coordinator.

Decomposes user intents into parallel agent tasks across Food, Instamart,
and Dineout MCP servers. Manages shared session state, resolves constraints,
and gates every transactional call behind explicit user confirmation.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from bgi_trident.mcp.protocol import MCPServer, MCPToolResult

logger = logging.getLogger(__name__)


class IntentType(str, Enum):
    FOOD_ORDER = "food_order"
    INSTAMART_PURCHASE = "instamart_purchase"
    DINEOUT_BOOKING = "dineout_booking"
    MULTI_DOMAIN = "multi_domain"


@dataclass
class AgentTask:
    """A single task dispatched to a domain agent."""

    agent_name: str  # "food", "instamart", "dineout"
    intent: IntentType
    params: dict[str, Any]
    priority: int = 0
    requires_confirmation: bool = False


@dataclass
class SessionState:
    """Cross-agent session state shared across all domain agents."""

    user_id: str
    delivery_address: str | None = None
    payment_method: str | None = None
    language: str = "en"
    food_cart: dict[str, Any] = field(default_factory=dict)
    instamart_cart: dict[str, Any] = field(default_factory=dict)
    dineout_booking: dict[str, Any] = field(default_factory=dict)
    confirmation_pending: bool = False
    total_amount: float = 0.0


@dataclass
class ExecutionResult:
    """Result from executing a set of agent tasks."""

    tasks_completed: list[str]
    tasks_failed: list[str]
    results: dict[str, MCPToolResult]
    total_amount: float
    confirmation_required: bool
    summary: str  # Human-readable summary in user's language


class TridentCoordinator:
    """Multi-agent coordinator for cross-domain execution.

    Handles the Decide and Execute phases of the Predict-Decide-Execute loop.
    Kumo (graph engine) provides the Predict phase.
    """

    def __init__(
        self,
        food_server: MCPServer,
        instamart_server: MCPServer,
        dineout_server: MCPServer,
    ) -> None:
        self.servers = {
            "food": food_server,
            "instamart": instamart_server,
            "dineout": dineout_server,
        }
        self.session: SessionState | None = None

    async def start_session(self, user_id: str, language: str = "en") -> SessionState:
        """Initialize a new cross-domain session."""
        self.session = SessionState(user_id=user_id, language=language)
        for server in self.servers.values():
            await server.connect()
        logger.info("Session started for user %s", user_id)
        return self.session

    async def execute(self, tasks: list[AgentTask]) -> ExecutionResult:
        """Execute a set of agent tasks in parallel where possible.

        Tasks are grouped by agent and executed concurrently.
        All results are collected before any transactional call fires.
        Transactional calls require explicit user confirmation.
        """
        if self.session is None:
            raise RuntimeError("No active session. Call start_session() first.")

        # Separate search/browse tasks from transactional tasks
        search_tasks = [t for t in tasks if not t.requires_confirmation]
        txn_tasks = [t for t in tasks if t.requires_confirmation]

        # Execute search tasks in parallel
        results: dict[str, MCPToolResult] = {}
        completed: list[str] = []
        failed: list[str] = []

        if search_tasks:
            search_results = await asyncio.gather(
                *[self._execute_task(t) for t in search_tasks],
                return_exceptions=True,
            )
            for task, result in zip(search_tasks, search_results):
                task_key = f"{task.agent_name}:{task.params.get('tool', 'unknown')}"
                if isinstance(result, Exception):
                    failed.append(task_key)
                    logger.error("Task failed: %s - %s", task_key, result)
                else:
                    results[task_key] = result
                    completed.append(task_key)

        # Calculate total
        total = self._calculate_total()

        # Transactional tasks are NOT executed here -- they wait for confirmation
        confirmation_required = len(txn_tasks) > 0

        summary = self._build_summary(results, total, self.session.language)

        return ExecutionResult(
            tasks_completed=completed,
            tasks_failed=failed,
            results=results,
            total_amount=total,
            confirmation_required=confirmation_required,
            summary=summary,
        )

    async def confirm_and_execute(self, tasks: list[AgentTask]) -> ExecutionResult:
        """Execute transactional tasks after user confirmation.

        This is the confirmation gate. No MCP call to place_food_order,
        checkout, or book_table fires without passing through here.
        """
        if self.session is None:
            raise RuntimeError("No active session.")

        results: dict[str, MCPToolResult] = {}
        completed: list[str] = []
        failed: list[str] = []

        for task in tasks:
            task_key = f"{task.agent_name}:{task.params.get('tool', 'unknown')}"
            try:
                result = await self._execute_task(task)
                results[task_key] = result
                completed.append(task_key)
                logger.info("Transaction confirmed and executed: %s", task_key)
            except Exception as e:
                failed.append(task_key)
                logger.error("Transaction failed: %s - %s", task_key, e)

        return ExecutionResult(
            tasks_completed=completed,
            tasks_failed=failed,
            results=results,
            total_amount=self._calculate_total(),
            confirmation_required=False,
            summary=f"Completed {len(completed)} transactions, {len(failed)} failed.",
        )

    async def handle_fallback(
        self,
        failed_task: AgentTask,
        ensemble_scores: dict[str, float],
    ) -> AgentTask | None:
        """Generate fallback task when primary option is unavailable.

        Uses ensemble scores to find the next-best alternative.
        Example: Meghana's is closed -> suggest Nandhana Palace (affinity 0.81).
        """
        if failed_task.agent_name == "food":
            # Search for alternative restaurants with similar cuisine
            search_result = await self.servers["food"].call_tool(
                "search_restaurants",
                {"query": failed_task.params.get("cuisine", "biryani"), "limit": 5},
            )
            if search_result.success and search_result.data.get("restaurants"):
                alt = search_result.data["restaurants"][0]
                return AgentTask(
                    agent_name="food",
                    intent=IntentType.FOOD_ORDER,
                    params={
                        "tool": "search_restaurants",
                        "restaurant_id": alt.get("id"),
                        "restaurant_name": alt.get("name"),
                        "is_fallback": True,
                    },
                )
        return None

    async def _execute_task(self, task: AgentTask) -> MCPToolResult:
        """Execute a single task against the appropriate MCP server."""
        server = self.servers.get(task.agent_name)
        if server is None:
            raise ValueError(f"Unknown agent: {task.agent_name}")

        tool_name = task.params.pop("tool", None)
        if tool_name is None:
            raise ValueError(f"No tool specified in task params for {task.agent_name}")

        return await server.call_tool(tool_name, task.params)

    def _calculate_total(self) -> float:
        """Calculate total amount across all carts."""
        if self.session is None:
            return 0.0
        food_total = self.session.food_cart.get("total", 0.0)
        insta_total = self.session.instamart_cart.get("total", 0.0)
        return food_total + insta_total

    def _build_summary(
        self,
        results: dict[str, MCPToolResult],
        total: float,
        language: str,
    ) -> str:
        """Build human-readable order summary in user's language."""
        # Placeholder -- production version uses Swar NLU for vernacular generation
        items = []
        for key, result in results.items():
            if result.success:
                items.append(f"{key}: {result.data.get('summary', 'OK')}")
        summary = "; ".join(items)
        return f"Order summary (Rs {total:.0f}): {summary}"

    async def close(self) -> None:
        """Close all MCP server connections."""
        for server in self.servers.values():
            await server.close()
        self.session = None
