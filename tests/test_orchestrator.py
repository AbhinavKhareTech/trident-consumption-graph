"""Tests for Trident orchestrator."""
import pytest

from bgi_trident.mcp.mock.dineout_mock import MockDineoutMCP
from bgi_trident.mcp.mock.food_mock import MockFoodMCP
from bgi_trident.mcp.mock.instamart_mock import MockInstamartMCP
from bgi_trident.orchestrator.coordinator import AgentTask, IntentType, TridentCoordinator


@pytest.mark.asyncio
async def test_coordinator_session():
    coord = TridentCoordinator(MockFoodMCP(), MockInstamartMCP(), MockDineoutMCP())
    session = await coord.start_session("U0001", "kn")
    assert session.user_id == "U0001"
    assert session.language == "kn"
    await coord.close()


@pytest.mark.asyncio
async def test_coordinator_execute_search():
    coord = TridentCoordinator(MockFoodMCP(), MockInstamartMCP(), MockDineoutMCP())
    await coord.start_session("U0001")
    tasks = [
        AgentTask(agent_name="food", intent=IntentType.FOOD_ORDER,
                  params={"tool": "search_restaurants", "query": "biryani"}),
    ]
    result = await coord.execute(tasks)
    assert len(result.tasks_completed) == 1
    assert len(result.tasks_failed) == 0
    await coord.close()
