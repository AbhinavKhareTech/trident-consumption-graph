"""Tests for domain agents."""
import pytest
from bgi_trident.agents.food import FoodAgent
from bgi_trident.agents.instamart import InstamartAgent
from bgi_trident.agents.dineout import DineoutAgent
from bgi_trident.mcp.mock.food_mock import MockFoodMCP
from bgi_trident.mcp.mock.instamart_mock import MockInstamartMCP
from bgi_trident.mcp.mock.dineout_mock import MockDineoutMCP


@pytest.mark.asyncio
async def test_food_agent_search():
    agent = FoodAgent(MockFoodMCP())
    await agent.connect()
    result = await agent.search("biryani")
    assert result.success
    assert len(result.data["restaurants"]) > 0
    await agent.close()


@pytest.mark.asyncio
async def test_instamart_agent_search():
    agent = InstamartAgent(MockInstamartMCP())
    await agent.connect()
    result = await agent.search("harpic")
    assert result.success
    await agent.close()


@pytest.mark.asyncio
async def test_dineout_agent_slots():
    agent = DineoutAgent(MockDineoutMCP())
    await agent.connect()
    result = await agent.get_slots("V0001", "2025-12-06", 4)
    assert result.success
    assert len(result.data["slots"]) > 0
    await agent.close()
