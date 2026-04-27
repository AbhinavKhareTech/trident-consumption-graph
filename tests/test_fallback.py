"""Tests for fallback engine."""
import pytest

from bgi_trident.mcp.mock.food_mock import MockFoodMCP
from bgi_trident.orchestrator.fallback import FallbackEngine


@pytest.mark.asyncio
async def test_restaurant_fallback():
    engine = FallbackEngine(food_server=MockFoodMCP())
    await engine.servers["food"].connect()
    suggestion = await engine.suggest_restaurant_alternative("R9999", "Biryani")
    assert suggestion is not None
    assert suggestion.suggested_name != ""
    assert suggestion.affinity_score > 0
