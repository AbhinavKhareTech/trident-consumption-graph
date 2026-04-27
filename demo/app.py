"""FastAPI demo server for BGI Trident.

Runs in mock MCP mode by default. Set MCP_MODE=live for production APIs.
"""
from __future__ import annotations

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from pydantic import BaseModel

from bgi_trident.config import MCP_MODE
from bgi_trident.mcp.mock.food_mock import MockFoodMCP
from bgi_trident.mcp.mock.instamart_mock import MockInstamartMCP
from bgi_trident.mcp.mock.dineout_mock import MockDineoutMCP
from bgi_trident.orchestrator.coordinator import TridentCoordinator, AgentTask, IntentType
from bgi_trident.nlu.intent import classify_intent
from bgi_trident.nlu.entities import extract_entities
from bgi_trident.nlu.codemix import normalize_codemix, detect_language

coordinator: TridentCoordinator | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global coordinator
    food = MockFoodMCP()
    instamart = MockInstamartMCP()
    dineout = MockDineoutMCP()
    coordinator = TridentCoordinator(food, instamart, dineout)
    await coordinator.start_session("demo_user", "kn")
    yield
    if coordinator:
        await coordinator.close()


app = FastAPI(title="BGI Trident Demo", version="0.1.0", lifespan=lifespan)


class UserInput(BaseModel):
    text: str
    language: str | None = None


class TridentResponse(BaseModel):
    intent: str
    entities: dict
    language: str
    results: dict
    summary: str


@app.post("/process", response_model=TridentResponse)
async def process_input(user_input: UserInput):
    """Process a user command through the full Trident pipeline."""
    text = user_input.text
    language = user_input.language or detect_language(text)
    normalized = normalize_codemix(text)
    intent = classify_intent(normalized)
    entities = extract_entities(normalized)

    results = {}
    summary = ""

    if coordinator and intent in (IntentType.FOOD_ORDER, IntentType.MULTI_DOMAIN):
        for restaurant in entities.restaurants or ["biryani"]:
            tasks = [AgentTask(agent_name="food", intent=IntentType.FOOD_ORDER,
                               params={"tool": "search_restaurants", "query": restaurant})]
            exec_result = await coordinator.execute(tasks)
            results["food"] = {k: v.data for k, v in exec_result.results.items()}
            summary = exec_result.summary

    return TridentResponse(
        intent=intent.value, entities={"restaurants": entities.restaurants, "products": entities.products},
        language=language, results=results, summary=summary or "Processed"
    )


@app.get("/health")
async def health():
    return {"status": "ok", "mcp_mode": MCP_MODE}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
