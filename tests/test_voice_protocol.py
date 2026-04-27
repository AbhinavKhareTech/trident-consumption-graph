"""Tests for voice provider protocol."""
import pytest
from bgi_trident.voice.adapters.vapi import VapiAdapter
from bgi_trident.voice.protocol import SessionConfig


@pytest.mark.asyncio
async def test_vapi_session_lifecycle():
    adapter = VapiAdapter(api_key="test-key")
    config = SessionConfig(language="kn")
    session_id = await adapter.start_session(config)
    assert session_id.startswith("vapi-")
    await adapter.end_session(session_id)
