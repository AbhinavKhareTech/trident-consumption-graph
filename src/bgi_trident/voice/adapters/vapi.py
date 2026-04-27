"""Vapi voice provider adapter. Default provider."""
from __future__ import annotations

from bgi_trident.voice.protocol import SessionConfig, Transcript


class VapiAdapter:
    """Adapts Vapi's WebSocket protocol to VoiceProvider interface."""

    def __init__(self, api_key: str) -> None:
        self.api_key = api_key
        self._sessions: dict[str, dict] = {}

    async def start_session(self, config: SessionConfig) -> str:
        session_id = f"vapi-{id(config)}"
        self._sessions[session_id] = {"config": config, "active": True}
        return session_id

    async def stream_audio_in(self, session_id: str, audio: bytes) -> None:
        if session_id not in self._sessions:
            raise ValueError(f"Unknown session: {session_id}")

    async def receive_transcript(self, session_id: str) -> Transcript:
        return Transcript(text="", language="en", confidence=0.0, is_final=False)

    async def synthesize_speech(self, text: str, language: str) -> bytes:
        return b""  # Placeholder

    async def end_session(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)
