"""Retell voice provider adapter."""
from __future__ import annotations

from bgi_trident.voice.protocol import SessionConfig, Transcript


class RetellAdapter:
    """Adapts Retell's agent protocol to VoiceProvider interface."""

    def __init__(self, api_key: str) -> None:
        self.api_key = api_key

    async def start_session(self, config: SessionConfig) -> str:
        return f"retell-{id(config)}"

    async def stream_audio_in(self, session_id: str, audio: bytes) -> None:
        pass

    async def receive_transcript(self, session_id: str) -> Transcript:
        return Transcript(text="", language="en", confidence=0.0, is_final=False)

    async def synthesize_speech(self, text: str, language: str) -> bytes:
        return b""

    async def end_session(self, session_id: str) -> None:
        pass
