"""ElevenLabs TTS + Deepgram ASR composite adapter."""
from __future__ import annotations
from bgi_trident.voice.protocol import SessionConfig, Transcript


class ElevenLabsAdapter:
    """Composite adapter: ElevenLabs for TTS, Deepgram for ASR."""

    def __init__(self, elevenlabs_key: str, deepgram_key: str = "") -> None:
        self.elevenlabs_key = elevenlabs_key
        self.deepgram_key = deepgram_key

    async def start_session(self, config: SessionConfig) -> str:
        return f"eleven-{id(config)}"

    async def stream_audio_in(self, session_id: str, audio: bytes) -> None:
        pass  # Route to Deepgram ASR

    async def receive_transcript(self, session_id: str) -> Transcript:
        return Transcript(text="", language="en", confidence=0.0, is_final=False)

    async def synthesize_speech(self, text: str, language: str) -> bytes:
        return b""  # Route to ElevenLabs TTS

    async def end_session(self, session_id: str) -> None:
        pass
