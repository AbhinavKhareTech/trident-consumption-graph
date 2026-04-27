"""Voice Provider abstraction layer.

Defines the protocol that all voice infrastructure providers implement.
The orchestration layer imports VoiceProvider, never vapi or retell directly.
Swap providers via config, not code changes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass
class SessionConfig:
    """Configuration for a voice session."""

    language: str = "kn"  # ISO 639-1: kn=Kannada, hi=Hindi, ta=Tamil, en=English
    sample_rate: int = 16000
    encoding: str = "pcm_s16le"
    vad_enabled: bool = True
    endpointing_ms: int = 500


@dataclass
class Transcript:
    """Transcript from ASR with confidence and language detection."""

    text: str
    language: str
    confidence: float
    is_final: bool
    alternatives: list[str] | None = None


@runtime_checkable
class VoiceProvider(Protocol):
    """Protocol for voice infrastructure providers.

    Five methods define the complete voice session lifecycle.
    Any provider (Vapi, Bolna, Retell, ElevenLabs+Deepgram) can be
    adapted to this interface with ~400-500 lines of adapter code.
    """

    async def start_session(self, config: SessionConfig) -> str:
        """Start a voice session. Returns session_id."""
        ...

    async def stream_audio_in(self, session_id: str, audio: bytes) -> None:
        """Stream raw audio bytes from the user to the provider."""
        ...

    async def receive_transcript(self, session_id: str) -> Transcript:
        """Receive the next transcript from the provider's ASR."""
        ...

    async def synthesize_speech(self, text: str, language: str) -> bytes:
        """Convert text to speech audio bytes in the given language."""
        ...

    async def end_session(self, session_id: str) -> None:
        """End the voice session and release resources."""
        ...
