"""Cross-agent session state manager.

Re-exports SessionState from coordinator.py and adds state persistence.
"""
from __future__ import annotations

from bgi_trident.orchestrator.coordinator import SessionState


class SessionStateManager:
    """Manages session state lifecycle across agent interactions."""

    def __init__(self) -> None:
        self._sessions: dict[str, SessionState] = {}

    def create(self, user_id: str, language: str = "en") -> SessionState:
        session = SessionState(user_id=user_id, language=language)
        self._sessions[user_id] = session
        return session

    def get(self, user_id: str) -> SessionState | None:
        return self._sessions.get(user_id)

    def update_food_cart(self, user_id: str, items: dict) -> None:
        session = self._sessions.get(user_id)
        if session:
            session.food_cart.update(items)
            session.total_amount = (
                session.food_cart.get("total", 0) + session.instamart_cart.get("total", 0)
            )

    def update_instamart_cart(self, user_id: str, items: dict) -> None:
        session = self._sessions.get(user_id)
        if session:
            session.instamart_cart.update(items)
            session.total_amount = (
                session.food_cart.get("total", 0) + session.instamart_cart.get("total", 0)
            )

    def update_dineout_booking(self, user_id: str, booking: dict) -> None:
        session = self._sessions.get(user_id)
        if session:
            session.dineout_booking.update(booking)

    def clear(self, user_id: str) -> None:
        self._sessions.pop(user_id, None)

    @property
    def active_sessions(self) -> int:
        return len(self._sessions)
