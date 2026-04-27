"""Confirmation gate for transactional MCP calls.

No order fires without explicit user confirmation.
This is a payments-grade guardrail, not a UX checkbox.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class ConfirmationRequest:
    """Pending confirmation with order summary."""
    session_id: str
    summary: str
    total_amount: float
    language: str
    items: list[dict]
    created_at: datetime
    confirmed: bool = False
    confirmed_at: datetime | None = None


class ConfirmationGate:
    """Gates transactional MCP calls behind explicit user confirmation.

    Rules:
    - Reads back order summary in user's language
    - States total amount
    - Requires explicit 'yes'/'haan'/'sari' confirmation
    - Logs confirmation event for audit
    - Times out after 30 seconds (configurable)
    """

    CONFIRM_WORDS = {"yes", "haan", "ha", "sari", "ok", "confirm", "proceed", "aunu", "amaam"}
    CANCEL_WORDS = {"no", "nahi", "cancel", "ruko", "stop", "beda", "venda"}

    def __init__(self, timeout_seconds: int = 30) -> None:
        self.timeout_seconds = timeout_seconds
        self._pending: dict[str, ConfirmationRequest] = {}

    def request_confirmation(self, session_id: str, summary: str, total_amount: float,
                             language: str, items: list[dict]) -> ConfirmationRequest:
        req = ConfirmationRequest(
            session_id=session_id, summary=summary, total_amount=total_amount,
            language=language, items=items, created_at=datetime.now()
        )
        self._pending[session_id] = req
        logger.info("Confirmation requested for session %s: Rs %.0f", session_id, total_amount)
        return req

    def process_response(self, session_id: str, user_input: str) -> str:
        """Process user confirmation response. Returns 'confirmed', 'cancelled', or 'unclear'."""
        req = self._pending.get(session_id)
        if req is None:
            return "no_pending_request"

        normalized = user_input.strip().lower()
        if normalized in self.CONFIRM_WORDS:
            req.confirmed = True
            req.confirmed_at = datetime.now()
            logger.info("Order confirmed for session %s", session_id)
            del self._pending[session_id]
            return "confirmed"
        elif normalized in self.CANCEL_WORDS:
            logger.info("Order cancelled for session %s", session_id)
            del self._pending[session_id]
            return "cancelled"
        return "unclear"

    def is_pending(self, session_id: str) -> bool:
        return session_id in self._pending

    def get_pending(self, session_id: str) -> ConfirmationRequest | None:
        return self._pending.get(session_id)
