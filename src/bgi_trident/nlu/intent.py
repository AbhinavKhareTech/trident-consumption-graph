"""Intent classifier for multi-domain consumption commands."""
from __future__ import annotations
from enum import Enum


class Intent(str, Enum):
    FOOD_ORDER = "food_order"
    INSTAMART_PURCHASE = "instamart_purchase"
    DINEOUT_BOOKING = "dineout_booking"
    MULTI_DOMAIN = "multi_domain"
    TRACK_ORDER = "track_order"
    CANCEL = "cancel"
    CONFIRM = "confirm"
    UNKNOWN = "unknown"


# Keyword-based intent detection (production uses fine-tuned classifier)
FOOD_KEYWORDS = {"order", "biryani", "pizza", "food", "restaurant", "dinner", "lunch", "khana", "oota"}
INSTAMART_KEYWORDS = {"buy", "add", "grocery", "milk", "harpic", "coke", "instamart", "saamaan"}
DINEOUT_KEYWORDS = {"book", "table", "reserve", "dineout", "restaurant booking", "saturday", "weekend"}
CONFIRM_KEYWORDS = {"yes", "haan", "ha", "confirm", "ok", "sari", "aunu", "amaam"}
CANCEL_KEYWORDS = {"no", "nahi", "cancel", "ruko", "stop", "beda", "venda"}


def classify_intent(text: str) -> Intent:
    """Classify user text into an intent. Supports code-mixed input."""
    tokens = set(text.lower().split())

    has_food = bool(tokens & FOOD_KEYWORDS)
    has_instamart = bool(tokens & INSTAMART_KEYWORDS)
    has_dineout = bool(tokens & DINEOUT_KEYWORDS)

    if tokens & CONFIRM_KEYWORDS:
        return Intent.CONFIRM
    if tokens & CANCEL_KEYWORDS:
        return Intent.CANCEL

    domain_count = sum([has_food, has_instamart, has_dineout])
    if domain_count >= 2:
        return Intent.MULTI_DOMAIN
    if has_food:
        return Intent.FOOD_ORDER
    if has_instamart:
        return Intent.INSTAMART_PURCHASE
    if has_dineout:
        return Intent.DINEOUT_BOOKING
    if "track" in tokens or "where" in tokens:
        return Intent.TRACK_ORDER
    return Intent.UNKNOWN
