"""Entity extraction for consumption commands."""
from __future__ import annotations
import re
from dataclasses import dataclass


@dataclass
class ExtractedEntities:
    restaurants: list[str]
    products: list[str]
    quantities: dict[str, int]
    date: str | None = None
    time: str | None = None
    party_size: int | None = None
    area: str | None = None


# Known restaurant names for fuzzy matching
KNOWN_RESTAURANTS = {
    "meghana", "meghana's", "nandhana", "truffles", "mtr", "empire",
    "vidyarthi bhavan", "corner house", "domino's", "nagarjuna",
}

KNOWN_PRODUCTS = {
    "coke", "coca cola", "harpic", "milk", "bread", "eggs", "butter",
    "surf excel", "colgate", "lays", "red bull", "curd", "paneer",
}

QUANTITY_PATTERN = re.compile(r'(\d+)\s*(kg|l|ml|g|pc|pcs|packet|bottle|pack)?', re.IGNORECASE)


def extract_entities(text: str) -> ExtractedEntities:
    """Extract restaurants, products, quantities, and booking details from text."""
    lower = text.lower()
    tokens = lower.split()

    restaurants = [r for r in KNOWN_RESTAURANTS if r in lower]
    products = [p for p in KNOWN_PRODUCTS if p in lower]

    quantities: dict[str, int] = {}
    for match in QUANTITY_PATTERN.finditer(lower):
        qty = int(match.group(1))
        # Associate with nearest entity
        for entity in products + restaurants:
            if entity in lower[:match.start() + 20]:
                quantities[entity] = qty
                break

    party_size = None
    for token in tokens:
        if token.isdigit() and 1 <= int(token) <= 20:
            if "people" in lower or "person" in lower or "janaru" in lower:
                party_size = int(token)

    date = None
    if "saturday" in lower or "shanivara" in lower:
        date = "saturday"
    elif "sunday" in lower or "bhanuvaara" in lower:
        date = "sunday"
    elif "tomorrow" in lower or "naale" in lower:
        date = "tomorrow"

    return ExtractedEntities(
        restaurants=restaurants, products=products, quantities=quantities,
        date=date, party_size=party_size
    )
