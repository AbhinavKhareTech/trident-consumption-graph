"""Code-mixed language handler for Kannada-English, Hindi-English, Tamil-English.

Normalizes code-mixed input so downstream intent and entity modules
can process it uniformly.
"""
from __future__ import annotations


# Common code-mixed tokens mapped to normalized English
CODEMIX_MAP: dict[str, str] = {
    # Hindi-English
    "karo": "do", "chahiye": "want", "bhi": "also", "aur": "and",
    "se": "from", "mein": "in", "hatao": "remove", "ruko": "stop",
    "haan": "yes", "nahi": "no", "kitna": "how much",
    # Kannada-English
    "maadu": "do", "beku": "want", "nu": "also", "inda": "from",
    "ge": "for", "nalli": "in", "sari": "ok", "beda": "no",
    "oota": "food", "maadla": "shall do",
    # Tamil-English
    "pannu": "do", "venum": "want", "um": "also", "irundhu": "from",
    "ku": "for", "la": "in", "amaam": "yes", "venda": "no",
}


def normalize_codemix(text: str) -> str:
    """Normalize code-mixed tokens to English equivalents."""
    tokens = text.split()
    normalized = []
    for token in tokens:
        lower = token.lower().strip(".,!?")
        mapped = CODEMIX_MAP.get(lower)
        normalized.append(mapped if mapped else token)
    return " ".join(normalized)


def detect_language(text: str) -> str:
    """Detect primary language from code-mixed input."""
    lower = text.lower()
    kannada_signals = {"maadu", "beku", "inda", "nalli", "sari", "oota", "maadla", "ge"}
    hindi_signals = {"karo", "chahiye", "mein", "haan", "nahi", "hatao", "se", "aur"}
    tamil_signals = {"pannu", "venum", "irundhu", "amaam", "venda", "la", "ku"}

    tokens = set(lower.split())
    scores = {
        "kn": len(tokens & kannada_signals),
        "hi": len(tokens & hindi_signals),
        "ta": len(tokens & tamil_signals),
    }
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "en"
