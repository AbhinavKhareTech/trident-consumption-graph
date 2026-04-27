"""Configuration for BGI Trident.

All provider and mode selection happens here.
Environment variables control mock-to-live swap.
"""

from __future__ import annotations

import os

# MCP Mode: "mock" for demo, "live" for production Swiggy APIs
MCP_MODE: str = os.getenv("MCP_MODE", "mock")

# Voice provider: "vapi", "bolna", "retell", "elevenlabs"
VOICE_PROVIDER: str = os.getenv("VOICE_PROVIDER", "vapi")

# Default language for voice sessions
DEFAULT_LANGUAGE: str = os.getenv("DEFAULT_LANGUAGE", "kn")  # Kannada

# GNN training
EMBED_DIM: int = 128
HIDDEN_DIM: int = 128
GNN_LAYERS: int = 2
TEMPORAL_HALF_LIFE_DAYS: float = 14.0

# XGBoost
XGB_MAX_DEPTH: int = 6
XGB_LEARNING_RATE: float = 0.1
XGB_N_ESTIMATORS: int = 200

# Ensemble
ENSEMBLE_CALIBRATE: bool = True

# Data paths
DATA_DIR: str = os.getenv("DATA_DIR", "src/data")
FIXTURES_DIR: str = os.getenv("FIXTURES_DIR", "src/bgi_trident/mcp/mock/fixtures")
MODEL_DIR: str = os.getenv("MODEL_DIR", "models")

# Swiggy API credentials (live mode only)
SWIGGY_API_KEY: str = os.getenv("SWIGGY_API_KEY", "")
SWIGGY_FOOD_MCP_URL: str = os.getenv("SWIGGY_FOOD_MCP_URL", "")
SWIGGY_INSTAMART_MCP_URL: str = os.getenv("SWIGGY_INSTAMART_MCP_URL", "")
SWIGGY_DINEOUT_MCP_URL: str = os.getenv("SWIGGY_DINEOUT_MCP_URL", "")

# Voice provider credentials
VAPI_API_KEY: str = os.getenv("VAPI_API_KEY", "")
BOLNA_API_KEY: str = os.getenv("BOLNA_API_KEY", "")
RETELL_API_KEY: str = os.getenv("RETELL_API_KEY", "")
