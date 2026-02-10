"""
Configuration settings for meta-analysis toolkit.

Set your API keys here or via environment variables.
"""

import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"

# LLM API Configuration
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# Default LLM settings
DEFAULT_LLM_PROVIDER = "anthropic"
DEFAULT_MODEL = "claude-sonnet-4-20250514"

# PubMed API (optional, for higher rate limits)
NCBI_API_KEY = os.getenv("NCBI_API_KEY", "")

# Coordinate validation settings
MNI_BOUNDS = {
    "x": (-90, 90),
    "y": (-126, 90),
    "z": (-72, 108)
}

# Default analysis settings
DEFAULT_ALE_SETTINGS = {
    "kernel_fwhm": None,  # Sample-size based
    "null_method": "approximate",
    "n_iters": 10000,
    "alpha": 0.05
}

DEFAULT_EFFECT_SIZE_SETTINGS = {
    "method": "DL",  # DerSimonian-Laird
    "ci_level": 0.95
}
