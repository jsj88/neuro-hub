"""LLM prompt templates for extraction and screening."""

from .extraction_prompts import (
    COORDINATE_EXTRACTION_PROMPT,
    EFFECT_SIZE_EXTRACTION_PROMPT,
    SCREENING_PROMPT,
    STUDY_METADATA_PROMPT
)

__all__ = [
    "COORDINATE_EXTRACTION_PROMPT",
    "EFFECT_SIZE_EXTRACTION_PROMPT", 
    "SCREENING_PROMPT",
    "STUDY_METADATA_PROMPT"
]
