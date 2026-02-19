"""Representational Similarity Analysis for T-maze."""

from .rsa import (
    compute_rdm,
    compare_rdms,
    searchlight_rsa,
    model_rdm_tmaze
)

__all__ = [
    "compute_rdm",
    "compare_rdms",
    "searchlight_rsa",
    "model_rdm_tmaze"
]
