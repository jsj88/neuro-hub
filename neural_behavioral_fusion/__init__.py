"""
Neural-Behavioral Fusion with Linear Mixed-Effects Models.

Replace simple correlations (pearsonr) with proper LME models
for linking behavioral model outputs (RPEs) with neural signals (REWP).

Supports random intercepts/slopes, multi-channel analysis,
time-resolved analysis, and model comparison.
"""

from .core import FusionLME, FusionLMEResult
from .models import MODEL_REGISTRY, get_model_spec, list_models
from .multichannel import multichannel_lme
from .temporal import temporal_lme
from .comparison import compare_predictors, likelihood_ratio_test
from .data import simulate_fusion_dataset, load_real_fusion_data
from .viz import (
    plot_rpe_rewp_scatter,
    plot_random_effects,
    plot_multichannel_results,
    plot_temporal_lme,
    plot_model_comparison,
    plot_diagnostic,
)

__all__ = [
    # Core
    "FusionLME",
    "FusionLMEResult",
    # Models
    "MODEL_REGISTRY",
    "get_model_spec",
    "list_models",
    # Analysis
    "multichannel_lme",
    "temporal_lme",
    "compare_predictors",
    "likelihood_ratio_test",
    # Data
    "simulate_fusion_dataset",
    "load_real_fusion_data",
    # Visualization
    "plot_rpe_rewp_scatter",
    "plot_random_effects",
    "plot_multichannel_results",
    "plot_temporal_lme",
    "plot_model_comparison",
    "plot_diagnostic",
]
