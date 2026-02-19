"""
Group-level statistical analysis for T-maze data.

Includes:
- Group-level t-tests and ANOVA
- Linear mixed effects models
- Permutation testing with cluster correction
- Effect size calculations
"""

from .group_stats import (
    group_ttest,
    paired_ttest,
    repeated_measures_anova,
    group_roi_analysis,
    fdr_correction,
    bonferroni_correction
)

from .mixed_effects import (
    LinearMixedEffects,
    lme_roi_analysis,
    lme_temporal_analysis
)

from .permutation import (
    GroupPermutationTest,
    cluster_permutation_group,
    tfce_correction,
    permutation_cluster_1d
)

from .effect_sizes import (
    cohens_d,
    hedges_g,
    bootstrap_ci,
    bayesian_estimation,
    EffectSizeResult
)

__all__ = [
    # group_stats
    'group_ttest',
    'paired_ttest',
    'repeated_measures_anova',
    'group_roi_analysis',
    'fdr_correction',
    'bonferroni_correction',
    # mixed_effects
    'LinearMixedEffects',
    'lme_roi_analysis',
    'lme_temporal_analysis',
    # permutation
    'GroupPermutationTest',
    'cluster_permutation_group',
    'tfce_correction',
    'permutation_cluster_1d',
    # effect_sizes
    'cohens_d',
    'hedges_g',
    'bootstrap_ci',
    'bayesian_estimation',
    'EffectSizeResult'
]
