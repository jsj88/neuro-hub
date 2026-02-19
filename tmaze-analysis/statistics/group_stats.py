"""
Group-level statistical functions for T-maze analysis.

Provides statistical tests for comparing classification performance across subjects.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from scipy import stats

try:
    import pingouin as pg
    HAS_PINGOUIN = True
except ImportError:
    HAS_PINGOUIN = False


@dataclass
class GroupStatResult:
    """Container for group-level statistical results."""
    statistic: float
    p_value: float
    test_name: str
    n_subjects: int
    mean: float
    std: float
    ci_lower: float
    ci_upper: float
    effect_size: Optional[float] = None
    df: Optional[float] = None
    metadata: Dict = field(default_factory=dict)

    def __repr__(self):
        sig = "*" if self.p_value < 0.05 else ""
        return (f"GroupStatResult({self.test_name}: "
                f"stat={self.statistic:.3f}, p={self.p_value:.4f}{sig}, "
                f"n={self.n_subjects}, mean={self.mean:.3f}±{self.std:.3f})")

    def is_significant(self, alpha: float = 0.05) -> bool:
        """Check if result is significant at given alpha level."""
        return self.p_value < alpha


def group_ttest(
    accuracies: np.ndarray,
    chance_level: float = 0.5,
    alternative: str = 'greater',
    confidence: float = 0.95
) -> GroupStatResult:
    """
    One-sample t-test comparing group accuracy to chance.

    Tests whether mean classification accuracy across subjects is
    significantly above chance level.

    Parameters
    ----------
    accuracies : np.ndarray
        Per-subject accuracies (n_subjects,)
    chance_level : float
        Chance level to test against (default: 0.5 for binary)
    alternative : str
        'two-sided', 'greater', or 'less'
    confidence : float
        Confidence level for CI (default: 0.95)

    Returns
    -------
    GroupStatResult
        Statistical test results with effect size
    """
    accuracies = np.asarray(accuracies)
    n = len(accuracies)

    # T-test
    t_stat, p_value = stats.ttest_1samp(accuracies, chance_level)

    # Adjust p-value for one-sided test
    if alternative == 'greater':
        p_value = p_value / 2 if t_stat > 0 else 1 - p_value / 2
    elif alternative == 'less':
        p_value = p_value / 2 if t_stat < 0 else 1 - p_value / 2

    # Confidence interval
    sem = stats.sem(accuracies)
    t_crit = stats.t.ppf((1 + confidence) / 2, df=n-1)
    ci_lower = np.mean(accuracies) - t_crit * sem
    ci_upper = np.mean(accuracies) + t_crit * sem

    # Cohen's d (effect size relative to chance)
    effect_size = (np.mean(accuracies) - chance_level) / np.std(accuracies, ddof=1)

    return GroupStatResult(
        statistic=t_stat,
        p_value=p_value,
        test_name='one-sample t-test',
        n_subjects=n,
        mean=np.mean(accuracies),
        std=np.std(accuracies, ddof=1),
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        effect_size=effect_size,
        df=n - 1,
        metadata={
            'chance_level': chance_level,
            'alternative': alternative,
            'confidence': confidence
        }
    )


def paired_ttest(
    accuracies_a: np.ndarray,
    accuracies_b: np.ndarray,
    alternative: str = 'two-sided',
    confidence: float = 0.95
) -> GroupStatResult:
    """
    Paired t-test comparing two conditions across subjects.

    Useful for comparing classification methods or condition contrasts.

    Parameters
    ----------
    accuracies_a : np.ndarray
        Per-subject accuracies for condition A (n_subjects,)
    accuracies_b : np.ndarray
        Per-subject accuracies for condition B (n_subjects,)
    alternative : str
        'two-sided', 'greater', or 'less' (A vs B)
    confidence : float
        Confidence level for CI

    Returns
    -------
    GroupStatResult
        Statistical test results with effect size
    """
    accuracies_a = np.asarray(accuracies_a)
    accuracies_b = np.asarray(accuracies_b)

    if len(accuracies_a) != len(accuracies_b):
        raise ValueError("Arrays must have same length for paired test")

    n = len(accuracies_a)
    diff = accuracies_a - accuracies_b

    # Paired t-test
    t_stat, p_value = stats.ttest_rel(accuracies_a, accuracies_b)

    # Adjust for alternative
    if alternative == 'greater':
        p_value = p_value / 2 if t_stat > 0 else 1 - p_value / 2
    elif alternative == 'less':
        p_value = p_value / 2 if t_stat < 0 else 1 - p_value / 2

    # CI on difference
    sem = stats.sem(diff)
    t_crit = stats.t.ppf((1 + confidence) / 2, df=n-1)
    ci_lower = np.mean(diff) - t_crit * sem
    ci_upper = np.mean(diff) + t_crit * sem

    # Cohen's d for paired samples
    effect_size = np.mean(diff) / np.std(diff, ddof=1)

    return GroupStatResult(
        statistic=t_stat,
        p_value=p_value,
        test_name='paired t-test',
        n_subjects=n,
        mean=np.mean(diff),
        std=np.std(diff, ddof=1),
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        effect_size=effect_size,
        df=n - 1,
        metadata={
            'mean_a': np.mean(accuracies_a),
            'mean_b': np.mean(accuracies_b),
            'alternative': alternative,
            'confidence': confidence
        }
    )


def repeated_measures_anova(
    data: np.ndarray,
    factor_levels: Optional[List[str]] = None,
    subject_ids: Optional[List[str]] = None
) -> Dict:
    """
    Repeated-measures ANOVA for within-subject comparisons.

    Tests whether accuracy differs across multiple conditions.

    Parameters
    ----------
    data : np.ndarray
        Accuracy matrix (n_subjects, n_conditions)
    factor_levels : List[str], optional
        Names of conditions
    subject_ids : List[str], optional
        Subject identifiers

    Returns
    -------
    Dict
        ANOVA results including F-statistic, p-value, effect size
    """
    data = np.asarray(data)
    n_subjects, n_conditions = data.shape

    if factor_levels is None:
        factor_levels = [f'Condition_{i}' for i in range(n_conditions)]
    if subject_ids is None:
        subject_ids = [f'S{i:02d}' for i in range(n_subjects)]

    if HAS_PINGOUIN:
        # Use pingouin for proper sphericity correction
        import pandas as pd

        # Create long-format DataFrame
        df_long = pd.DataFrame({
            'subject': np.repeat(subject_ids, n_conditions),
            'condition': np.tile(factor_levels, n_subjects),
            'accuracy': data.flatten()
        })

        # Run repeated-measures ANOVA
        aov = pg.rm_anova(
            data=df_long,
            dv='accuracy',
            within='condition',
            subject='subject',
            correction=True  # Greenhouse-Geisser correction
        )

        return {
            'F': aov['F'].values[0],
            'p_value': aov['p-unc'].values[0],
            'p_value_gg': aov['p-GG-corr'].values[0] if 'p-GG-corr' in aov.columns else None,
            'eta_squared': aov['np2'].values[0],  # Partial eta-squared
            'epsilon': aov['eps'].values[0] if 'eps' in aov.columns else None,
            'df_factor': aov['ddof1'].values[0],
            'df_error': aov['ddof2'].values[0],
            'sphericity_violated': aov['sphericity'].values[0] if 'sphericity' in aov.columns else None,
            'n_subjects': n_subjects,
            'n_conditions': n_conditions,
            'factor_levels': factor_levels,
            'method': 'pingouin.rm_anova'
        }
    else:
        # Fallback to scipy
        # Simple repeated-measures F-test using within-subject design
        grand_mean = data.mean()
        subject_means = data.mean(axis=1)
        condition_means = data.mean(axis=0)

        # Sum of squares
        ss_between = n_conditions * np.sum((subject_means - grand_mean) ** 2)
        ss_treatment = n_subjects * np.sum((condition_means - grand_mean) ** 2)
        ss_total = np.sum((data - grand_mean) ** 2)
        ss_error = ss_total - ss_between - ss_treatment

        # Degrees of freedom
        df_treatment = n_conditions - 1
        df_subjects = n_subjects - 1
        df_error = df_treatment * df_subjects

        # Mean squares and F
        ms_treatment = ss_treatment / df_treatment
        ms_error = ss_error / df_error
        f_stat = ms_treatment / ms_error

        # P-value
        p_value = 1 - stats.f.cdf(f_stat, df_treatment, df_error)

        # Partial eta-squared
        eta_sq = ss_treatment / (ss_treatment + ss_error)

        return {
            'F': f_stat,
            'p_value': p_value,
            'eta_squared': eta_sq,
            'df_factor': df_treatment,
            'df_error': df_error,
            'n_subjects': n_subjects,
            'n_conditions': n_conditions,
            'factor_levels': factor_levels,
            'method': 'scipy (no sphericity correction)'
        }


def group_roi_analysis(
    subject_roi_accuracies: Dict[str, Dict[str, float]],
    chance_level: float = 0.5,
    correction: str = 'fdr',
    alpha: float = 0.05
) -> Dict[str, GroupStatResult]:
    """
    Test all ROIs for significance with multiple comparison correction.

    Parameters
    ----------
    subject_roi_accuracies : Dict[str, Dict[str, float]]
        Nested dict: {subject_id: {roi_name: accuracy}}
    chance_level : float
        Chance level for t-tests
    correction : str
        'fdr', 'bonferroni', or 'none'
    alpha : float
        Significance threshold

    Returns
    -------
    Dict[str, GroupStatResult]
        Results for each ROI with corrected p-values
    """
    # Reorganize by ROI
    subjects = list(subject_roi_accuracies.keys())
    first_subj = subjects[0]
    roi_names = list(subject_roi_accuracies[first_subj].keys())

    results = {}
    p_values = []

    # Run t-test for each ROI
    for roi in roi_names:
        accuracies = np.array([
            subject_roi_accuracies[subj][roi]
            for subj in subjects
            if roi in subject_roi_accuracies[subj]
        ])

        if len(accuracies) >= 3:  # Minimum for t-test
            result = group_ttest(accuracies, chance_level=chance_level)
            results[roi] = result
            p_values.append(result.p_value)

    # Apply multiple comparison correction
    p_values = np.array(p_values)

    if correction == 'fdr':
        corrected_pvals, significant = fdr_correction(p_values, alpha=alpha)
    elif correction == 'bonferroni':
        corrected_pvals, significant = bonferroni_correction(p_values, alpha=alpha)
    else:
        corrected_pvals = p_values
        significant = p_values < alpha

    # Update results with corrected p-values
    for i, roi in enumerate(results.keys()):
        results[roi].metadata['p_corrected'] = corrected_pvals[i]
        results[roi].metadata['significant_corrected'] = significant[i]
        results[roi].metadata['correction'] = correction

    return results


def fdr_correction(
    p_values: np.ndarray,
    alpha: float = 0.05,
    method: str = 'bh'
) -> Tuple[np.ndarray, np.ndarray]:
    """
    False Discovery Rate correction for multiple comparisons.

    Parameters
    ----------
    p_values : np.ndarray
        Array of p-values
    alpha : float
        Significance threshold
    method : str
        'bh' (Benjamini-Hochberg) or 'by' (Benjamini-Yekutieli)

    Returns
    -------
    corrected_pvals : np.ndarray
        FDR-corrected p-values
    significant : np.ndarray
        Boolean mask of significant tests
    """
    p_values = np.asarray(p_values)
    n = len(p_values)

    # Sort p-values
    sorted_idx = np.argsort(p_values)
    sorted_pvals = p_values[sorted_idx]

    # Calculate critical values
    rank = np.arange(1, n + 1)

    if method == 'bh':
        # Benjamini-Hochberg
        threshold = rank * alpha / n
    elif method == 'by':
        # Benjamini-Yekutieli (more conservative)
        c_m = np.sum(1 / rank)
        threshold = rank * alpha / (n * c_m)
    else:
        raise ValueError(f"Unknown method: {method}")

    # Find largest k where p(k) <= threshold(k)
    below_threshold = sorted_pvals <= threshold
    if np.any(below_threshold):
        max_k = np.max(np.where(below_threshold)[0]) + 1
    else:
        max_k = 0

    # Calculate corrected p-values
    corrected = np.zeros(n)
    corrected[sorted_idx] = np.minimum.accumulate(
        sorted_pvals[::-1] * n / rank[::-1]
    )[::-1]
    corrected = np.minimum(corrected, 1.0)

    # Determine significance
    significant = np.zeros(n, dtype=bool)
    significant[sorted_idx[:max_k]] = True

    return corrected, significant


def bonferroni_correction(
    p_values: np.ndarray,
    alpha: float = 0.05
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Bonferroni correction for multiple comparisons.

    Parameters
    ----------
    p_values : np.ndarray
        Array of p-values
    alpha : float
        Significance threshold

    Returns
    -------
    corrected_pvals : np.ndarray
        Bonferroni-corrected p-values
    significant : np.ndarray
        Boolean mask of significant tests
    """
    p_values = np.asarray(p_values)
    n = len(p_values)

    corrected = np.minimum(p_values * n, 1.0)
    significant = corrected < alpha

    return corrected, significant


def posthoc_pairwise(
    data: np.ndarray,
    factor_levels: Optional[List[str]] = None,
    correction: str = 'bonferroni'
) -> Dict[Tuple[str, str], Dict]:
    """
    Post-hoc pairwise comparisons after significant ANOVA.

    Parameters
    ----------
    data : np.ndarray
        Accuracy matrix (n_subjects, n_conditions)
    factor_levels : List[str], optional
        Names of conditions
    correction : str
        'bonferroni', 'fdr', or 'none'

    Returns
    -------
    Dict[Tuple[str, str], Dict]
        Pairwise comparison results
    """
    n_subjects, n_conditions = data.shape

    if factor_levels is None:
        factor_levels = [f'Condition_{i}' for i in range(n_conditions)]

    results = {}
    comparisons = []
    p_values = []

    # All pairwise comparisons
    for i in range(n_conditions):
        for j in range(i + 1, n_conditions):
            t_stat, p_val = stats.ttest_rel(data[:, i], data[:, j])
            comparisons.append((factor_levels[i], factor_levels[j]))
            p_values.append(p_val)

            results[(factor_levels[i], factor_levels[j])] = {
                't': t_stat,
                'p_uncorrected': p_val,
                'mean_diff': np.mean(data[:, i] - data[:, j]),
                'cohen_d': np.mean(data[:, i] - data[:, j]) / np.std(data[:, i] - data[:, j], ddof=1)
            }

    # Apply correction
    p_values = np.array(p_values)
    n_comparisons = len(p_values)

    if correction == 'bonferroni':
        corrected = np.minimum(p_values * n_comparisons, 1.0)
    elif correction == 'fdr':
        corrected, _ = fdr_correction(p_values)
    else:
        corrected = p_values

    # Update results
    for i, (pair) in enumerate(comparisons):
        results[pair]['p_corrected'] = corrected[i]
        results[pair]['significant'] = corrected[i] < 0.05

    return results


def compute_bayes_factor(
    accuracies: np.ndarray,
    chance_level: float = 0.5,
    prior_scale: float = 0.707
) -> float:
    """
    Compute Bayes Factor for one-sample test.

    BF10 > 3 indicates moderate evidence for H1.
    BF10 > 10 indicates strong evidence for H1.

    Parameters
    ----------
    accuracies : np.ndarray
        Per-subject accuracies
    chance_level : float
        Chance level (null hypothesis)
    prior_scale : float
        Scale of Cauchy prior (default: sqrt(2)/2)

    Returns
    -------
    float
        Bayes Factor (BF10)
    """
    if HAS_PINGOUIN:
        result = pg.ttest(accuracies, chance_level, alternative='greater')
        return result['BF10'].values[0]
    else:
        # Simple approximation using BIC
        n = len(accuracies)
        t_stat, _ = stats.ttest_1samp(accuracies, chance_level)

        # Approximate BF using Wagenmakers formula
        r2 = t_stat ** 2 / (t_stat ** 2 + n - 1)
        bf10 = np.sqrt((n - 1) / (2 * np.pi)) * np.exp(0.5 * n * r2)

        return bf10
