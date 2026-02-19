"""
Effect size calculations and confidence intervals for T-maze analysis.

Provides standardized effect sizes (Cohen's d, Hedges' g) and
bootstrap/Bayesian confidence intervals.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from scipy import stats


@dataclass
class EffectSizeResult:
    """Container for effect size results."""
    effect_size: float
    effect_name: str
    ci_lower: float
    ci_upper: float
    ci_method: str
    interpretation: str
    n: int
    metadata: Dict = field(default_factory=dict)

    def __repr__(self):
        return (f"EffectSizeResult({self.effect_name}={self.effect_size:.3f}, "
                f"95% CI=[{self.ci_lower:.3f}, {self.ci_upper:.3f}], "
                f"{self.interpretation})")

    @staticmethod
    def interpret_cohens_d(d: float) -> str:
        """Interpret Cohen's d using conventional thresholds."""
        d_abs = abs(d)
        if d_abs < 0.2:
            return "negligible"
        elif d_abs < 0.5:
            return "small"
        elif d_abs < 0.8:
            return "medium"
        else:
            return "large"


def cohens_d(
    group1: np.ndarray,
    group2: Optional[np.ndarray] = None,
    mu: float = 0.0,
    paired: bool = False,
    confidence: float = 0.95,
    bootstrap_ci: bool = False,
    n_bootstrap: int = 10000
) -> EffectSizeResult:
    """
    Calculate Cohen's d effect size.

    For one-sample: d = (mean - mu) / sd
    For two-sample: d = (mean1 - mean2) / sd_pooled
    For paired: d = mean_diff / sd_diff

    Parameters
    ----------
    group1 : np.ndarray
        First group (or single group for one-sample)
    group2 : np.ndarray, optional
        Second group for two-sample test
    mu : float
        Population mean for one-sample test (default: 0)
    paired : bool
        Whether samples are paired
    confidence : float
        Confidence level (default: 0.95)
    bootstrap_ci : bool
        Use bootstrap for CI (otherwise use analytic)
    n_bootstrap : int
        Number of bootstrap resamples

    Returns
    -------
    EffectSizeResult
    """
    group1 = np.asarray(group1)
    n1 = len(group1)

    if group2 is None:
        # One-sample Cohen's d
        mean_diff = np.mean(group1) - mu
        sd = np.std(group1, ddof=1)
        d = mean_diff / sd

        # Analytic CI using non-central t distribution
        if not bootstrap_ci:
            se_d = np.sqrt(1/n1 + d**2 / (2*n1))
            z_crit = stats.norm.ppf((1 + confidence) / 2)
            ci_lower = d - z_crit * se_d
            ci_upper = d + z_crit * se_d
        else:
            ci_lower, ci_upper = _bootstrap_effect_ci(
                group1, None, mu, paired, n_bootstrap, confidence
            )

        return EffectSizeResult(
            effect_size=d,
            effect_name="Cohen's d (one-sample)",
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            ci_method="bootstrap" if bootstrap_ci else "analytic",
            interpretation=EffectSizeResult.interpret_cohens_d(d),
            n=n1,
            metadata={'mu': mu}
        )

    group2 = np.asarray(group2)
    n2 = len(group2)

    if paired:
        if n1 != n2:
            raise ValueError("Paired samples must have same length")

        # Paired Cohen's d (using difference scores)
        diff = group1 - group2
        d = np.mean(diff) / np.std(diff, ddof=1)

        if not bootstrap_ci:
            se_d = np.sqrt(1/n1 + d**2 / (2*n1))
            z_crit = stats.norm.ppf((1 + confidence) / 2)
            ci_lower = d - z_crit * se_d
            ci_upper = d + z_crit * se_d
        else:
            ci_lower, ci_upper = _bootstrap_effect_ci(
                group1, group2, mu, paired, n_bootstrap, confidence
            )

        return EffectSizeResult(
            effect_size=d,
            effect_name="Cohen's d (paired)",
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            ci_method="bootstrap" if bootstrap_ci else "analytic",
            interpretation=EffectSizeResult.interpret_cohens_d(d),
            n=n1,
            metadata={'paired': True}
        )

    # Two independent samples
    mean_diff = np.mean(group1) - np.mean(group2)

    # Pooled standard deviation
    var1 = np.var(group1, ddof=1)
    var2 = np.var(group2, ddof=1)
    sd_pooled = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

    d = mean_diff / sd_pooled

    if not bootstrap_ci:
        # Analytic SE
        se_d = np.sqrt((n1 + n2) / (n1 * n2) + d**2 / (2 * (n1 + n2)))
        z_crit = stats.norm.ppf((1 + confidence) / 2)
        ci_lower = d - z_crit * se_d
        ci_upper = d + z_crit * se_d
    else:
        ci_lower, ci_upper = _bootstrap_effect_ci(
            group1, group2, mu, paired, n_bootstrap, confidence
        )

    return EffectSizeResult(
        effect_size=d,
        effect_name="Cohen's d (independent)",
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        ci_method="bootstrap" if bootstrap_ci else "analytic",
        interpretation=EffectSizeResult.interpret_cohens_d(d),
        n=n1 + n2,
        metadata={'n1': n1, 'n2': n2}
    )


def hedges_g(
    group1: np.ndarray,
    group2: Optional[np.ndarray] = None,
    mu: float = 0.0,
    paired: bool = False,
    confidence: float = 0.95
) -> EffectSizeResult:
    """
    Calculate Hedges' g (bias-corrected Cohen's d).

    Hedges' g applies a correction factor for small samples.
    Use when n < 20 per group.

    Parameters
    ----------
    group1 : np.ndarray
        First group
    group2 : np.ndarray, optional
        Second group
    mu : float
        Population mean for one-sample test
    paired : bool
        Whether samples are paired
    confidence : float
        Confidence level

    Returns
    -------
    EffectSizeResult
    """
    # First compute Cohen's d
    d_result = cohens_d(group1, group2, mu, paired, confidence)
    d = d_result.effect_size

    # Correction factor (Hedges & Olkin, 1985)
    if group2 is None:
        n = len(group1)
        df = n - 1
    elif paired:
        n = len(group1)
        df = n - 1
    else:
        n = len(group1) + len(group2)
        df = n - 2

    # Correction factor J
    j = 1 - (3 / (4 * df - 1))
    g = d * j

    # Adjusted CI
    ci_lower = d_result.ci_lower * j
    ci_upper = d_result.ci_upper * j

    return EffectSizeResult(
        effect_size=g,
        effect_name="Hedges' g",
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        ci_method="analytic (corrected)",
        interpretation=EffectSizeResult.interpret_cohens_d(g),
        n=n,
        metadata={'correction_factor': j, 'cohens_d': d}
    )


def bootstrap_ci(
    data: np.ndarray,
    statistic: callable = np.mean,
    n_bootstrap: int = 10000,
    confidence: float = 0.95,
    method: str = 'percentile',
    random_state: Optional[int] = None
) -> Tuple[float, float, np.ndarray]:
    """
    Bootstrap confidence interval for any statistic.

    Parameters
    ----------
    data : np.ndarray
        Data to resample
    statistic : callable
        Function to compute statistic (default: mean)
    n_bootstrap : int
        Number of bootstrap samples
    confidence : float
        Confidence level
    method : str
        'percentile', 'bca', or 'basic'
    random_state : int, optional
        Random seed

    Returns
    -------
    ci_lower : float
        Lower CI bound
    ci_upper : float
        Upper CI bound
    bootstrap_distribution : np.ndarray
        Full bootstrap distribution
    """
    if random_state is not None:
        np.random.seed(random_state)

    n = len(data)
    bootstrap_stats = np.zeros(n_bootstrap)

    for i in range(n_bootstrap):
        sample = data[np.random.randint(0, n, size=n)]
        bootstrap_stats[i] = statistic(sample)

    alpha = 1 - confidence

    if method == 'percentile':
        ci_lower = np.percentile(bootstrap_stats, 100 * alpha / 2)
        ci_upper = np.percentile(bootstrap_stats, 100 * (1 - alpha / 2))

    elif method == 'basic':
        # Basic bootstrap (reverse percentile)
        theta_hat = statistic(data)
        q_lower = np.percentile(bootstrap_stats, 100 * alpha / 2)
        q_upper = np.percentile(bootstrap_stats, 100 * (1 - alpha / 2))
        ci_lower = 2 * theta_hat - q_upper
        ci_upper = 2 * theta_hat - q_lower

    elif method == 'bca':
        # BCa (Bias-Corrected and Accelerated)
        theta_hat = statistic(data)

        # Bias correction
        z0 = stats.norm.ppf(np.mean(bootstrap_stats < theta_hat))

        # Acceleration (jackknife)
        jackknife_stats = np.zeros(n)
        for i in range(n):
            jack_sample = np.delete(data, i)
            jackknife_stats[i] = statistic(jack_sample)

        jack_mean = np.mean(jackknife_stats)
        num = np.sum((jack_mean - jackknife_stats) ** 3)
        den = 6 * (np.sum((jack_mean - jackknife_stats) ** 2) ** 1.5)
        a = num / den if den != 0 else 0

        # Adjusted quantiles
        z_alpha_low = stats.norm.ppf(alpha / 2)
        z_alpha_high = stats.norm.ppf(1 - alpha / 2)

        alpha_low = stats.norm.cdf(z0 + (z0 + z_alpha_low) / (1 - a * (z0 + z_alpha_low)))
        alpha_high = stats.norm.cdf(z0 + (z0 + z_alpha_high) / (1 - a * (z0 + z_alpha_high)))

        ci_lower = np.percentile(bootstrap_stats, 100 * alpha_low)
        ci_upper = np.percentile(bootstrap_stats, 100 * alpha_high)

    else:
        raise ValueError(f"Unknown method: {method}")

    return ci_lower, ci_upper, bootstrap_stats


def bayesian_estimation(
    data: np.ndarray,
    prior_mu: float = 0.0,
    prior_sigma: float = 1.0,
    n_samples: int = 10000,
    credible_interval: float = 0.95
) -> Dict:
    """
    Bayesian estimation of mean with normal prior.

    Uses conjugate prior for normal mean with known variance.

    Parameters
    ----------
    data : np.ndarray
        Observed data
    prior_mu : float
        Prior mean
    prior_sigma : float
        Prior standard deviation
    n_samples : int
        Number of posterior samples
    credible_interval : float
        Credible interval width

    Returns
    -------
    Dict
        Posterior mean, SD, credible interval, samples
    """
    n = len(data)
    data_mean = np.mean(data)
    data_var = np.var(data, ddof=1)

    # Posterior parameters (conjugate update)
    prior_precision = 1 / prior_sigma ** 2
    data_precision = n / data_var

    posterior_precision = prior_precision + data_precision
    posterior_var = 1 / posterior_precision
    posterior_mean = (prior_precision * prior_mu + data_precision * data_mean) / posterior_precision
    posterior_sd = np.sqrt(posterior_var)

    # Sample from posterior
    posterior_samples = np.random.normal(posterior_mean, posterior_sd, n_samples)

    # Credible interval
    alpha = 1 - credible_interval
    ci_lower = np.percentile(posterior_samples, 100 * alpha / 2)
    ci_upper = np.percentile(posterior_samples, 100 * (1 - alpha / 2))

    # Probability of being greater than chance (0.5 for classification)
    prob_above_chance = np.mean(posterior_samples > 0.5)

    # Bayes Factor approximation (Savage-Dickey ratio)
    prior_at_null = stats.norm.pdf(prior_mu, prior_mu, prior_sigma)
    posterior_at_null = stats.norm.pdf(prior_mu, posterior_mean, posterior_sd)
    bf10 = prior_at_null / posterior_at_null if posterior_at_null > 0 else np.inf

    return {
        'posterior_mean': posterior_mean,
        'posterior_sd': posterior_sd,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'credible_interval': credible_interval,
        'prob_above_chance': prob_above_chance,
        'bayes_factor_10': bf10,
        'posterior_samples': posterior_samples,
        'prior_mu': prior_mu,
        'prior_sigma': prior_sigma
    }


def _bootstrap_effect_ci(
    group1: np.ndarray,
    group2: Optional[np.ndarray],
    mu: float,
    paired: bool,
    n_bootstrap: int,
    confidence: float
) -> Tuple[float, float]:
    """Internal function for bootstrapped effect size CI."""
    n1 = len(group1)
    boot_effects = np.zeros(n_bootstrap)

    for i in range(n_bootstrap):
        idx1 = np.random.randint(0, n1, size=n1)
        sample1 = group1[idx1]

        if group2 is None:
            # One-sample
            d = (np.mean(sample1) - mu) / np.std(sample1, ddof=1)
        elif paired:
            sample2 = group2[idx1]  # Same indices for paired
            diff = sample1 - sample2
            d = np.mean(diff) / np.std(diff, ddof=1)
        else:
            n2 = len(group2)
            idx2 = np.random.randint(0, n2, size=n2)
            sample2 = group2[idx2]
            var1 = np.var(sample1, ddof=1)
            var2 = np.var(sample2, ddof=1)
            sd_pooled = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1 + n2 - 2))
            d = (np.mean(sample1) - np.mean(sample2)) / sd_pooled

        boot_effects[i] = d

    alpha = 1 - confidence
    ci_lower = np.percentile(boot_effects, 100 * alpha / 2)
    ci_upper = np.percentile(boot_effects, 100 * (1 - alpha / 2))

    return ci_lower, ci_upper


def effect_size_classification(
    accuracies: np.ndarray,
    chance_level: float = 0.5
) -> EffectSizeResult:
    """
    Compute effect size for classification accuracy vs chance.

    Convenience function for T-maze classification analysis.

    Parameters
    ----------
    accuracies : np.ndarray
        Per-subject classification accuracies
    chance_level : float
        Chance level (default: 0.5 for binary)

    Returns
    -------
    EffectSizeResult
    """
    return cohens_d(accuracies, mu=chance_level, bootstrap_ci=True)


def common_language_effect_size(d: float) -> float:
    """
    Convert Cohen's d to common language effect size (CLES).

    CLES is the probability that a randomly selected value from
    group 1 is greater than a randomly selected value from group 2.

    Parameters
    ----------
    d : float
        Cohen's d

    Returns
    -------
    float
        CLES probability (0-1)
    """
    return stats.norm.cdf(d / np.sqrt(2))


def odds_ratio_from_d(d: float) -> float:
    """
    Convert Cohen's d to odds ratio.

    Useful for comparing to logistic regression results.

    Parameters
    ----------
    d : float
        Cohen's d

    Returns
    -------
    float
        Odds ratio
    """
    # Using logistic distribution approximation
    return np.exp(d * np.pi / np.sqrt(3))
