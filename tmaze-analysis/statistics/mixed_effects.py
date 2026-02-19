"""
Linear Mixed Effects models for T-maze group analysis.

LME models account for within-subject correlation by treating
subject as a random effect.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
import warnings

try:
    import statsmodels.api as sm
    from statsmodels.formula.api import mixedlm
    import pandas as pd
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False


@dataclass
class LMEResult:
    """Container for Linear Mixed Effects results."""
    coefficient: float
    std_error: float
    z_value: float
    p_value: float
    ci_lower: float
    ci_upper: float
    random_effect_var: float
    residual_var: float
    icc: float  # Intraclass correlation
    n_subjects: int
    n_observations: int
    converged: bool
    aic: Optional[float] = None
    bic: Optional[float] = None
    log_likelihood: Optional[float] = None
    metadata: Dict = field(default_factory=dict)

    def __repr__(self):
        sig = "*" if self.p_value < 0.05 else ""
        return (f"LMEResult(coef={self.coefficient:.4f}, "
                f"SE={self.std_error:.4f}, z={self.z_value:.2f}, "
                f"p={self.p_value:.4f}{sig}, ICC={self.icc:.3f})")

    def is_significant(self, alpha: float = 0.05) -> bool:
        return self.p_value < alpha


class LinearMixedEffects:
    """
    Linear Mixed Effects model for T-maze group analysis.

    Fits LME with subject as random intercept to account for
    within-subject correlation when analyzing classification accuracy.

    Parameters
    ----------
    random_slopes : bool
        Include random slopes for condition (default: False)
    reml : bool
        Use REML estimation (default: True)

    Examples
    --------
    >>> lme = LinearMixedEffects()
    >>> lme.fit(accuracies, subjects, conditions)
    >>> print(lme.summary())
    """

    def __init__(
        self,
        random_slopes: bool = False,
        reml: bool = True
    ):
        if not HAS_STATSMODELS:
            raise ImportError("statsmodels required for LME analysis")

        self.random_slopes = random_slopes
        self.reml = reml
        self.model_ = None
        self.results_ = None
        self.data_ = None

    def fit(
        self,
        y: np.ndarray,
        subjects: np.ndarray,
        condition: Optional[np.ndarray] = None,
        covariates: Optional[Dict[str, np.ndarray]] = None
    ) -> 'LinearMixedEffects':
        """
        Fit the LME model.

        Parameters
        ----------
        y : np.ndarray
            Dependent variable (e.g., accuracy per trial)
        subjects : np.ndarray
            Subject identifiers for each observation
        condition : np.ndarray, optional
            Condition labels (fixed effect)
        covariates : Dict[str, np.ndarray], optional
            Additional covariates

        Returns
        -------
        self
        """
        # Build DataFrame
        data = {'y': y, 'subject': subjects}

        formula = 'y ~ 1'  # Intercept-only model if no condition

        if condition is not None:
            data['condition'] = condition
            formula = 'y ~ condition'

        if covariates:
            for name, values in covariates.items():
                data[name] = values
                formula += f' + {name}'

        df = pd.DataFrame(data)
        self.data_ = df

        # Random effects specification
        if self.random_slopes and condition is not None:
            # Random intercept and slope
            re_formula = '~condition'
        else:
            # Random intercept only
            re_formula = None

        # Fit model
        self.model_ = mixedlm(formula, df, groups=df['subject'],
                              re_formula=re_formula)

        try:
            self.results_ = self.model_.fit(reml=self.reml)
        except Exception as e:
            warnings.warn(f"Model did not converge: {e}")
            self.results_ = None

        return self

    def get_result(self, effect_name: str = 'condition') -> LMEResult:
        """
        Get results for a specific effect.

        Parameters
        ----------
        effect_name : str
            Name of the fixed effect

        Returns
        -------
        LMEResult
        """
        if self.results_ is None:
            raise ValueError("Model not fitted or did not converge")

        res = self.results_

        # Find the coefficient
        if effect_name == 'intercept':
            coef_name = 'Intercept'
        else:
            # Find matching coefficient name
            matching = [n for n in res.params.index if effect_name.lower() in n.lower()]
            if not matching:
                raise ValueError(f"Effect '{effect_name}' not found in model")
            coef_name = matching[0]

        coef = res.params[coef_name]
        se = res.bse[coef_name]
        z = res.tvalues[coef_name]
        p = res.pvalues[coef_name]

        # Confidence interval
        ci = res.conf_int()
        ci_lower = ci.loc[coef_name, 0]
        ci_upper = ci.loc[coef_name, 1]

        # Random effects variance
        random_var = float(res.cov_re.iloc[0, 0]) if hasattr(res, 'cov_re') else 0
        residual_var = res.scale

        # Intraclass correlation (ICC)
        icc = random_var / (random_var + residual_var) if (random_var + residual_var) > 0 else 0

        return LMEResult(
            coefficient=coef,
            std_error=se,
            z_value=z,
            p_value=p,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            random_effect_var=random_var,
            residual_var=residual_var,
            icc=icc,
            n_subjects=len(self.data_['subject'].unique()),
            n_observations=len(self.data_),
            converged=self.results_ is not None,
            aic=res.aic,
            bic=res.bic,
            log_likelihood=res.llf,
            metadata={'effect': effect_name, 'formula': str(self.model_.formula)}
        )

    def summary(self) -> str:
        """Get text summary of model results."""
        if self.results_ is None:
            return "Model not fitted or did not converge"
        return str(self.results_.summary())

    def predict(self, data: Optional[pd.DataFrame] = None) -> np.ndarray:
        """Get predicted values."""
        if self.results_ is None:
            raise ValueError("Model not fitted")
        if data is None:
            return self.results_.fittedvalues.values
        return self.results_.predict(data)

    def random_effects(self) -> pd.DataFrame:
        """Get estimated random effects per subject."""
        if self.results_ is None:
            raise ValueError("Model not fitted")
        return pd.DataFrame(self.results_.random_effects).T


def lme_roi_analysis(
    subject_roi_accuracies: Dict[str, Dict[str, float]],
    chance_level: float = 0.5,
    n_jobs: int = 1
) -> Dict[str, LMEResult]:
    """
    Run LME for each ROI to test group-level significance.

    Parameters
    ----------
    subject_roi_accuracies : Dict[str, Dict[str, float]]
        Nested dict: {subject_id: {roi_name: accuracy}}
    chance_level : float
        Chance level for comparison (not used in LME, for reference)
    n_jobs : int
        Parallel jobs (not yet implemented)

    Returns
    -------
    Dict[str, LMEResult]
        LME results for each ROI
    """
    if not HAS_STATSMODELS:
        raise ImportError("statsmodels required for LME analysis")

    # Get ROI names
    subjects = list(subject_roi_accuracies.keys())
    roi_names = list(subject_roi_accuracies[subjects[0]].keys())

    results = {}

    for roi in roi_names:
        # Collect data
        y_list = []
        subj_list = []

        for subj in subjects:
            if roi in subject_roi_accuracies[subj]:
                y_list.append(subject_roi_accuracies[subj][roi])
                subj_list.append(subj)

        if len(y_list) < 3:
            continue

        y = np.array(y_list)
        subjects_arr = np.array(subj_list)

        # Fit intercept-only model (tests if mean != 0)
        # For testing vs chance, we can center the data
        y_centered = y - chance_level

        try:
            lme = LinearMixedEffects()
            lme.fit(y_centered, subjects_arr)
            result = lme.get_result('intercept')
            result.metadata['roi'] = roi
            result.metadata['chance_level'] = chance_level
            results[roi] = result
        except Exception as e:
            warnings.warn(f"LME failed for ROI {roi}: {e}")

    return results


def lme_temporal_analysis(
    subject_temporal_accuracies: np.ndarray,
    times: np.ndarray,
    subject_ids: Optional[List[str]] = None,
    chance_level: float = 0.5
) -> Dict[str, Any]:
    """
    Run LME across time points for temporal decoding.

    Tests at which time points accuracy is significantly above chance
    using LME to account for between-subject variability.

    Parameters
    ----------
    subject_temporal_accuracies : np.ndarray
        Accuracy matrix (n_subjects, n_times)
    times : np.ndarray
        Time vector
    subject_ids : List[str], optional
        Subject identifiers
    chance_level : float
        Chance level for centering

    Returns
    -------
    Dict
        Results including p-values, significant clusters, etc.
    """
    if not HAS_STATSMODELS:
        raise ImportError("statsmodels required for LME analysis")

    n_subjects, n_times = subject_temporal_accuracies.shape

    if subject_ids is None:
        subject_ids = [f'S{i:02d}' for i in range(n_subjects)]

    # Store results per time point
    coefficients = np.zeros(n_times)
    std_errors = np.zeros(n_times)
    z_values = np.zeros(n_times)
    p_values = np.zeros(n_times)
    iccs = np.zeros(n_times)

    for t_idx in range(n_times):
        y = subject_temporal_accuracies[:, t_idx] - chance_level
        subjects = np.array(subject_ids)

        try:
            lme = LinearMixedEffects()
            lme.fit(y, subjects)
            result = lme.get_result('intercept')

            coefficients[t_idx] = result.coefficient
            std_errors[t_idx] = result.std_error
            z_values[t_idx] = result.z_value
            p_values[t_idx] = result.p_value
            iccs[t_idx] = result.icc
        except Exception:
            coefficients[t_idx] = np.nan
            std_errors[t_idx] = np.nan
            z_values[t_idx] = np.nan
            p_values[t_idx] = 1.0
            iccs[t_idx] = np.nan

    # Find significant clusters (FDR correction)
    from .group_stats import fdr_correction
    _, significant = fdr_correction(p_values, alpha=0.05)

    # Identify contiguous significant clusters
    clusters = []
    in_cluster = False
    cluster_start = None

    for i, sig in enumerate(significant):
        if sig and not in_cluster:
            in_cluster = True
            cluster_start = i
        elif not sig and in_cluster:
            in_cluster = False
            clusters.append((times[cluster_start], times[i-1]))

    if in_cluster:
        clusters.append((times[cluster_start], times[-1]))

    return {
        'times': times,
        'coefficients': coefficients,
        'std_errors': std_errors,
        'z_values': z_values,
        'p_values': p_values,
        'p_values_fdr': fdr_correction(p_values)[0],
        'significant': significant,
        'significant_clusters': clusters,
        'iccs': iccs,
        'mean_icc': np.nanmean(iccs),
        'n_subjects': n_subjects,
        'chance_level': chance_level
    }


def lme_condition_contrast(
    data: np.ndarray,
    subjects: np.ndarray,
    conditions: np.ndarray,
    contrast: Optional[List[float]] = None
) -> LMEResult:
    """
    Test condition effect using LME with custom contrast.

    Parameters
    ----------
    data : np.ndarray
        Dependent variable (n_observations,)
    subjects : np.ndarray
        Subject IDs (n_observations,)
    conditions : np.ndarray
        Condition labels (n_observations,)
    contrast : List[float], optional
        Custom contrast weights

    Returns
    -------
    LMEResult
    """
    if not HAS_STATSMODELS:
        raise ImportError("statsmodels required")

    lme = LinearMixedEffects()
    lme.fit(data, subjects, conditions)

    # Get the condition effect
    return lme.get_result('condition')


def compare_models(
    y: np.ndarray,
    subjects: np.ndarray,
    condition: Optional[np.ndarray] = None,
    covariates: Optional[Dict[str, np.ndarray]] = None
) -> Dict[str, Any]:
    """
    Compare nested LME models using likelihood ratio test.

    Compares model with condition effect vs intercept-only.

    Parameters
    ----------
    y : np.ndarray
        Dependent variable
    subjects : np.ndarray
        Subject IDs
    condition : np.ndarray, optional
        Condition labels
    covariates : Dict[str, np.ndarray], optional
        Additional covariates

    Returns
    -------
    Dict
        Model comparison results including LRT statistic and p-value
    """
    if not HAS_STATSMODELS:
        raise ImportError("statsmodels required")

    # Fit null model (intercept only)
    lme_null = LinearMixedEffects(reml=False)  # Use ML for LRT
    lme_null.fit(y, subjects)

    # Fit full model (with condition)
    lme_full = LinearMixedEffects(reml=False)
    lme_full.fit(y, subjects, condition, covariates)

    if lme_null.results_ is None or lme_full.results_ is None:
        return {'error': 'One or both models failed to converge'}

    # Likelihood ratio test
    ll_null = lme_null.results_.llf
    ll_full = lme_full.results_.llf

    # LRT statistic (chi-squared distributed)
    lrt = -2 * (ll_null - ll_full)
    df_diff = len(lme_full.results_.params) - len(lme_null.results_.params)

    from scipy import stats as sp_stats
    p_value = 1 - sp_stats.chi2.cdf(lrt, df_diff)

    return {
        'lrt_statistic': lrt,
        'df': df_diff,
        'p_value': p_value,
        'aic_null': lme_null.results_.aic,
        'aic_full': lme_full.results_.aic,
        'bic_null': lme_null.results_.bic,
        'bic_full': lme_full.results_.bic,
        'prefer_full': p_value < 0.05
    }
