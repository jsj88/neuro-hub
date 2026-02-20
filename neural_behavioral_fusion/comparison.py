"""
Model comparison: AIC/BIC tables and likelihood ratio tests.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from .core import FusionLME, FusionLMEResult
from .models import get_model_spec


def compare_predictors(
    df,
    model_names,
    groups="subject",
):
    """Compare multiple pre-defined LME models by AIC/BIC.

    Fits each model from the registry with REML=True and collects
    fit statistics for comparison.

    Parameters
    ----------
    df : pd.DataFrame
        Data with required columns for all models.
    model_names : list of str
        Model names from MODEL_REGISTRY.
    groups : str
        Grouping variable column.

    Returns
    -------
    pd.DataFrame
        Comparison table sorted by BIC, with columns:
        model, formula, re_formula, AIC, BIC, log_likelihood,
        converged, n_params, rpe_coef, rpe_p.
    """
    rows = []
    lme = FusionLME(reml=True)

    for name in model_names:
        spec = get_model_spec(name)

        try:
            result = lme.fit(
                df,
                formula=spec["formula"],
                re_formula=spec["re_formula"],
                groups=groups,
            )

            # Extract RPE coefficient info
            rpe_coef = np.nan
            rpe_p = np.nan
            for key, val in result.coefficients.items():
                if "rpe" in key.lower() and key != "Intercept":
                    rpe_coef = val["estimate"]
                    rpe_p = val["p"]
                    break

            rows.append({
                "model": name,
                "formula": spec["formula"],
                "re_formula": spec["re_formula"] or "~1",
                "AIC": result.model_fit.get("AIC"),
                "BIC": result.model_fit.get("BIC"),
                "log_likelihood": result.model_fit.get("log_likelihood"),
                "converged": result.model_fit.get("converged", False),
                "rpe_coef": rpe_coef,
                "rpe_p": rpe_p,
                "description": spec["description"],
            })
        except Exception as e:
            rows.append({
                "model": name,
                "formula": spec["formula"],
                "re_formula": spec["re_formula"] or "~1",
                "AIC": np.nan,
                "BIC": np.nan,
                "log_likelihood": np.nan,
                "converged": False,
                "rpe_coef": np.nan,
                "rpe_p": np.nan,
                "description": f"FAILED: {e}",
            })

    comp = pd.DataFrame(rows)
    comp = comp.sort_values("BIC", ascending=True).reset_index(drop=True)
    return comp


def likelihood_ratio_test(result_restricted, result_full, df_diff=1):
    """Likelihood ratio test between nested LME models.

    Both models must be fit with REML=False for valid comparison.

    Parameters
    ----------
    result_restricted : FusionLMEResult
        The restricted (simpler) model result.
    result_full : FusionLMEResult
        The full (more complex) model result.
    df_diff : int
        Difference in number of parameters between models.

    Returns
    -------
    dict
        chi2: test statistic
        df: degrees of freedom
        p: p-value
        prefer_full: bool
    """
    ll_restricted = result_restricted.model_fit.get("log_likelihood")
    ll_full = result_full.model_fit.get("log_likelihood")

    if ll_restricted is None or ll_full is None:
        raise ValueError("Log-likelihood not available. Ensure models are fitted.")

    chi2 = -2 * (ll_restricted - ll_full)
    chi2 = max(0, chi2)  # Ensure non-negative
    p = 1 - sp_stats.chi2.cdf(chi2, df_diff)

    return {
        "chi2": float(chi2),
        "df": df_diff,
        "p": float(p),
        "prefer_full": p < 0.05,
        "ll_restricted": float(ll_restricted),
        "ll_full": float(ll_full),
    }
