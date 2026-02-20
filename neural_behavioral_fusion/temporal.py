"""
Time-resolved LME analysis at each EEG timepoint.
"""

from typing import Optional

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

from .core import FusionLME


def temporal_lme(
    df_wide,
    times,
    formula_template="amp ~ rpe",
    re_formula="~rpe",
    groups="subject",
    reml=True,
    rpe_predictor="rpe",
    fdr_alpha=0.05,
    amp_prefix="amp_t",
):
    """Fit LME at each EEG timepoint to get a time course of RPE effects.

    Parameters
    ----------
    df_wide : pd.DataFrame
        Wide-format data with columns: subject, rpe, and one amplitude column
        per timepoint named '{amp_prefix}{i}' (e.g., amp_t0, amp_t1, ...).
    times : array-like
        Time vector in seconds (length must match number of amp columns).
    formula_template : str
        Formula with 'amp' as placeholder for the DV.
    re_formula : str or None
        Random effects formula.
    groups : str
        Grouping variable.
    reml : bool
        Use REML estimation.
    rpe_predictor : str
        Predictor name for extracting coefficients.
    fdr_alpha : float
        Alpha level for FDR correction.
    amp_prefix : str
        Prefix for amplitude columns.

    Returns
    -------
    dict
        Keys:
        - times: time vector
        - coefficients: RPE coefficient at each timepoint
        - std_errors: SE at each timepoint
        - pvalues: uncorrected p-values
        - pvalues_fdr: FDR-corrected p-values
        - significant: boolean array of FDR-significant timepoints
        - z_values: z-statistics at each timepoint
    """
    times = np.asarray(times)
    n_times = len(times)

    coefficients = np.zeros(n_times)
    std_errors = np.zeros(n_times)
    z_values = np.zeros(n_times)
    pvalues = np.ones(n_times)

    lme = FusionLME(reml=reml)

    for i in range(n_times):
        col_name = f"{amp_prefix}{i}"
        if col_name not in df_wide.columns:
            continue

        # Build formula with actual column name
        t_formula = formula_template.replace("amp", col_name)

        try:
            result = lme.fit(df_wide, t_formula, re_formula=re_formula, groups=groups)

            # Extract RPE coefficient
            rpe_key = _find_rpe_key(result.coefficients, rpe_predictor)
            if rpe_key:
                coefficients[i] = result.coefficients[rpe_key]["estimate"]
                std_errors[i] = result.coefficients[rpe_key]["se"]
                z_values[i] = result.coefficients[rpe_key]["z"]
                pvalues[i] = result.coefficients[rpe_key]["p"]
        except Exception:
            coefficients[i] = np.nan
            std_errors[i] = np.nan
            z_values[i] = np.nan
            pvalues[i] = 1.0

    # FDR correction
    valid = ~np.isnan(pvalues)
    pvalues_fdr = np.ones(n_times)
    significant = np.zeros(n_times, dtype=bool)

    if valid.sum() > 0:
        sig_valid, fdr_valid, _, _ = multipletests(
            pvalues[valid], alpha=fdr_alpha, method="fdr_bh"
        )
        pvalues_fdr[valid] = fdr_valid
        significant[valid] = sig_valid

    return {
        "times": times,
        "coefficients": coefficients,
        "std_errors": std_errors,
        "z_values": z_values,
        "pvalues": pvalues,
        "pvalues_fdr": pvalues_fdr,
        "significant": significant,
    }


def _find_rpe_key(coefficients, rpe_predictor):
    """Find the RPE coefficient key in the coefficients dict."""
    if rpe_predictor in coefficients:
        return rpe_predictor
    for key in coefficients:
        if rpe_predictor.lower() in key.lower() and key != "Intercept":
            return key
    return None
