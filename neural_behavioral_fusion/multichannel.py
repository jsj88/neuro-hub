"""
Multi-channel LME analysis with FDR correction.
"""

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

from .core import FusionLME, FusionLMEResult


def multichannel_lme(
    df,
    channels,
    formula="rewp ~ rpe",
    re_formula="~rpe",
    groups="subject",
    reml=True,
    rpe_predictor="rpe",
    fdr_alpha=0.05,
):
    """Fit LME separately per EEG channel with FDR correction.

    Parameters
    ----------
    df : pd.DataFrame
        Data with subject, rpe, and channel amplitude columns.
    channels : list of str
        Column names for each channel's amplitude.
    formula : str
        Fixed-effects formula. 'rewp' will be replaced with each channel name.
    re_formula : str or None
        Random effects formula.
    groups : str
        Grouping variable.
    reml : bool
        Use REML estimation.
    rpe_predictor : str
        Name of the RPE predictor in the formula, for extracting p-values.
    fdr_alpha : float
        Alpha level for FDR correction.

    Returns
    -------
    dict
        Keys:
        - channel_results: {channel_name: FusionLMEResult}
        - pvalues: array of RPE p-values per channel
        - pvalues_fdr: FDR-corrected p-values
        - significant: boolean array of FDR-significant channels
        - coefficients: array of RPE coefficients per channel
    """
    lme = FusionLME(reml=reml)
    channel_results = {}
    pvals = []
    coefs = []

    for ch in channels:
        if ch not in df.columns:
            raise ValueError(f"Channel '{ch}' not found in DataFrame columns")

        # Replace 'rewp' in formula with the channel column name
        ch_formula = formula.replace("rewp", ch)

        try:
            result = lme.fit(df, ch_formula, re_formula=re_formula, groups=groups)
            channel_results[ch] = result

            # Extract RPE coefficient p-value
            rpe_key = _find_rpe_key(result.coefficients, rpe_predictor)
            if rpe_key:
                pvals.append(result.coefficients[rpe_key]["p"])
                coefs.append(result.coefficients[rpe_key]["estimate"])
            else:
                pvals.append(1.0)
                coefs.append(0.0)
        except Exception as e:
            print(f"  Warning: LME failed for channel {ch}: {e}")
            pvals.append(1.0)
            coefs.append(0.0)

    pvals = np.array(pvals)
    coefs = np.array(coefs)

    # FDR correction (Benjamini-Hochberg)
    if len(pvals) > 0:
        significant, pvals_fdr, _, _ = multipletests(pvals, alpha=fdr_alpha, method="fdr_bh")
    else:
        pvals_fdr = np.array([])
        significant = np.array([], dtype=bool)

    return {
        "channel_results": channel_results,
        "pvalues": pvals,
        "pvalues_fdr": pvals_fdr,
        "significant": significant,
        "coefficients": coefs,
        "channels": channels,
    }


def _find_rpe_key(coefficients, rpe_predictor):
    """Find the RPE coefficient key in the coefficients dict."""
    if rpe_predictor in coefficients:
        return rpe_predictor
    # Try case-insensitive match
    for key in coefficients:
        if rpe_predictor.lower() in key.lower() and key != "Intercept":
            return key
    return None
