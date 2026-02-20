"""
FusionLME engine: fit linear mixed-effects models for RPE-REWP fusion.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional
import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from .models import get_model_spec


@dataclass
class FusionLMEResult:
    """Container for a fitted LME model result.

    Attributes
    ----------
    coefficients : dict
        Fixed effects: {predictor: {estimate, se, z, p, ci_lower, ci_upper}}.
    random_effects : dict
        Variance components from the random effects covariance matrix.
    model_fit : dict
        AIC, BIC, log_likelihood, converged.
    n_subjects : int
        Number of unique groups (subjects).
    n_observations : int
        Total number of observations.
    formula : str
        The formula used for fixed effects.
    re_formula : str or None
        The formula used for random effects.
    summary_text : str
        Full statsmodels summary output.
    """

    coefficients: Dict = field(default_factory=dict)
    random_effects: Dict = field(default_factory=dict)
    model_fit: Dict = field(default_factory=dict)
    n_subjects: int = 0
    n_observations: int = 0
    formula: str = ""
    re_formula: Optional[str] = None
    summary_text: str = ""

    def __repr__(self):
        sig_effects = [
            k for k, v in self.coefficients.items()
            if k != "Intercept" and v["p"] < 0.05
        ]
        status = "converged" if self.model_fit.get("converged", False) else "NOT converged"
        return (
            f"FusionLMEResult({status}, n_subj={self.n_subjects}, "
            f"n_obs={self.n_observations}, "
            f"sig_effects={sig_effects or 'none'})"
        )


def _extract_result(model_result, formula, re_formula, df, groups_col):
    """Extract FusionLMEResult from a fitted statsmodels MixedLMResults."""
    res = model_result

    # Fixed effects
    coefficients = {}
    ci = res.conf_int()
    for name in res.params.index:
        coefficients[name] = {
            "estimate": float(res.params[name]),
            "se": float(res.bse[name]),
            "z": float(res.tvalues[name]),
            "p": float(res.pvalues[name]),
            "ci_lower": float(ci.loc[name, 0]),
            "ci_upper": float(ci.loc[name, 1]),
        }

    # Random effects variance components
    random_effects = {}
    if hasattr(res, "cov_re") and res.cov_re is not None:
        cov_re = res.cov_re
        if hasattr(cov_re, "values"):
            for i, name in enumerate(cov_re.index):
                random_effects[name] = {
                    "variance": float(cov_re.iloc[i, i]),
                    "sd": float(np.sqrt(max(0, cov_re.iloc[i, i]))),
                }
        else:
            random_effects["Group"] = {
                "variance": float(cov_re),
                "sd": float(np.sqrt(max(0, float(cov_re)))),
            }
    random_effects["Residual"] = {
        "variance": float(res.scale),
        "sd": float(np.sqrt(max(0, res.scale))),
    }

    # Compute AIC/BIC — statsmodels sometimes returns NaN for these
    llf = float(res.llf) if hasattr(res, "llf") else None
    n_params = len(res.params)
    n_obs = len(df)

    aic = float(res.aic) if hasattr(res, "aic") and not np.isnan(res.aic) else None
    bic = float(res.bic) if hasattr(res, "bic") and not np.isnan(res.bic) else None

    # Fallback: compute from log-likelihood if statsmodels gives NaN
    if aic is None and llf is not None and not np.isnan(llf):
        aic = -2 * llf + 2 * n_params
    if bic is None and llf is not None and not np.isnan(llf):
        bic = -2 * llf + n_params * np.log(n_obs)

    model_fit = {
        "AIC": aic,
        "BIC": bic,
        "log_likelihood": llf,
        "converged": res.converged if hasattr(res, "converged") else True,
    }

    n_subjects = df[groups_col].nunique()
    n_observations = len(df)

    try:
        summary_text = str(res.summary())
    except Exception:
        summary_text = "Summary unavailable"

    return FusionLMEResult(
        coefficients=coefficients,
        random_effects=random_effects,
        model_fit=model_fit,
        n_subjects=n_subjects,
        n_observations=n_observations,
        formula=formula,
        re_formula=re_formula,
        summary_text=summary_text,
    )


class FusionLME:
    """Linear mixed-effects model engine for neural-behavioral fusion.

    Parameters
    ----------
    reml : bool
        Use REML estimation (default True). Set False for LRT comparison.

    Examples
    --------
    >>> lme = FusionLME()
    >>> result = lme.fit(df, "rewp ~ rpe", re_formula="~rpe", groups="subject")
    >>> print(result.coefficients["rpe"]["p"])
    """

    def __init__(self, reml=True):
        self.reml = reml
        self._last_model = None
        self._last_result = None

    def fit(self, df, formula, re_formula=None, groups="subject"):
        """Fit an LME model.

        Parameters
        ----------
        df : pd.DataFrame
            Data with columns referenced in formula and groups.
        formula : str
            Statsmodels formula for fixed effects (e.g., "rewp ~ rpe").
        re_formula : str or None
            Random effects formula (e.g., "~rpe" for random slope).
            None gives random intercept only.
        groups : str
            Column name for the grouping variable (default "subject").

        Returns
        -------
        FusionLMEResult
        """
        model = smf.mixedlm(formula, df, groups=df[groups], re_formula=re_formula)

        # Try multiple optimizers for robustness
        optimizers = ["powell", "lbfgs", "cg"]
        last_error = None

        for method in optimizers:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    result = model.fit(reml=self.reml, method=method)
                self._last_model = model
                self._last_result = result
                return _extract_result(result, formula, re_formula, df, groups)
            except Exception as e:
                last_error = e
                continue

        # All optimizers failed — fall back to simpler random effects
        if re_formula is not None:
            warnings.warn(
                f"Model with re_formula='{re_formula}' failed. "
                "Falling back to random intercept only."
            )
            try:
                model = smf.mixedlm(formula, df, groups=df[groups], re_formula=None)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    result = model.fit(reml=self.reml)
                self._last_model = model
                self._last_result = result
                return _extract_result(result, formula, None, df, groups)
            except Exception as e2:
                raise RuntimeError(
                    f"LME fit failed even with intercept-only RE: {e2}"
                ) from e2
        else:
            raise RuntimeError(f"LME fit failed with all optimizers: {last_error}") from last_error

    def fit_model(self, df, model_name, groups="subject"):
        """Fit a pre-defined model from the registry.

        Parameters
        ----------
        df : pd.DataFrame
            Data with required columns.
        model_name : str
            Name from MODEL_REGISTRY.
        groups : str
            Grouping variable column name.

        Returns
        -------
        FusionLMEResult
        """
        spec = get_model_spec(model_name)
        return self.fit(
            df,
            formula=spec["formula"],
            re_formula=spec["re_formula"],
            groups=groups,
        )

    def predict(self, df=None):
        """Get predicted values from the last fitted model."""
        if self._last_result is None:
            raise ValueError("No model has been fitted yet")
        if df is None:
            return self._last_result.fittedvalues.values
        return self._last_result.predict(df)

    def random_effects_df(self):
        """Get per-subject random effects as a DataFrame."""
        if self._last_result is None:
            raise ValueError("No model has been fitted yet")
        return pd.DataFrame(self._last_result.random_effects).T
