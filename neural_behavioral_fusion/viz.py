"""
Publication-quality visualizations for neural-behavioral fusion LME results.
"""

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats as sp_stats

from .core import FusionLMEResult


def plot_rpe_rewp_scatter(df, result, ax=None, rpe_col="rpe", rewp_col="rewp",
                          subject_col="subject"):
    """Scatter plot of RPE vs REWP with per-subject and group regression lines.

    Parameters
    ----------
    df : pd.DataFrame
        Data with rpe, rewp, and subject columns.
    result : FusionLMEResult
        Fitted LME result for annotation.
    ax : matplotlib Axes or None
        Axes to plot on.
    rpe_col, rewp_col, subject_col : str
        Column names.

    Returns
    -------
    matplotlib Axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    subjects = df[subject_col].unique()
    colors = plt.cm.tab20(np.linspace(0, 1, len(subjects)))

    # Per-subject regression lines (thin)
    for i, subj in enumerate(subjects):
        mask = df[subject_col] == subj
        x = df.loc[mask, rpe_col].values
        y = df.loc[mask, rewp_col].values
        ax.scatter(x, y, alpha=0.15, s=10, color=colors[i])

        if len(x) > 2:
            z = np.polyfit(x, y, 1)
            xline = np.linspace(x.min(), x.max(), 50)
            ax.plot(xline, np.polyval(z, xline), color=colors[i],
                    alpha=0.4, linewidth=0.8)

    # Group-level regression line (thick)
    x_all = df[rpe_col].values
    y_all = df[rewp_col].values
    z_all = np.polyfit(x_all, y_all, 1)
    xline = np.linspace(x_all.min(), x_all.max(), 100)
    ax.plot(xline, np.polyval(z_all, xline), "k-", linewidth=3,
            label="Group LME fit")

    # Annotate with RPE coefficient
    rpe_key = _find_rpe_key(result.coefficients, "rpe")
    if rpe_key:
        coef = result.coefficients[rpe_key]
        sig = "*" if coef["p"] < 0.05 else ""
        ax.set_title(
            f"RPE-REWP Fusion (LME)\n"
            f"b = {coef['estimate']:.3f}, z = {coef['z']:.2f}, "
            f"p = {coef['p']:.4f}{sig}",
            fontsize=11,
        )

    ax.set_xlabel("Reward Prediction Error (RPE)")
    ax.set_ylabel("REWP Amplitude (uV)")
    ax.legend(loc="upper left", fontsize=9)
    return ax


def plot_random_effects(result_or_df, ax=None, predictor="rpe"):
    """Forest plot of per-subject random effects (slopes) with CIs.

    Parameters
    ----------
    result_or_df : pd.DataFrame
        DataFrame of random effects (from FusionLME.random_effects_df()).
    ax : matplotlib Axes or None
    predictor : str
        Column name for the random slope to plot.

    Returns
    -------
    matplotlib Axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    if isinstance(result_or_df, pd.DataFrame):
        re_df = result_or_df
    else:
        raise TypeError("Pass the random_effects_df() DataFrame")

    if predictor not in re_df.columns:
        # Fall back to Group column if present
        if "Group" in re_df.columns:
            predictor = "Group"
        else:
            predictor = re_df.columns[0]

    values = re_df[predictor].sort_values()
    subjects = values.index

    y_pos = np.arange(len(subjects))
    ax.barh(y_pos, values.values, height=0.6, color="steelblue", alpha=0.7,
            edgecolor="navy", linewidth=0.5)
    ax.axvline(0, color="black", linestyle="--", linewidth=1)
    ax.axvline(values.mean(), color="red", linestyle="-", linewidth=1.5,
               label=f"Mean = {values.mean():.3f}")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(subjects, fontsize=7)
    ax.set_xlabel(f"Random Effect ({predictor})")
    ax.set_ylabel("Subject")
    ax.set_title(f"Per-Subject Random Slopes: {predictor}")
    ax.legend(fontsize=9)
    return ax


def plot_multichannel_results(channel_results, channels=None, ax=None):
    """Bar chart of RPE coefficients per channel with significance stars.

    Parameters
    ----------
    channel_results : dict
        Output from multichannel_lme().
    channels : list of str or None
        Channel order. Defaults to channel_results['channels'].
    ax : matplotlib Axes or None

    Returns
    -------
    matplotlib Axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))

    if channels is None:
        channels = channel_results.get("channels", [])

    coefs = channel_results["coefficients"]
    pvals_fdr = channel_results["pvalues_fdr"]
    significant = channel_results["significant"]

    x = np.arange(len(channels))
    colors = ["steelblue" if sig else "lightgray" for sig in significant]

    bars = ax.bar(x, coefs, color=colors, edgecolor="navy", linewidth=0.5)

    # Add significance stars
    for i, (sig, pval) in enumerate(zip(significant, pvals_fdr)):
        if sig:
            star = "***" if pval < 0.001 else ("**" if pval < 0.01 else "*")
            y_pos = coefs[i] + 0.02 * np.sign(coefs[i]) * max(abs(coefs))
            ax.text(i, y_pos, star, ha="center", va="bottom", fontsize=12,
                    fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(channels, rotation=45, ha="right")
    ax.axhline(0, color="black", linestyle="-", linewidth=0.5)
    ax.set_xlabel("Channel")
    ax.set_ylabel("RPE Coefficient (LME)")
    ax.set_title("Multi-Channel RPE Effects (FDR corrected)")
    return ax


def plot_temporal_lme(times, coefficients, pvalues_fdr, ax=None,
                      significant=None, rewp_window=(0.240, 0.340)):
    """Time course of RPE coefficient with shaded significant clusters.

    Parameters
    ----------
    times : array-like
        Time vector in seconds.
    coefficients : array-like
        RPE coefficient at each timepoint.
    pvalues_fdr : array-like
        FDR-corrected p-values.
    ax : matplotlib Axes or None
    significant : array-like or None
        Boolean mask of significant timepoints.
    rewp_window : tuple
        (tmin, tmax) of expected REWP window for reference.

    Returns
    -------
    matplotlib Axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4))

    times = np.asarray(times)
    coefficients = np.asarray(coefficients)
    pvalues_fdr = np.asarray(pvalues_fdr)

    if significant is None:
        significant = pvalues_fdr < 0.05

    times_ms = times * 1000

    ax.plot(times_ms, coefficients, color="navy", linewidth=2)
    ax.fill_between(
        times_ms, 0, coefficients,
        where=significant, alpha=0.3, color="steelblue",
        label="FDR significant",
    )
    ax.axhline(0, color="black", linestyle="--", linewidth=0.8)

    # Mark REWP window
    if rewp_window:
        ax.axvspan(rewp_window[0] * 1000, rewp_window[1] * 1000,
                    alpha=0.1, color="orange", label="REWP window")

    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("RPE Coefficient (LME)")
    ax.set_title("Time-Resolved RPE Effect on EEG Amplitude")
    ax.legend(fontsize=9)
    return ax


def plot_model_comparison(comparison_df, ax=None, metric="BIC"):
    """Grouped bar chart of AIC/BIC across models.

    Parameters
    ----------
    comparison_df : pd.DataFrame
        Output from compare_predictors().
    ax : matplotlib Axes or None
    metric : str
        "BIC", "AIC", or "both".

    Returns
    -------
    matplotlib Axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))

    df = comparison_df.dropna(subset=["AIC", "BIC"])
    x = np.arange(len(df))

    if metric == "both":
        width = 0.35
        ax.bar(x - width / 2, df["AIC"].values, width, label="AIC",
               color="steelblue", edgecolor="navy", linewidth=0.5)
        ax.bar(x + width / 2, df["BIC"].values, width, label="BIC",
               color="coral", edgecolor="darkred", linewidth=0.5)
        ax.legend()
    elif metric == "AIC":
        ax.bar(x, df["AIC"].values, color="steelblue", edgecolor="navy",
               linewidth=0.5)
    else:
        ax.bar(x, df["BIC"].values, color="coral", edgecolor="darkred",
               linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(df["model"].values, rotation=45, ha="right")
    ax.set_ylabel(metric if metric != "both" else "Information Criterion")
    ax.set_title(f"Model Comparison ({metric})")

    # Highlight best model
    if metric != "both":
        best_idx = df[metric].idxmin()
        best_name = df.loc[best_idx, "model"]
        ax.annotate(
            f"Best: {best_name}",
            xy=(0.02, 0.95), xycoords="axes fraction",
            fontsize=10, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow"),
        )

    return ax


def plot_diagnostic(result, df, ax=None, rpe_col="rpe", rewp_col="rewp"):
    """Residual diagnostic plots: residuals vs fitted + QQ plot.

    Parameters
    ----------
    result : FusionLMEResult
        Fitted result (needs summary_text at minimum).
    df : pd.DataFrame
        Original data.
    ax : array of Axes or None
        Should have 2 axes (1x2).

    Returns
    -------
    array of Axes
    """
    if ax is None:
        fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    # Get predicted and residuals from the data
    y = df[rewp_col].values
    x = df[rpe_col].values

    # Simple OLS residuals as proxy (true LME residuals require the model object)
    slope, intercept = np.polyfit(x, y, 1)
    predicted = intercept + slope * x
    residuals = y - predicted

    # Residuals vs fitted
    ax[0].scatter(predicted, residuals, alpha=0.3, s=10, color="steelblue")
    ax[0].axhline(0, color="red", linestyle="--", linewidth=1)
    # LOWESS smoother
    try:
        from statsmodels.nonparametric.smoothers_lowess import lowess
        smoothed = lowess(residuals, predicted, frac=0.3)
        ax[0].plot(smoothed[:, 0], smoothed[:, 1], color="red", linewidth=2)
    except ImportError:
        pass
    ax[0].set_xlabel("Fitted Values")
    ax[0].set_ylabel("Residuals")
    ax[0].set_title("Residuals vs Fitted")

    # QQ plot
    (osm, osr), (slope_qq, intercept_qq, r) = sp_stats.probplot(residuals)
    ax[1].scatter(osm, osr, alpha=0.4, s=10, color="steelblue")
    qqline_x = np.array([osm.min(), osm.max()])
    ax[1].plot(qqline_x, intercept_qq + slope_qq * qqline_x, "r-", linewidth=2)
    ax[1].set_xlabel("Theoretical Quantiles")
    ax[1].set_ylabel("Sample Quantiles")
    ax[1].set_title(f"Q-Q Plot (r = {r:.4f})")

    return ax


def _find_rpe_key(coefficients, rpe_predictor):
    """Find the RPE coefficient key in the coefficients dict."""
    if rpe_predictor in coefficients:
        return rpe_predictor
    for key in coefficients:
        if rpe_predictor.lower() in key.lower() and key != "Intercept":
            return key
    return None
