"""
Example 11: LME Fusion with Real T-Maze Data.

Template for applying LME neural-behavioral fusion to the real
T-maze dataset (merged_clean_V3_scaled.csv).

Pipeline:
1. Load behavioral CSV with real column names
2. Fit RW model per subject, extract trial-level RPEs
3. Use ERP amplitudes from CSV (FCz, Cz, etc.)
4. Merge into fusion DataFrame
5. Run LME models including T-maze-specific ones
6. Run full LME suite (core, multichannel, temporal, comparison)
7. Save results

Usage:
    cd ~/neuro-hub
    python examples/11_lme_real_data_fusion.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from neural_behavioral_fusion import (
    FusionLME,
    load_real_fusion_data,
    multichannel_lme,
    compare_predictors,
    likelihood_ratio_test,
    plot_rpe_rewp_scatter,
    plot_random_effects,
    plot_multichannel_results,
    plot_model_comparison,
    plot_diagnostic,
)

# ── Configuration ───────────────────────────────────────────────
DATA_CSV = os.path.join(
    os.path.dirname(__file__), "..", "sample_data", "merged_clean_V3_scaled.csv"
)
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "lme_real_fusion")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ERP channels available in the CSV
ERP_CHANNELS = ["FCz", "Cz", "Pz", "FC1", "FC2", "F4", "F5", "C4", "C3", "P4", "P3", "P8", "P7"]
DEFAULT_REWP_CHANNEL = "FCz"

print("=" * 70)
print("LME Neural-Behavioral Fusion — Real T-Maze Data")
print("=" * 70)

# ── Step 1: Load real data ──────────────────────────────────────
print("\n[1] Loading real T-maze data...")
df = load_real_fusion_data(behavior_csv=DATA_CSV, channels=[DEFAULT_REWP_CHANNEL])
print(f"    Loaded: {len(df)} trials, {df['subject'].nunique()} subjects")
print(f"    Columns: {list(df.columns[:15])}...")

# ── Step 2: Fit RW model per subject, extract RPEs ──────────────
print("\n[2] Fitting Rescorla-Wagner model per subject and extracting RPEs...")

try:
    from agent.tools.model_fitting import fit_model, extract_trial_variables
    USE_MODEL_FITTING = True
except ImportError:
    USE_MODEL_FITTING = False
    print("    Warning: agent.tools.model_fitting not available, using simple RPE proxy")

if USE_MODEL_FITTING:
    all_rpes = []
    subjects = df["subject"].unique()
    for subj in subjects:
        mask = df["subject"] == subj
        subj_df = df[mask].copy()

        # Encode choices: Direction 1=Left(0), 2=Right(1)
        choices = (subj_df["Direction"].values - 1).astype(int)
        outcomes = subj_df["reward"].values.astype(float)

        try:
            fit_result = fit_model(
                "rw", choices, outcomes,
                n_options=2, method="multistart", n_starts=5,
            )
            latent = extract_trial_variables("rw", fit_result.params, choices, outcomes)
            rpes = latent["rpes"]
        except Exception as e:
            print(f"    Warning: Fit failed for {subj}: {e}")
            # Fallback: simple outcome - 0.5 as RPE proxy
            rpes = outcomes - 0.5

        all_rpes.append(pd.Series(rpes, index=subj_df.index))

    df["rpe"] = pd.concat(all_rpes)
    print(f"    RPEs extracted for {len(subjects)} subjects")
else:
    # Simple RPE proxy: outcome - running mean
    df["rpe"] = df.groupby("subject")["reward"].transform(
        lambda x: x - x.expanding().mean().shift(1).fillna(0.5)
    )
    print("    Using simple RPE proxy (reward - running mean)")

# ── Step 3: Set REWP from ERP channel ──────────────────────────
print(f"\n[3] Using {DEFAULT_REWP_CHANNEL} as REWP amplitude...")
if DEFAULT_REWP_CHANNEL in df.columns:
    df["rewp"] = df[DEFAULT_REWP_CHANNEL]
else:
    print(f"    ERROR: {DEFAULT_REWP_CHANNEL} not found in data")
    sys.exit(1)

# Drop rows with missing RPE or REWP
df_clean = df.dropna(subset=["rpe", "rewp", "subject"]).copy()
print(f"    Clean dataset: {len(df_clean)} trials")

# ── Step 4: Core LME model ─────────────────────────────────────
print("\n[4] Fitting core LME: rewp ~ rpe (random intercept + slope)...")
lme = FusionLME()
result = lme.fit_model(df_clean, "rpe_random_slope")
print(f"\n    {result}")
print(f"\n    Fixed effects:")
for name, coef in result.coefficients.items():
    sig = "***" if coef["p"] < 0.001 else ("**" if coef["p"] < 0.01 else ("*" if coef["p"] < 0.05 else ""))
    print(f"      {name:15s}  b={coef['estimate']:8.4f}  SE={coef['se']:.4f}  "
          f"z={coef['z']:7.3f}  p={coef['p']:.4f}{sig}")

# Plot scatter
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
plot_rpe_rewp_scatter(df_clean, result, ax=axes[0])

try:
    re_df = lme.random_effects_df()
    plot_random_effects(re_df, ax=axes[1])
except Exception:
    axes[1].text(0.5, 0.5, "Random effects\nnot available", ha="center", va="center",
                 transform=axes[1].transAxes, fontsize=14)

plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "01_core_model_real.png"), dpi=150, bbox_inches="tight")
print(f"    Saved: {OUTPUT_DIR}/01_core_model_real.png")
plt.close(fig)

# ── Step 5: Multi-channel analysis ──────────────────────────────
print("\n[5] Running multi-channel LME across ERP channels...")
available_channels = [ch for ch in ERP_CHANNELS if ch in df_clean.columns]
print(f"    Available channels: {available_channels}")

mc_results = multichannel_lme(
    df_clean, available_channels,
    formula="rewp ~ rpe",
    re_formula=None,  # Random intercept only for stability with real data
)

print(f"    FDR-significant: {[ch for ch, sig in zip(available_channels, mc_results['significant']) if sig]}")
for ch, coef, p_fdr, sig in zip(available_channels, mc_results["coefficients"],
                                  mc_results["pvalues_fdr"], mc_results["significant"]):
    star = "*" if sig else ""
    print(f"      {ch:5s}  coef={coef:7.4f}  p_fdr={p_fdr:.4f}{star}")

fig, ax = plt.subplots(figsize=(10, 5))
plot_multichannel_results(mc_results, channels=available_channels, ax=ax)
plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "02_multichannel_real.png"), dpi=150, bbox_inches="tight")
print(f"    Saved: {OUTPUT_DIR}/02_multichannel_real.png")
plt.close(fig)

# ── Step 6: T-maze-specific models ─────────────────────────────
print("\n[6] Comparing predictor models (including T-maze-specific)...")
# Determine which models can run on this data
models_to_try = ["rpe_only", "rpe_random_slope", "rpe_outcome", "rpe_trial"]

# Check if T-maze-specific columns exist
if "PES_PRS_index" in df_clean.columns:
    models_to_try.append("rpe_stay_shift")
    print("    Including rpe_stay_shift (PES_PRS_index available)")
if "Vel_Cond" in df_clean.columns:
    models_to_try.append("rpe_velocity")
    print("    Including rpe_velocity (Vel_Cond available)")

comp = compare_predictors(df_clean, models_to_try)
print("\n    Model comparison (sorted by BIC):")
print(comp[["model", "AIC", "BIC", "rpe_coef", "rpe_p", "converged"]].to_string(index=False))

fig, ax = plt.subplots(figsize=(10, 5))
plot_model_comparison(comp, ax=ax, metric="both")
plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "03_model_comparison_real.png"), dpi=150, bbox_inches="tight")
print(f"    Saved: {OUTPUT_DIR}/03_model_comparison_real.png")
plt.close(fig)

# ── Step 7: LRT for random slope ───────────────────────────────
print("\n[7] Likelihood ratio test: intercept-only vs random slope...")
lme_ml = FusionLME(reml=False)
try:
    result_restricted = lme_ml.fit_model(df_clean, "rpe_only")
    result_full = lme_ml.fit_model(df_clean, "rpe_random_slope")
    lrt = likelihood_ratio_test(result_restricted, result_full, df_diff=2)
    print(f"    chi2 = {lrt['chi2']:.3f}, df = {lrt['df']}, p = {lrt['p']:.4f}")
    print(f"    Prefer random slope: {lrt['prefer_full']}")
except Exception as e:
    print(f"    LRT failed: {e}")

# ── Step 8: Diagnostic plots ────────────────────────────────────
print("\n[8] Generating diagnostic plots...")
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
plot_diagnostic(result, df_clean, ax=axes)
plt.suptitle("LME Diagnostic Plots — Real T-Maze Data", fontsize=14)
plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "04_diagnostics_real.png"), dpi=150, bbox_inches="tight")
print(f"    Saved: {OUTPUT_DIR}/04_diagnostics_real.png")
plt.close(fig)

# ── Save results to CSV ────────────────────────────────────────
print("\n[9] Saving results...")
comp.to_csv(os.path.join(OUTPUT_DIR, "model_comparison.csv"), index=False)

# Save fixed effects from core model
fx_df = pd.DataFrame(result.coefficients).T
fx_df.to_csv(os.path.join(OUTPUT_DIR, "core_fixed_effects.csv"))

# Save multichannel results
mc_df = pd.DataFrame({
    "channel": available_channels,
    "coefficient": mc_results["coefficients"],
    "pvalue": mc_results["pvalues"],
    "pvalue_fdr": mc_results["pvalues_fdr"],
    "significant": mc_results["significant"],
})
mc_df.to_csv(os.path.join(OUTPUT_DIR, "multichannel_results.csv"), index=False)

print(f"    Results saved to: {OUTPUT_DIR}/")

# ── Summary ─────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("SUMMARY — Real T-Maze Data")
print("=" * 70)
rpe_key = "rpe" if "rpe" in result.coefficients else list(result.coefficients.keys())[1]
print(f"  Core model RPE coefficient: {result.coefficients[rpe_key]['estimate']:.4f} "
      f"(p = {result.coefficients[rpe_key]['p']:.4f})")
print(f"  Subjects: {df_clean['subject'].nunique()}")
print(f"  Trials: {len(df_clean)}")
sig_channels = [ch for ch, sig in zip(available_channels, mc_results['significant']) if sig]
print(f"  Multi-channel FDR-significant: {sig_channels if sig_channels else 'none'}")
print(f"  Best model by BIC: {comp.iloc[0]['model']}")
print(f"\n  All outputs saved to: {OUTPUT_DIR}/")
print("=" * 70)
