"""
Example 10: Neural-Behavioral Fusion with Linear Mixed-Effects Models.

Demonstrates the full LME-based fusion pipeline with simulated data:
1. Generate multi-subject RPE-REWP dataset
2. Fit core RPE -> REWP model with random slopes
3. Print results, plot scatter + random effects
4. Run multi-channel analysis (simulated 5 channels)
5. Run time-resolved analysis (simulated 50 timepoints)
6. Compare all predictor models
7. Run LRT between nested models
8. Save all figures to results/lme_fusion/

Usage:
    cd ~/neuro-hub
    python examples/10_lme_neural_behavioral_fusion.py
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
    simulate_fusion_dataset,
    multichannel_lme,
    temporal_lme,
    compare_predictors,
    likelihood_ratio_test,
    list_models,
    plot_rpe_rewp_scatter,
    plot_random_effects,
    plot_multichannel_results,
    plot_temporal_lme,
    plot_model_comparison,
    plot_diagnostic,
)

# ── Configuration ───────────────────────────────────────────────
N_SUBJECTS = 20
N_TRIALS = 100
RPE_COUPLING = 2.0
SEED = 42
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "lme_fusion")
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 70)
print("Neural-Behavioral Fusion with Linear Mixed-Effects Models")
print("=" * 70)

# ── Step 1: Generate multi-subject fusion dataset ───────────────
print("\n[1] Generating simulated fusion dataset...")
df = simulate_fusion_dataset(
    n_subjects=N_SUBJECTS,
    n_trials=N_TRIALS,
    rpe_coupling=RPE_COUPLING,
    noise_std=5.0,
    seed=SEED,
)
print(f"    Dataset: {len(df)} observations, "
      f"{df['subject'].nunique()} subjects, "
      f"{N_TRIALS} trials each")
print(f"    Columns: {list(df.columns)}")

# ── Step 2: Fit core RPE -> REWP model ──────────────────────────
print("\n[2] Fitting core LME model (rewp ~ rpe, random slope)...")
lme = FusionLME()
result = lme.fit_model(df, "rpe_random_slope")
print(f"\n    {result}")
print(f"\n    Fixed effects:")
for name, coef in result.coefficients.items():
    sig = "***" if coef["p"] < 0.001 else ("**" if coef["p"] < 0.01 else ("*" if coef["p"] < 0.05 else ""))
    print(f"      {name:15s}  b={coef['estimate']:8.4f}  SE={coef['se']:.4f}  "
          f"z={coef['z']:7.3f}  p={coef['p']:.4f}{sig}")

print(f"\n    Random effects:")
for name, re in result.random_effects.items():
    print(f"      {name:15s}  var={re['variance']:.4f}  sd={re['sd']:.4f}")

print(f"\n    Model fit: AIC={result.model_fit['AIC']:.1f}, "
      f"BIC={result.model_fit['BIC']:.1f}")

# ── Step 3: Visualize scatter + random effects ──────────────────
print("\n[3] Plotting RPE-REWP scatter and random effects...")
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

plot_rpe_rewp_scatter(df, result, ax=axes[0])

re_df = lme.random_effects_df()
plot_random_effects(re_df, ax=axes[1])

plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "01_core_model.png"), dpi=150, bbox_inches="tight")
print(f"    Saved: {OUTPUT_DIR}/01_core_model.png")
plt.close(fig)

# ── Step 4: Multi-channel analysis ──────────────────────────────
print("\n[4] Running multi-channel LME analysis...")
# Simulate 5 channels with varying RPE coupling
rng = np.random.default_rng(SEED)
channels = ["FCz", "Cz", "Fz", "Pz", "Oz"]
coupling_strengths = [2.0, 1.5, 1.0, 0.3, 0.0]  # FCz strongest, Oz none

for ch, coupling in zip(channels, coupling_strengths):
    df[ch] = (df.groupby("subject").transform(lambda _: rng.normal(0, 2.0)).iloc[:, 0]
              + coupling * df["rpe"]
              + rng.normal(0, 5.0, len(df)))

mc_results = multichannel_lme(
    df, channels,
    formula="rewp ~ rpe",
    re_formula="~rpe",
)

print(f"    Channels tested: {channels}")
print(f"    FDR-significant: {[ch for ch, sig in zip(channels, mc_results['significant']) if sig]}")
for ch, coef, p_fdr, sig in zip(channels, mc_results["coefficients"],
                                  mc_results["pvalues_fdr"], mc_results["significant"]):
    star = "*" if sig else ""
    print(f"      {ch:5s}  coef={coef:7.4f}  p_fdr={p_fdr:.4f}{star}")

fig, ax = plt.subplots(figsize=(8, 5))
plot_multichannel_results(mc_results, ax=ax)
plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "02_multichannel.png"), dpi=150, bbox_inches="tight")
print(f"    Saved: {OUTPUT_DIR}/02_multichannel.png")
plt.close(fig)

# ── Step 5: Time-resolved analysis ──────────────────────────────
print("\n[5] Running time-resolved LME analysis...")
# Simulate 50 timepoints (-200 to 800 ms)
n_timepoints = 50
times = np.linspace(-0.2, 0.8, n_timepoints)
rewp_window = (0.240, 0.340)

# Generate amplitude at each timepoint: RPE effect only in REWP window
for i, t in enumerate(times):
    in_rewp = rewp_window[0] <= t <= rewp_window[1]
    coupling_t = RPE_COUPLING if in_rewp else 0.0
    df[f"amp_t{i}"] = (
        rng.normal(0, 2.0, len(df))
        + coupling_t * df["rpe"]
        + rng.normal(0, 5.0, len(df))
    )

temp_results = temporal_lme(
    df, times,
    formula_template="amp ~ rpe",
    re_formula=None,  # Intercept only for speed
)

n_sig = temp_results["significant"].sum()
print(f"    Timepoints: {n_timepoints}")
print(f"    FDR-significant: {n_sig}/{n_timepoints}")

fig, ax = plt.subplots(figsize=(10, 4))
plot_temporal_lme(
    temp_results["times"], temp_results["coefficients"],
    temp_results["pvalues_fdr"], ax=ax,
    significant=temp_results["significant"],
)
plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "03_temporal.png"), dpi=150, bbox_inches="tight")
print(f"    Saved: {OUTPUT_DIR}/03_temporal.png")
plt.close(fig)

# ── Step 6: Compare predictor models ────────────────────────────
print("\n[6] Comparing predictor models...")
# Only compare models whose columns exist in the data
models_to_compare = ["rpe_only", "rpe_random_slope", "rpe_outcome", "rpe_trial"]

print("\n    Available models:")
list_models()

comp = compare_predictors(df, models_to_compare)
print("\n    Model comparison (sorted by BIC):")
print(comp[["model", "AIC", "BIC", "rpe_coef", "rpe_p", "converged"]].to_string(index=False))

fig, ax = plt.subplots(figsize=(8, 5))
plot_model_comparison(comp, ax=ax, metric="both")
plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "04_model_comparison.png"), dpi=150, bbox_inches="tight")
print(f"    Saved: {OUTPUT_DIR}/04_model_comparison.png")
plt.close(fig)

# ── Step 7: Likelihood ratio test ───────────────────────────────
print("\n[7] Likelihood ratio test: rpe_only vs rpe_random_slope...")
lme_ml = FusionLME(reml=False)  # Must use ML for LRT
result_restricted = lme_ml.fit_model(df, "rpe_only")
result_full = lme_ml.fit_model(df, "rpe_random_slope")

lrt = likelihood_ratio_test(result_restricted, result_full, df_diff=2)
print(f"    chi2 = {lrt['chi2']:.3f}, df = {lrt['df']}, p = {lrt['p']:.4f}")
print(f"    Prefer full model (random slope): {lrt['prefer_full']}")

# ── Step 8: Diagnostic plots ────────────────────────────────────
print("\n[8] Generating diagnostic plots...")
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
plot_diagnostic(result, df, ax=axes)
plt.suptitle("LME Diagnostic Plots", fontsize=14)
plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "05_diagnostics.png"), dpi=150, bbox_inches="tight")
print(f"    Saved: {OUTPUT_DIR}/05_diagnostics.png")
plt.close(fig)

# ── Summary ─────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"  Core model RPE coefficient: {result.coefficients['rpe']['estimate']:.4f} "
      f"(p = {result.coefficients['rpe']['p']:.4f})")
print(f"  True coupling: {RPE_COUPLING}")
print(f"  Multi-channel FDR-significant: "
      f"{[ch for ch, sig in zip(channels, mc_results['significant']) if sig]}")
print(f"  Temporal FDR-significant timepoints: {n_sig}/{n_timepoints}")
print(f"  Best model by BIC: {comp.iloc[0]['model']}")
print(f"  LRT prefers random slope: {lrt['prefer_full']}")
print(f"\n  All figures saved to: {OUTPUT_DIR}/")
print("=" * 70)
