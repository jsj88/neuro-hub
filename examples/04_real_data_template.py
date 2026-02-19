"""
Example 4: Template for fitting models to REAL behavioral data.

Replace the data loading section with your actual data.
Expected CSV format:
    subject,trial,choice,outcome
    0,0,1,1
    0,1,0,0
    ...

Usage:
    cd ~/neuro-hub
    python examples/04_real_data_template.py --data path/to/data.csv
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from agent.tools.model_fitting import fit_model, MODEL_SPECS, extract_trial_variables

# ── Argument parsing ──────────────────────────────────────────
parser = argparse.ArgumentParser(description="Fit reward models to behavioral data")
parser.add_argument("--data", type=str, default=None,
                    help="Path to CSV file (columns: subject, trial, choice, outcome)")
parser.add_argument("--subject", type=int, default=None,
                    help="Subject ID to fit (if multi-subject)")
parser.add_argument("--models", nargs="+",
                    default=["random", "wsls", "rw", "q_dual",
                             "actor_critic", "q_decay", "ck", "rwck"],
                    help="Models to compare")
parser.add_argument("--n-options", type=int, default=2,
                    help="Number of choice options (2 for bandit, 4 for IGT)")
args = parser.parse_args()

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "real_data")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Step 1: Load data ─────────────────────────────────────────
if args.data is not None:
    print(f"Loading data from: {args.data}")
    df = pd.read_csv(args.data)
else:
    # Generate demo data if no real data provided
    print("No data file provided. Generating demo data...")
    from agent.tools.simulate import simulate_rw
    sim = simulate_rw(200, 2, np.array([0.7, 0.3]), alpha=0.25, beta=4.0, seed=99)
    df = pd.DataFrame({
        "subject": 0,
        "trial": np.arange(200),
        "choice": sim["choices"],
        "outcome": sim["outcomes"],
    })
    print("  (Using simulated RW data as demo)")

# Filter by subject if specified
if args.subject is not None:
    df = df[df["subject"] == args.subject]

choices = df["choice"].values.astype(int)
outcomes = df["outcome"].values.astype(float)
n_trials = len(choices)

print(f"  Trials: {n_trials}")
print(f"  Choice distribution: {np.bincount(choices, minlength=args.n_options)}")
print(f"  Mean reward: {outcomes.mean():.3f}")

# ── Step 2: Fit all models ────────────────────────────────────
print(f"\nFitting {len(args.models)} models...")
results = []
for model_name in args.models:
    if model_name not in MODEL_SPECS:
        print(f"  Skipping unknown model: {model_name}")
        continue
    r = fit_model(model_name, choices, outcomes,
                  n_options=args.n_options, method="de")
    results.append(r)
    print(f"  {r.model:<15} NLL={r.nll:>8.2f}  BIC={r.bic:>8.2f}  AIC={r.aic:>8.2f}")

# ── Step 3: Rank by BIC ──────────────────────────────────────
results.sort(key=lambda x: x.bic)
best = results[0]
print(f"\nBest model: {best.model}")
print(f"  Parameters: {best.params}")
print(f"  BIC: {best.bic:.2f}")
print(f"  Delta-BIC to 2nd best: {results[1].bic - best.bic:.2f}")

# ── Step 4: Extract latent variables ──────────────────────────
print(f"\nExtracting trial-level variables from {best.model}...")
latent = extract_trial_variables(best.model, best.params, choices, outcomes,
                                  n_options=args.n_options)

rpes = latent["rpes"]
print(f"  RPE range: [{rpes.min():.3f}, {rpes.max():.3f}]")

# Save RPEs for downstream neural analysis
rpe_df = df.copy()
rpe_df["rpe"] = rpes
if "q_values" in latent:
    for k in range(latent["q_values"].shape[1]):
        rpe_df[f"Q_{k}"] = latent["q_values"][:, k]

rpe_path = os.path.join(OUTPUT_DIR, "fitted_with_rpes.csv")
rpe_df.to_csv(rpe_path, index=False)
print(f"  Saved to: {rpe_path}")

# ── Step 5: Visualize ─────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# BIC comparison
ax = axes[0]
names = [r.model for r in results]
bics = [r.bic for r in results]
delta_bics = [b - best.bic for b in bics]
colors = ["green" if d == 0 else "orange" if d < 10 else "gray" for d in delta_bics]
ax.barh(names, delta_bics, color=colors)
ax.set_xlabel("Delta-BIC (vs best)")
ax.set_title("Model Comparison")
ax.axvline(10, ls="--", color="red", alpha=0.5, label="Strong evidence")
ax.legend()

# RPE time series
ax = axes[1]
ax.plot(rpes, alpha=0.6, color="teal")
ax.axhline(0, ls="--", color="black", alpha=0.3)
ax.set_xlabel("Trial")
ax.set_ylabel("RPE")
ax.set_title(f"Prediction Errors ({best.model})")

# Choice + outcome
ax = axes[2]
window = 20
rolling_choice = np.convolve(choices == 0, np.ones(window) / window, mode="valid")
rolling_reward = np.convolve(outcomes, np.ones(window) / window, mode="valid")
ax.plot(rolling_choice, label="P(option 0)", color="blue")
ax.plot(rolling_reward, label="Mean reward", color="green", alpha=0.7)
ax.set_xlabel("Trial")
ax.set_ylabel("Rate")
ax.set_title(f"Behavior ({window}-trial window)")
ax.legend()
ax.set_ylim(0, 1.05)

plt.tight_layout()
out_path = os.path.join(OUTPUT_DIR, "04_real_data_results.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"\nFigure saved: {out_path}")
plt.close()

print("\nDone! Next steps:")
print("  1. Use RPEs from fitted_with_rpes.csv as EEG/fMRI regressors")
print("  2. Correlate with REWP amplitude at FCz (240-340ms)")
print("  3. Or use as parametric modulators in GLM")
