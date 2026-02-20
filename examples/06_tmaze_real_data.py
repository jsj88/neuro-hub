"""
Example 6: Full analysis pipeline for T-Maze behavioral + ERP data.

Loads your real T-maze data, fits reward learning models per subject,
extracts RPEs, and correlates with ERP amplitudes (REWP, P3, N2, P2).

Usage:
    cd ~/neuro-hub
    python examples/06_tmaze_real_data.py
"""

import sys
import os
import re
import warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr, ttest_ind
from agent.tools.model_fitting import fit_model, MODEL_SPECS, extract_trial_variables

# ── Paths ────────────────────────────────────────────────────────
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "sample_data")
BEH_FILE = os.path.join(DATA_DIR, "RT_DATA_v9_earned_allSUBS.csv")
ERP_FILE = os.path.join(DATA_DIR, "ERP_STATS_ALL.xlsx")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "tmaze_analysis")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Models to fit (bandit-compatible) ────────────────────────────
MODELS = ["random", "wsls", "rw", "q_learning", "q_dual",
          "actor_critic", "q_decay", "rw_bias", "ck", "rwck"]

# =====================================================================
# STEP 1: Load and reformat behavioral data
# =====================================================================
print("=" * 60)
print("STEP 1: Loading behavioral data")
print("=" * 60)

beh = pd.read_csv(BEH_FILE)
print(f"  Raw: {beh.shape[0]} rows, {beh['SubjectCode'].nunique()} subjects")

# Remap to model-fitting format:
#   choice = Direction - 1 (0 or 1)
#   outcome = 1 if Reward==1, 0 if Reward==2
beh["choice"] = beh["Direction"] - 1        # 0 = left, 1 = right
beh["outcome"] = (beh["Reward"] == 1).astype(float)  # 1=reward, 0=no reward

# Also create outcome with magnitude (Earned values)
beh["outcome_mag"] = beh["Earned"]  # 0, 0.05, 0.10, 0.25

subjects = sorted(beh["SubjectCode"].unique())
print(f"  Subjects: {len(subjects)}")
print(f"  Choice (0=Dir1, 1=Dir2) dist: {beh['choice'].value_counts().to_dict()}")
print(f"  Outcome (0=no, 1=yes) dist: {beh['outcome'].value_counts().to_dict()}")

# =====================================================================
# STEP 2: Fit all models per subject
# =====================================================================
print(f"\n{'=' * 60}")
print(f"STEP 2: Fitting {len(MODELS)} models to each subject")
print("=" * 60)

all_fits = []  # Collect per-subject, per-model results
best_models = {}  # Best model per subject

for i, subj in enumerate(subjects):
    subj_data = beh[beh["SubjectCode"] == subj].sort_values("Trial")
    choices = subj_data["choice"].values.astype(int)
    outcomes = subj_data["outcome"].values.astype(float)
    n_trials = len(choices)

    subj_results = []
    for model_name in MODELS:
        try:
            r = fit_model(model_name, choices, outcomes,
                          n_options=2, method="de", seed=42)
            subj_results.append({
                "subject": subj,
                "model": model_name,
                "nll": r.nll,
                "bic": r.bic,
                "aic": r.aic,
                "n_params": r.n_params,
                "n_trials": n_trials,
                **r.params,
            })
        except Exception as e:
            pass

    # Find best model by BIC
    if subj_results:
        best = min(subj_results, key=lambda x: x["bic"])
        best_models[subj] = best["model"]
        all_fits.extend(subj_results)

    if (i + 1) % 10 == 0 or i == 0:
        print(f"  Subject {subj} ({i+1}/{len(subjects)}): "
              f"best={best['model']}, BIC={best['bic']:.1f}, "
              f"n_trials={n_trials}")

fits_df = pd.DataFrame(all_fits)
fits_df.to_csv(os.path.join(OUTPUT_DIR, "all_model_fits.csv"), index=False)

# Summary: count how many subjects prefer each model
print(f"\n  Best model counts:")
from collections import Counter
for model, count in Counter(best_models.values()).most_common():
    print(f"    {model:<15} {count} subjects ({count/len(subjects)*100:.0f}%)")

# =====================================================================
# STEP 3: Group-level model comparison (summed BIC)
# =====================================================================
print(f"\n{'=' * 60}")
print("STEP 3: Group-level model comparison")
print("=" * 60)

group_bic = fits_df.groupby("model")["bic"].sum().sort_values()
best_group_model = group_bic.index[0]
print(f"\n  Model ranking by summed BIC:")
for model in group_bic.index:
    delta = group_bic[model] - group_bic.iloc[0]
    marker = " *** BEST" if delta == 0 else ""
    print(f"    {model:<15} BIC={group_bic[model]:>10.1f}  "
          f"delta={delta:>8.1f}{marker}")

print(f"\n  Best group-level model: {best_group_model}")

# =====================================================================
# STEP 4: Extract RPEs from best-fitting model per subject
# =====================================================================
print(f"\n{'=' * 60}")
print("STEP 4: Extracting RPEs per subject")
print("=" * 60)

rpe_records = []
subject_rpe_metrics = []

for subj in subjects:
    subj_data = beh[beh["SubjectCode"] == subj].sort_values("Trial")
    choices = subj_data["choice"].values.astype(int)
    outcomes = subj_data["outcome"].values.astype(float)

    # Get best model for this subject
    model_name = best_models.get(subj, best_group_model)
    subj_fit = fits_df[(fits_df["subject"] == subj) &
                        (fits_df["model"] == model_name)].iloc[0]

    # Reconstruct params dict
    spec = MODEL_SPECS[model_name]
    params = {p: subj_fit[p] for p in spec["param_names"]}

    # Extract trial-level variables
    latent = extract_trial_variables(model_name, params, choices, outcomes,
                                      n_options=2)
    rpes = latent["rpes"]

    # Save trial-level RPEs
    for t_idx in range(len(choices)):
        rpe_records.append({
            "subject": subj,
            "trial": t_idx + 1,
            "choice": choices[t_idx],
            "outcome": outcomes[t_idx],
            "rpe": rpes[t_idx],
            "model": model_name,
        })

    # Compute subject-level RPE metrics
    subject_rpe_metrics.append({
        "subject": subj,
        "model": model_name,
        "mean_rpe": np.mean(rpes),
        "mean_abs_rpe": np.mean(np.abs(rpes)),
        "std_rpe": np.std(rpes),
        "mean_pos_rpe": np.mean(rpes[rpes > 0]) if np.any(rpes > 0) else 0,
        "mean_neg_rpe": np.mean(rpes[rpes < 0]) if np.any(rpes < 0) else 0,
        "rpe_range": rpes.max() - rpes.min(),
        "optimal_choice_rate": np.mean(choices == 0),
        "reward_rate": np.mean(outcomes),
        "bic": subj_fit["bic"],
        **params,
    })

rpe_df = pd.DataFrame(rpe_records)
rpe_df.to_csv(os.path.join(OUTPUT_DIR, "trial_level_rpes.csv"), index=False)
metrics_df = pd.DataFrame(subject_rpe_metrics)
metrics_df.to_csv(os.path.join(OUTPUT_DIR, "subject_rpe_metrics.csv"), index=False)

print(f"  Saved trial-level RPEs: {len(rpe_df)} rows")
print(f"  Saved subject metrics: {len(metrics_df)} rows")
print(f"  Mean |RPE| across subjects: {metrics_df['mean_abs_rpe'].mean():.3f}")

# =====================================================================
# STEP 5: Load ERP data and merge with model outputs
# =====================================================================
print(f"\n{'=' * 60}")
print("STEP 5: Loading ERP data and merging")
print("=" * 60)

erp = pd.read_excel(ERP_FILE)

# Map ERP subject IDs to behavioral SubjectCode
def extract_subj_num(s):
    m = re.search(r"(\d+)", s)
    return int(m.group(1)) if m else None

erp["SubjectCode"] = erp["Subject"].apply(extract_subj_num)

# Extract REWP (difference wave) at FCz per subject
rewp_data = erp[erp["LABEL"] == "REWP"][["SubjectCode", "FCz", "Cz", "Pz", "GROUP"]].copy()
rewp_data.columns = ["subject", "REWP_FCz", "REWP_Cz", "REWP_Pz", "GROUP"]

# Extract P3 reward vs no-reward
p3_rew = erp[(erp["LABEL"] == "P3") & (erp["CONDITION"] == "REW")][["SubjectCode", "FCz", "Cz", "Pz"]].copy()
p3_rew.columns = ["subject", "P3_REW_FCz", "P3_REW_Cz", "P3_REW_Pz"]
p3_nrew = erp[(erp["LABEL"] == "P3") & (erp["CONDITION"] == "NREW")][["SubjectCode", "FCz", "Cz", "Pz"]].copy()
p3_nrew.columns = ["subject", "P3_NREW_FCz", "P3_NREW_Cz", "P3_NREW_Pz"]

# Merge all
merged = metrics_df.merge(rewp_data, on="subject", how="inner")
merged = merged.merge(p3_rew, on="subject", how="left")
merged = merged.merge(p3_nrew, on="subject", how="left")
merged["P3_diff_FCz"] = merged["P3_REW_FCz"] - merged["P3_NREW_FCz"]

merged.to_csv(os.path.join(OUTPUT_DIR, "merged_model_erp.csv"), index=False)
print(f"  Matched subjects: {len(merged)} (behavioral + ERP)")
print(f"  REWP at FCz: mean={merged['REWP_FCz'].mean():.3f}, "
      f"SD={merged['REWP_FCz'].std():.3f}")

# =====================================================================
# STEP 6: Neural-model correlations
# =====================================================================
print(f"\n{'=' * 60}")
print("STEP 6: RPE-ERP correlations")
print("=" * 60)

correlations = []
neural_vars = {
    "REWP_FCz": "REWP at FCz (NREW-REW difference)",
    "REWP_Cz": "REWP at Cz",
    "P3_diff_FCz": "P3 difference (REW-NREW) at FCz",
}
model_vars = {
    "mean_abs_rpe": "Mean |RPE|",
    "std_rpe": "RPE variability",
    "mean_pos_rpe": "Mean positive RPE",
    "mean_neg_rpe": "Mean negative RPE",
    "reward_rate": "Overall reward rate",
}

print(f"\n  {'Model Variable':<20} {'Neural Variable':<35} {'r':>6} {'p':>8} {'sig':>5}")
print("  " + "-" * 78)

for mv_key, mv_label in model_vars.items():
    for nv_key, nv_label in neural_vars.items():
        x = merged[mv_key].values
        y = merged[nv_key].values
        mask = np.isfinite(x) & np.isfinite(y)
        if mask.sum() >= 5:
            r, p = pearsonr(x[mask], y[mask])
            sig = "*" if p < 0.05 else ("~" if p < 0.10 else "")
            correlations.append({
                "model_var": mv_key, "neural_var": nv_key,
                "r": r, "p": p, "n": mask.sum(),
            })
            if nv_key == "REWP_FCz":  # Print main channel
                print(f"  {mv_label:<20} {nv_label:<35} {r:>6.3f} {p:>8.4f} {sig:>5}")

corr_df = pd.DataFrame(correlations)
corr_df.to_csv(os.path.join(OUTPUT_DIR, "rpe_erp_correlations.csv"), index=False)

# ── Group comparison if groups exist ─────────────────────────────
if merged["GROUP"].nunique() > 1:
    print(f"\n  Group comparisons (GROUP variable):")
    for g in sorted(merged["GROUP"].unique()):
        gdata = merged[merged["GROUP"] == g]
        print(f"    Group {g} (n={len(gdata)}): "
              f"mean_abs_RPE={gdata['mean_abs_rpe'].mean():.3f}, "
              f"REWP_FCz={gdata['REWP_FCz'].mean():.3f}")

# =====================================================================
# STEP 7: Visualize results
# =====================================================================
print(f"\n{'=' * 60}")
print("STEP 7: Generating figures")
print("=" * 60)

fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# 7a: Model comparison (group BIC)
ax = axes[0, 0]
models_ranked = group_bic.index.tolist()
delta_bics = [group_bic[m] - group_bic.iloc[0] for m in models_ranked]
colors = ["green" if d == 0 else "orange" if d < 100 else "gray" for d in delta_bics]
ax.barh(models_ranked, delta_bics, color=colors)
ax.set_xlabel("Delta-BIC (vs best)")
ax.set_title("Group-Level Model Comparison (summed BIC)")
ax.axvline(100, ls="--", color="red", alpha=0.5, label="Strong evidence")
ax.legend()

# 7b: Best model distribution
ax = axes[0, 1]
model_counts = Counter(best_models.values())
models_sorted = sorted(model_counts.keys(), key=lambda x: -model_counts[x])
ax.bar(range(len(models_sorted)),
       [model_counts[m] for m in models_sorted],
       color="steelblue")
ax.set_xticks(range(len(models_sorted)))
ax.set_xticklabels(models_sorted, rotation=45, ha="right")
ax.set_ylabel("# Subjects")
ax.set_title("Best-Fitting Model per Subject")

# 7c: RPE-REWP scatter (main result)
ax = axes[0, 2]
x = merged["mean_abs_rpe"]
y = merged["REWP_FCz"]
ax.scatter(x, y, alpha=0.7, edgecolors="black", linewidth=0.5, color="teal")
# Regression line
mask = np.isfinite(x) & np.isfinite(y)
if mask.sum() > 2:
    z = np.polyfit(x[mask], y[mask], 1)
    xline = np.linspace(x[mask].min(), x[mask].max(), 100)
    ax.plot(xline, np.polyval(z, xline), "k-", linewidth=2)
    r, p = pearsonr(x[mask], y[mask])
    ax.set_title(f"RPE-REWP Correlation (r={r:.3f}, p={p:.4f})")
ax.set_xlabel("Mean |RPE| (model-derived)")
ax.set_ylabel("REWP amplitude at FCz (uV)")

# 7d: RPE distribution across subjects
ax = axes[1, 0]
ax.hist(metrics_df["mean_abs_rpe"], bins=20, color="steelblue",
        edgecolor="black", alpha=0.7)
ax.set_xlabel("Mean |RPE|")
ax.set_ylabel("# Subjects")
ax.set_title("Distribution of Mean |RPE| Across Subjects")

# 7e: Learning curves (first 10 subjects)
ax = axes[1, 1]
window = 15
for subj in subjects[:10]:
    subj_rpes = rpe_df[rpe_df["subject"] == subj]["rpe"].values
    if len(subj_rpes) > window:
        smoothed = np.convolve(np.abs(subj_rpes),
                               np.ones(window) / window, mode="valid")
        ax.plot(smoothed, alpha=0.5, linewidth=1)
ax.set_xlabel("Trial")
ax.set_ylabel("|RPE| (smoothed)")
ax.set_title(f"Learning Curves (|RPE|, {window}-trial window)")

# 7f: RPE variability vs REWP
ax = axes[1, 2]
x = merged["std_rpe"]
y = merged["REWP_FCz"]
groups = merged["GROUP"].values
for g in sorted(merged["GROUP"].unique()):
    mask_g = groups == g
    ax.scatter(x[mask_g], y[mask_g], alpha=0.7, label=f"Group {g}",
               edgecolors="black", linewidth=0.5)
ax.set_xlabel("RPE variability (SD)")
ax.set_ylabel("REWP at FCz (uV)")
r, p = pearsonr(x, y)
ax.set_title(f"RPE Variability vs REWP (r={r:.3f}, p={p:.4f})")
ax.legend()

plt.suptitle("T-Maze: Reward Learning Models × ERP Analysis", fontsize=16, y=1.02)
plt.tight_layout()
fig_path = os.path.join(OUTPUT_DIR, "06_tmaze_full_analysis.png")
plt.savefig(fig_path, dpi=150, bbox_inches="tight")
print(f"  Main figure saved: {fig_path}")
plt.close()

# ── Additional figure: Parameter distributions ───────────────────
# Get RW params for all subjects (since RW is common)
rw_fits = fits_df[fits_df["model"] == "rw"].dropna(subset=["alpha", "beta"])
if len(rw_fits) > 0:
    fig2, axes2 = plt.subplots(1, 3, figsize=(15, 5))

    ax = axes2[0]
    ax.hist(rw_fits["alpha"], bins=20, color="coral", edgecolor="black", alpha=0.7)
    ax.set_xlabel("Learning rate (alpha)")
    ax.set_ylabel("# Subjects")
    ax.set_title(f"RW Alpha Distribution (mean={rw_fits['alpha'].mean():.3f})")

    ax = axes2[1]
    ax.hist(rw_fits["beta"], bins=20, color="skyblue", edgecolor="black", alpha=0.7)
    ax.set_xlabel("Inverse temperature (beta)")
    ax.set_ylabel("# Subjects")
    ax.set_title(f"RW Beta Distribution (mean={rw_fits['beta'].mean():.3f})")

    # Alpha vs REWP
    ax = axes2[2]
    rw_merged = rw_fits.rename(columns={"subject": "subject"})\
        .merge(rewp_data, on="subject", how="inner")
    if len(rw_merged) > 5:
        r, p = pearsonr(rw_merged["alpha"], rw_merged["REWP_FCz"])
        ax.scatter(rw_merged["alpha"], rw_merged["REWP_FCz"],
                   alpha=0.7, color="purple", edgecolors="black", linewidth=0.5)
        ax.set_xlabel("RW Learning Rate (alpha)")
        ax.set_ylabel("REWP at FCz (uV)")
        ax.set_title(f"Learning Rate vs REWP (r={r:.3f}, p={p:.4f})")

    plt.suptitle("RW Model Parameters × REWP", fontsize=14)
    plt.tight_layout()
    fig2_path = os.path.join(OUTPUT_DIR, "06_rw_params_erp.png")
    plt.savefig(fig2_path, dpi=150, bbox_inches="tight")
    print(f"  Parameter figure saved: {fig2_path}")
    plt.close()

# =====================================================================
# SUMMARY
# =====================================================================
print(f"\n{'=' * 60}")
print("SUMMARY")
print("=" * 60)
print(f"  Subjects analyzed: {len(subjects)} behavioral, {len(merged)} with ERP")
print(f"  Best group model: {best_group_model}")
print(f"  Models fitted: {len(MODELS)} per subject")
print(f"  Trial-level RPEs: {len(rpe_df)} total trials")
print(f"\n  Output files in: {OUTPUT_DIR}")
print(f"    all_model_fits.csv      — All model fits per subject")
print(f"    trial_level_rpes.csv    — RPEs for every trial")
print(f"    subject_rpe_metrics.csv — Subject-level RPE summary")
print(f"    merged_model_erp.csv    — Model metrics + ERP amplitudes")
print(f"    rpe_erp_correlations.csv— All correlation results")
print(f"\nNext steps:")
print(f"  1. Use trial_level_rpes.csv as EEG single-trial regressors")
print(f"  2. Correlate RPEs with single-trial ERP amplitudes")
print(f"  3. Run temporal decoding on raw EEG epochs")
print(f"  4. Test group differences in model parameters")
