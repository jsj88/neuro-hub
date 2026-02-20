"""
Example 8: Single-trial RPE × ERP analysis.

Uses merged behavioral + single-trial ERP data to correlate
trial-by-trial prediction errors with ERP amplitudes.

This is the proper approach: within-subject single-trial correlations,
then group-level t-test on Fisher-z-transformed r values.

Usage:
    cd ~/neuro-hub
    python examples/08_single_trial_rpe_erp.py
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
from scipy.stats import pearsonr, ttest_1samp, wilcoxon
from agent.tools.model_fitting import fit_model, MODEL_SPECS, extract_trial_variables

# ── Paths ────────────────────────────────────────────────────────
DATA_FILE = "/Volumes/LaCie_2/ARmaze_v2/CSVFiles/BEH_RESULTS/merged_clean_V3_scaled.csv"
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "single_trial_fusion")
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODELS = ["random", "wsls", "rw", "q_learning", "q_dual",
          "actor_critic", "q_decay", "rw_bias", "ck", "rwck"]

ERP_CHANNELS = ["FCz", "Cz", "FC2", "FC1", "F4", "F5",
                "C4", "C3", "P4", "P3", "P8", "P7", "Pz"]

# =====================================================================
# STEP 1: Load single-trial data
# =====================================================================
print("=" * 60)
print("STEP 1: Loading single-trial merged data")
print("=" * 60)

df = pd.read_csv(DATA_FILE)
df["choice"] = df["Direction"] - 1
df["outcome"] = (df["Reward"] == 1).astype(float)

subjects = sorted(df["SubjectCode"].unique())
print(f"  {len(df)} trials, {len(subjects)} subjects")
print(f"  ERP channels: {ERP_CHANNELS}")
print(f"  FCz range: [{df['FCz'].min():.1f}, {df['FCz'].max():.1f}] uV")

# =====================================================================
# STEP 2: Fit models and extract trial-level RPEs per subject
# =====================================================================
print(f"\n{'=' * 60}")
print("STEP 2: Fitting models and extracting single-trial RPEs")
print("=" * 60)

all_trial_data = []
subject_fits = []
best_models = {}

for i, subj in enumerate(subjects):
    sdata = df[df["SubjectCode"] == subj].sort_values("Trial").copy()
    choices = sdata["choice"].values.astype(int)
    outcomes = sdata["outcome"].values.astype(float)

    # Fit all models
    results = []
    for model_name in MODELS:
        try:
            r = fit_model(model_name, choices, outcomes,
                          n_options=2, method="de", seed=42)
            results.append({"model": model_name, "bic": r.bic,
                            "nll": r.nll, "result": r})
        except:
            pass

    # Best model
    best = min(results, key=lambda x: x["bic"])
    best_models[subj] = best["model"]
    best_result = best["result"]

    # Store all model fits
    for r in results:
        subject_fits.append({
            "subject": subj, "model": r["model"],
            "bic": r["bic"], "nll": r["nll"],
        })

    # Extract RPEs from best model
    spec = MODEL_SPECS[best["model"]]
    latent = extract_trial_variables(
        best["model"], best_result.params, choices, outcomes, n_options=2)
    rpes = latent["rpes"]

    # Also extract RPEs from RW (for consistency across subjects)
    rw_result = [r for r in results if r["model"] == "rw"][0]["result"]
    rw_latent = extract_trial_variables("rw", rw_result.params,
                                         choices, outcomes, n_options=2)
    rw_rpes = rw_latent["rpes"]

    # Attach RPEs to trial data
    sdata = sdata.reset_index(drop=True)
    sdata["rpe_best"] = rpes
    sdata["rpe_rw"] = rw_rpes
    sdata["abs_rpe_best"] = np.abs(rpes)
    sdata["abs_rpe_rw"] = np.abs(rw_rpes)
    sdata["best_model"] = best["model"]
    all_trial_data.append(sdata)

    if (i + 1) % 10 == 0 or i == 0:
        print(f"  {subj} ({i+1}/{len(subjects)}): best={best['model']}, "
              f"BIC={best['bic']:.1f}")

trial_df = pd.concat(all_trial_data, ignore_index=True)
trial_df.to_csv(os.path.join(OUTPUT_DIR, "single_trial_rpe_erp.csv"), index=False)
print(f"\n  Total trials with RPEs: {len(trial_df)}")

from collections import Counter
print(f"  Best model counts:")
for m, c in Counter(best_models.values()).most_common():
    print(f"    {m:<15} {c} subjects")

# =====================================================================
# STEP 3: Within-subject single-trial RPE-ERP correlations
# =====================================================================
print(f"\n{'=' * 60}")
print("STEP 3: Within-subject RPE-ERP correlations")
print("=" * 60)

rpe_types = ["rpe_best", "rpe_rw", "abs_rpe_best", "abs_rpe_rw"]
channels = ["FCz", "Cz", "Pz", "FC1", "FC2"]  # Key frontocentral channels

subject_correlations = []

for subj in subjects:
    sdata = trial_df[trial_df["SubjectCode"] == subj]

    for rpe_type in rpe_types:
        rpe_vals = sdata[rpe_type].values

        for ch in ERP_CHANNELS:
            erp_vals = sdata[ch].values
            mask = np.isfinite(rpe_vals) & np.isfinite(erp_vals)

            if mask.sum() >= 20:
                r, p = pearsonr(rpe_vals[mask], erp_vals[mask])
                # Fisher z-transform
                z = np.arctanh(np.clip(r, -0.999, 0.999))
                subject_correlations.append({
                    "subject": subj,
                    "rpe_type": rpe_type,
                    "channel": ch,
                    "r": r, "p": p, "z": z,
                    "n_trials": int(mask.sum()),
                    "best_model": best_models[subj],
                })

    # Also: split by reward/no-reward
    for ch in ["FCz", "Cz"]:
        rew_mask = sdata["outcome"] == 1
        norew_mask = sdata["outcome"] == 0

        for condition, cond_mask, label in [
            ("reward", rew_mask, "Reward trials"),
            ("noreward", norew_mask, "No-reward trials"),
        ]:
            sub = sdata[cond_mask]
            if len(sub) >= 15:
                r, p = pearsonr(sub["rpe_rw"].values, sub[ch].values)
                z = np.arctanh(np.clip(r, -0.999, 0.999))
                subject_correlations.append({
                    "subject": subj,
                    "rpe_type": f"rpe_rw_{condition}",
                    "channel": ch,
                    "r": r, "p": p, "z": z,
                    "n_trials": len(sub),
                    "best_model": best_models[subj],
                })

corr_df = pd.DataFrame(subject_correlations)
corr_df.to_csv(os.path.join(OUTPUT_DIR, "within_subject_correlations.csv"), index=False)

# =====================================================================
# STEP 4: Group-level statistics (t-test on Fisher-z values)
# =====================================================================
print(f"\n{'=' * 60}")
print("STEP 4: Group-level RPE-ERP statistics")
print("=" * 60)

print(f"\n  {'RPE Type':<20} {'Channel':<8} {'Mean r':>8} {'Mean z':>8} "
      f"{'t':>7} {'p':>9} {'sig':>4}")
print("  " + "-" * 68)

group_results = []

for rpe_type in rpe_types + ["rpe_rw_reward", "rpe_rw_noreward"]:
    for ch in ["FCz", "Cz", "Pz", "FC1", "FC2"]:
        sub_corrs = corr_df[(corr_df["rpe_type"] == rpe_type) &
                             (corr_df["channel"] == ch)]
        if len(sub_corrs) >= 10:
            z_vals = sub_corrs["z"].values
            r_vals = sub_corrs["r"].values

            # One-sample t-test: is mean z != 0?
            t_stat, p_val = ttest_1samp(z_vals, 0)
            # Also Wilcoxon (non-parametric)
            try:
                w_stat, w_p = wilcoxon(z_vals)
            except:
                w_stat, w_p = np.nan, np.nan

            sig = "**" if p_val < 0.01 else ("*" if p_val < 0.05 else (
                "~" if p_val < 0.10 else ""))

            group_results.append({
                "rpe_type": rpe_type, "channel": ch,
                "mean_r": np.mean(r_vals), "mean_z": np.mean(z_vals),
                "sd_z": np.std(z_vals), "t": t_stat, "p": p_val,
                "wilcoxon_p": w_p, "n_subjects": len(sub_corrs),
                "pct_positive": np.mean(r_vals > 0) * 100,
            })

            if ch in ["FCz", "Cz"]:
                print(f"  {rpe_type:<20} {ch:<8} {np.mean(r_vals):>8.4f} "
                      f"{np.mean(z_vals):>8.4f} {t_stat:>7.2f} "
                      f"{p_val:>9.4f} {sig:>4}")

group_df = pd.DataFrame(group_results)
group_df.to_csv(os.path.join(OUTPUT_DIR, "group_level_results.csv"), index=False)

# Print significant results
sig_results = group_df[group_df["p"] < 0.10].sort_values("p")
if len(sig_results) > 0:
    print(f"\n  All significant/trending results (p < .10):")
    for _, row in sig_results.iterrows():
        sig = "**" if row["p"] < 0.01 else "*" if row["p"] < 0.05 else "~"
        print(f"    {sig} {row['rpe_type']:<20} @ {row['channel']:<6} "
              f"r={row['mean_r']:.4f}, t({int(row['n_subjects'])-1})={row['t']:.2f}, "
              f"p={row['p']:.4f}, {row['pct_positive']:.0f}% positive")

# =====================================================================
# STEP 5: Figures
# =====================================================================
print(f"\n{'=' * 60}")
print("STEP 5: Generating figures")
print("=" * 60)

fig, axes = plt.subplots(2, 4, figsize=(22, 11))

# 5a: Distribution of within-subject r (RPE_rw × FCz)
ax = axes[0, 0]
r_vals = corr_df[(corr_df["rpe_type"] == "rpe_rw") &
                  (corr_df["channel"] == "FCz")]["r"].values
ax.hist(r_vals, bins=20, color="steelblue", edgecolor="black", alpha=0.7)
ax.axvline(0, ls="--", color="red", linewidth=2)
ax.axvline(np.mean(r_vals), ls="-", color="black", linewidth=2,
           label=f"Mean r={np.mean(r_vals):.4f}")
ax.set_xlabel("Within-subject Pearson r")
ax.set_ylabel("# Subjects")
ax.set_title("RPE(RW) × FCz: Subject r Distribution")
ax.legend()

# 5b: Distribution of r (|RPE| × FCz)
ax = axes[0, 1]
r_vals_abs = corr_df[(corr_df["rpe_type"] == "abs_rpe_rw") &
                      (corr_df["channel"] == "FCz")]["r"].values
ax.hist(r_vals_abs, bins=20, color="coral", edgecolor="black", alpha=0.7)
ax.axvline(0, ls="--", color="red", linewidth=2)
ax.axvline(np.mean(r_vals_abs), ls="-", color="black", linewidth=2,
           label=f"Mean r={np.mean(r_vals_abs):.4f}")
ax.set_xlabel("Within-subject Pearson r")
ax.set_ylabel("# Subjects")
ax.set_title("|RPE(RW)| × FCz: Subject r Distribution")
ax.legend()

# 5c: Scalp topography of RPE-ERP correlation
ax = axes[0, 2]
topo_data = {}
for ch in ERP_CHANNELS:
    sub = corr_df[(corr_df["rpe_type"] == "rpe_rw") & (corr_df["channel"] == ch)]
    if len(sub) > 0:
        topo_data[ch] = sub["r"].mean()

channels_plot = list(topo_data.keys())
r_values = [topo_data[ch] for ch in channels_plot]
colors = plt.cm.RdBu_r([(r + 0.15) / 0.30 for r in r_values])
bars = ax.barh(channels_plot, r_values, color=colors, edgecolor="black")
ax.axvline(0, ls="--", color="black")
ax.set_xlabel("Mean within-subject r (RPE × ERP)")
ax.set_title("RPE-ERP Correlation by Channel")

# 5d: Same for |RPE|
ax = axes[0, 3]
topo_abs = {}
for ch in ERP_CHANNELS:
    sub = corr_df[(corr_df["rpe_type"] == "abs_rpe_rw") & (corr_df["channel"] == ch)]
    if len(sub) > 0:
        topo_abs[ch] = sub["r"].mean()
channels_plot = list(topo_abs.keys())
r_values_abs = [topo_abs[ch] for ch in channels_plot]
colors = plt.cm.RdBu_r([(r + 0.15) / 0.30 for r in r_values_abs])
ax.barh(channels_plot, r_values_abs, color=colors, edgecolor="black")
ax.axvline(0, ls="--", color="black")
ax.set_xlabel("Mean within-subject r (|RPE| × ERP)")
ax.set_title("|RPE|-ERP Correlation by Channel")

# 5e: Example single subject scatter (RPE × FCz)
ax = axes[1, 0]
# Pick subject with strongest correlation
best_subj_row = corr_df[(corr_df["rpe_type"] == "rpe_rw") &
                          (corr_df["channel"] == "FCz")].sort_values("p").iloc[0]
best_subj = best_subj_row["subject"]
sdata = trial_df[trial_df["SubjectCode"] == best_subj]
x, y = sdata["rpe_rw"].values, sdata["FCz"].values
r_ex, p_ex = pearsonr(x, y)
ax.scatter(x, y, alpha=0.4, s=20, color="teal")
z = np.polyfit(x, y, 1)
xline = np.linspace(x.min(), x.max(), 100)
ax.plot(xline, np.polyval(z, xline), "k-", linewidth=2)
ax.set_xlabel("RPE (RW model)")
ax.set_ylabel("FCz amplitude (uV)")
ax.set_title(f"Best subject: {best_subj}\n(r={r_ex:.3f}, p={p_ex:.4f}, n={len(x)})")

# 5f: Reward vs no-reward trials (RPE × FCz)
ax = axes[1, 1]
rew_corrs = corr_df[(corr_df["rpe_type"] == "rpe_rw_reward") &
                     (corr_df["channel"] == "FCz")]["r"].values
norew_corrs = corr_df[(corr_df["rpe_type"] == "rpe_rw_noreward") &
                       (corr_df["channel"] == "FCz")]["r"].values
bp = ax.boxplot([rew_corrs, norew_corrs],
                labels=["Reward\ntrials", "No-Reward\ntrials"],
                patch_artist=True)
bp["boxes"][0].set_facecolor("green")
bp["boxes"][0].set_alpha(0.5)
bp["boxes"][1].set_facecolor("red")
bp["boxes"][1].set_alpha(0.5)
ax.axhline(0, ls="--", color="black", alpha=0.3)
ax.set_ylabel("Within-subject r (RPE × FCz)")
ax.set_title("RPE-FCz Correlation\nby Outcome Type")

# 5g: Group BIC comparison
ax = axes[1, 2]
fits_df = pd.DataFrame(subject_fits)
group_bic = fits_df.groupby("model")["bic"].sum().sort_values()
delta = [group_bic[m] - group_bic.iloc[0] for m in group_bic.index]
colors = ["green" if d == 0 else "orange" if d < 100 else "gray" for d in delta]
ax.barh(group_bic.index, delta, color=colors)
ax.set_xlabel("Delta-BIC (vs best)")
ax.set_title("Model Comparison\n(single-trial ERP subjects)")

# 5h: Grand average RPE × FCz (all trials pooled, binned)
ax = axes[1, 3]
# Bin RPEs into quantiles and plot mean ERP
all_rpes = trial_df["rpe_rw"].values
all_fcz = trial_df["FCz"].values
n_bins = 10
bin_edges = np.percentile(all_rpes, np.linspace(0, 100, n_bins + 1))
bin_centers = []
bin_means = []
bin_sems = []
for b in range(n_bins):
    mask = (all_rpes >= bin_edges[b]) & (all_rpes < bin_edges[b + 1])
    if b == n_bins - 1:
        mask = (all_rpes >= bin_edges[b]) & (all_rpes <= bin_edges[b + 1])
    if mask.sum() > 0:
        bin_centers.append(np.mean(all_rpes[mask]))
        bin_means.append(np.mean(all_fcz[mask]))
        bin_sems.append(np.std(all_fcz[mask]) / np.sqrt(mask.sum()))

ax.errorbar(bin_centers, bin_means, yerr=bin_sems,
            fmt="o-", color="teal", capsize=3, linewidth=2, markersize=6)
# Fit line
z = np.polyfit(bin_centers, bin_means, 1)
xfit = np.linspace(min(bin_centers), max(bin_centers), 100)
ax.plot(xfit, np.polyval(z, xfit), "k--", linewidth=1.5, alpha=0.5)
r_grand, p_grand = pearsonr(all_rpes, all_fcz)
ax.set_xlabel("RPE (RW model, binned)")
ax.set_ylabel("Mean FCz amplitude (uV)")
ax.set_title(f"Grand Average: RPE × FCz\n(r={r_grand:.4f}, p={p_grand:.4f}, n={len(all_rpes)})")

plt.suptitle("Single-Trial RPE × ERP Analysis (N=43)", fontsize=16, y=1.02)
plt.tight_layout()
fig_path = os.path.join(OUTPUT_DIR, "08_single_trial_rpe_erp.png")
plt.savefig(fig_path, dpi=150, bbox_inches="tight")
print(f"  Main figure: {fig_path}")
plt.close()

# ── Heatmap: RPE type × Channel ─────────────────────────────────
fig2, ax2 = plt.subplots(figsize=(12, 5))
rpe_labels = ["rpe_rw", "abs_rpe_rw", "rpe_best", "abs_rpe_best"]
ch_labels = ERP_CHANNELS

mat_r = np.zeros((len(rpe_labels), len(ch_labels)))
mat_p = np.zeros_like(mat_r)

for i, rpe in enumerate(rpe_labels):
    for j, ch in enumerate(ch_labels):
        sub = corr_df[(corr_df["rpe_type"] == rpe) & (corr_df["channel"] == ch)]
        if len(sub) >= 10:
            z_vals = sub["z"].values
            _, p_val = ttest_1samp(z_vals, 0)
            mat_r[i, j] = np.mean(sub["r"].values)
            mat_p[i, j] = p_val

im = ax2.imshow(mat_r, cmap="RdBu_r", vmin=-0.06, vmax=0.06, aspect="auto")
ax2.set_xticks(range(len(ch_labels)))
ax2.set_xticklabels(ch_labels, rotation=45, ha="right")
ax2.set_yticks(range(len(rpe_labels)))
ax2.set_yticklabels(["RPE (RW)", "|RPE| (RW)", "RPE (best)", "|RPE| (best)"])

for i in range(len(rpe_labels)):
    for j in range(len(ch_labels)):
        sig = "**" if mat_p[i, j] < 0.01 else ("*" if mat_p[i, j] < 0.05 else "")
        ax2.text(j, i, f"{mat_r[i, j]:.3f}{sig}", ha="center", va="center",
                 fontsize=8, color="white" if abs(mat_r[i, j]) > 0.03 else "black")

plt.colorbar(im, label="Mean within-subject r")
ax2.set_title("Single-Trial RPE × ERP: Channel × RPE Type (group-level t-test)", fontsize=13)
plt.tight_layout()
hm_path = os.path.join(OUTPUT_DIR, "08_rpe_erp_heatmap.png")
plt.savefig(hm_path, dpi=150, bbox_inches="tight")
print(f"  Heatmap: {hm_path}")
plt.close()

# =====================================================================
# SUMMARY
# =====================================================================
print(f"\n{'=' * 60}")
print("SUMMARY")
print("=" * 60)
print(f"  Subjects: {len(subjects)}")
print(f"  Total single trials: {len(trial_df)}")
print(f"  Best group model: {fits_df.groupby('model')['bic'].sum().idxmin()}")

# Top findings
top = group_df.sort_values("p").head(5)
print(f"\n  Top 5 RPE-ERP effects:")
for _, row in top.iterrows():
    sig = "**" if row["p"] < 0.01 else ("*" if row["p"] < 0.05 else "~" if row["p"] < 0.10 else "")
    print(f"    {sig:>2} {row['rpe_type']:<20} @ {row['channel']:<6} "
          f"mean_r={row['mean_r']:.4f}, t={row['t']:.2f}, p={row['p']:.4f}")

print(f"\n  Output: {OUTPUT_DIR}")
print(f"    single_trial_rpe_erp.csv       — {len(trial_df)} trials with RPEs + ERP")
print(f"    within_subject_correlations.csv — per-subject r values")
print(f"    group_level_results.csv        — group statistics")
