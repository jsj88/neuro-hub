"""
Example 9: Full T-Maze AR Pipeline — RPE × ERP × Reward Magnitude × Stay/Shift.

Correct variable definitions:
- Vel_Cond: 1=High Reward ($0.25), 2=No Reward ($0.00), 3=Low Reward ($0.05)
- PES_PRS_index: 0=first, 1=stay, 2=shift
- Condition_Direction_index: 0=first, 1=rew+stay, 2=rew+shift, 3=norew+stay, 4=norew+shift

Steps:
1. Fit reward learning models (binary AND magnitude outcomes)
2. Single-trial RPE × ERP correlations by reward level
3. High vs Low reward RPE-ERP comparison
4. Stay/Shift behavioral adjustments × ERP
5. Post-outcome walking speed as function of RPE
6. Summary table for paper

Usage:
    cd ~/neuro-hub
    python examples/09_tmaze_full_pipeline.py
"""

import sys, os, re, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr, ttest_1samp, ttest_ind, ttest_rel, wilcoxon
from collections import Counter
from agent.tools.model_fitting import fit_model, MODEL_SPECS, extract_trial_variables

# ── Paths ────────────────────────────────────────────────────────
DATA_FILE = "/Volumes/LaCie_2/ARmaze_v2/CSVFiles/BEH_RESULTS/merged_clean_V3_scaled.csv"
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "tmaze_full_pipeline")
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODELS = ["random", "wsls", "rw", "q_learning", "q_dual",
          "actor_critic", "q_decay", "rw_bias", "ck", "rwck"]

# =====================================================================
# STEP 1: Load and prepare data
# =====================================================================
print("=" * 70)
print("STEP 1: Loading data with correct Vel_Cond coding")
print("=" * 70)

df = pd.read_csv(DATA_FILE)

# Recode for model fitting
df["choice"] = df["Direction"] - 1          # 0=left, 1=right
df["outcome_binary"] = (df["Reward"] == 1).astype(float)  # 1=reward, 0=noreward
df["outcome_mag"] = df["Earned"]            # $0.25, $0.05, or $0.00

# Correct Vel_Cond labels
df["reward_level"] = df["Vel_Cond"].map({
    1: "high_rew", 2: "no_rew", 3: "low_rew"
})
df["reward_level_num"] = df["Vel_Cond"].map({1: 2, 2: 0, 3: 1})  # 0=none, 1=low, 2=high

# Other labels
df["stay_shift"] = df["PES_PRS_index"].map({0: "first", 1: "stay", 2: "shift"})
df["cond_dir"] = df["Condition_Direction_index"].map({
    0: "first", 1: "rew_stay", 2: "rew_shift",
    3: "norew_stay", 4: "norew_shift"
})
df["is_reward"] = (df["Reward"] == 1).astype(int)

subjects = sorted(df["SubjectCode"].unique())
print(f"  {len(df)} trials, {len(subjects)} subjects")
print(f"\n  Reward level distribution:")
for vc, label, earned in [(1, "High Reward", "$0.25"), (2, "No Reward", "$0.00"),
                           (3, "Low Reward", "$0.05")]:
    n = len(df[df["Vel_Cond"] == vc])
    rt = df[df["Vel_Cond"] == vc]["Start_RT"].median()
    print(f"    VC={vc} {label:<12} ({earned}): n={n}, median RT={rt:.3f}s")
print(f"\n  Stay/Shift: {df['stay_shift'].value_counts().to_dict()}")

# =====================================================================
# STEP 2: Fit models — binary AND magnitude outcomes
# =====================================================================
print(f"\n{'=' * 70}")
print("STEP 2: Fitting models (binary + magnitude RPEs)")
print("=" * 70)

all_trial_data = []
subject_info = []

for i, subj in enumerate(subjects):
    sdata = df[df["SubjectCode"] == subj].sort_values("Trial").copy()
    choices = sdata["choice"].values.astype(int)
    outcomes_bin = sdata["outcome_binary"].values.astype(float)
    outcomes_mag = sdata["outcome_mag"].values.astype(float)

    # Fit all models (binary outcome)
    results = {}
    for model_name in MODELS:
        try:
            r = fit_model(model_name, choices, outcomes_bin,
                          n_options=2, method="de", seed=42)
            results[model_name] = r
        except:
            pass

    # Best model by BIC
    best_name = min(results, key=lambda m: results[m].bic)
    best_r = results[best_name]
    rw_r = results.get("rw", None)

    # Extract RPEs — binary outcome
    if rw_r:
        rw_latent = extract_trial_variables("rw", rw_r.params,
                                             choices, outcomes_bin, n_options=2)
        sdata["rpe_binary"] = rw_latent["rpes"]
        sdata["abs_rpe_binary"] = np.abs(rw_latent["rpes"])
        if "q_values" in rw_latent:
            sdata["chosen_q_binary"] = rw_latent["q_values"][
                np.arange(len(choices)), choices]

    # Fit RW with magnitude outcomes (Earned: 0, 0.05, 0.25)
    try:
        rw_mag = fit_model("rw", choices, outcomes_mag,
                           n_options=2, method="de", seed=42)
        mag_latent = extract_trial_variables("rw", rw_mag.params,
                                              choices, outcomes_mag, n_options=2)
        sdata["rpe_mag"] = mag_latent["rpes"]
        sdata["abs_rpe_mag"] = np.abs(mag_latent["rpes"])
        if "q_values" in mag_latent:
            sdata["chosen_q_mag"] = mag_latent["q_values"][
                np.arange(len(choices)), choices]
        rw_mag_alpha = rw_mag.params.get("alpha", np.nan)
    except:
        sdata["rpe_mag"] = np.nan
        sdata["abs_rpe_mag"] = np.nan
        rw_mag_alpha = np.nan

    sdata["best_model"] = best_name
    all_trial_data.append(sdata)

    subject_info.append({
        "subject": subj, "best_model": best_name,
        "best_bic": best_r.bic,
        "rw_alpha_binary": rw_r.params.get("alpha", np.nan) if rw_r else np.nan,
        "rw_beta_binary": rw_r.params.get("beta", np.nan) if rw_r else np.nan,
        "rw_alpha_mag": rw_mag_alpha,
        "n_trials": len(sdata),
        "reward_rate": outcomes_bin.mean(),
        "mean_earned": outcomes_mag.mean(),
    })

    if (i + 1) % 10 == 0 or i == 0:
        print(f"  {subj} ({i+1}/{len(subjects)}): best={best_name}, "
              f"RW_bin alpha={rw_r.params.get('alpha',0):.3f}, "
              f"RW_mag alpha={rw_mag_alpha:.3f}")

trial_df = pd.concat(all_trial_data, ignore_index=True)
info_df = pd.DataFrame(subject_info)

print(f"\n  Best model counts:")
for m, c in Counter(info_df["best_model"]).most_common():
    print(f"    {m:<15} {c} subjects ({c/len(subjects)*100:.0f}%)")

# =====================================================================
# STEP 3: RPE characteristics by reward level
# =====================================================================
print(f"\n{'=' * 70}")
print("STEP 3: RPE characteristics by reward level")
print("=" * 70)

print(f"\n  {'Condition':<15} {'RPE_bin':>9} {'|RPE_bin|':>10} {'RPE_mag':>9} "
      f"{'|RPE_mag|':>10} {'FCz':>7} {'N':>6}")
print("  " + "-" * 70)
for vc, label in [(1, "High Rew"), (2, "No Rew"), (3, "Low Rew")]:
    sub = trial_df[trial_df["Vel_Cond"] == vc]
    print(f"  {label:<15} {sub['rpe_binary'].mean():>9.4f} "
          f"{sub['abs_rpe_binary'].mean():>10.4f} "
          f"{sub['rpe_mag'].mean():>9.4f} "
          f"{sub['abs_rpe_mag'].mean():>10.4f} "
          f"{sub['FCz'].mean():>7.3f} {len(sub):>6}")

# =====================================================================
# STEP 4: Single-trial RPE × ERP by reward level
# =====================================================================
print(f"\n{'=' * 70}")
print("STEP 4: Single-trial RPE × ERP correlations by reward level")
print("=" * 70)

channels = ["FCz", "Cz", "FC2", "FC1", "F4", "F5",
            "C4", "C3", "P4", "P3", "P8", "P7", "Pz"]

conditions = {
    "all": lambda d: d,
    "high_rew": lambda d: d[d["Vel_Cond"] == 1],
    "no_rew": lambda d: d[d["Vel_Cond"] == 2],
    "low_rew": lambda d: d[d["Vel_Cond"] == 3],
    "any_rew": lambda d: d[d["is_reward"] == 1],
    "stay": lambda d: d[d["stay_shift"] == "stay"],
    "shift": lambda d: d[d["stay_shift"] == "shift"],
}

rpe_types = ["rpe_binary", "rpe_mag"]

all_corrs = []
for subj in subjects:
    sdata = trial_df[trial_df["SubjectCode"] == subj]

    for rpe_type in rpe_types:
        if rpe_type not in sdata.columns or sdata[rpe_type].isna().all():
            continue
        for cond_name, cond_fn in conditions.items():
            subset = cond_fn(sdata)
            if len(subset) < 15:
                continue
            for ch in channels:
                rpe_vals = subset[rpe_type].values
                erp_vals = subset[ch].values
                mask = np.isfinite(rpe_vals) & np.isfinite(erp_vals)
                if mask.sum() >= 15:
                    r, p = pearsonr(rpe_vals[mask], erp_vals[mask])
                    z = np.arctanh(np.clip(r, -0.999, 0.999))
                    all_corrs.append({
                        "subject": subj, "rpe_type": rpe_type,
                        "condition": cond_name, "channel": ch,
                        "r": r, "p": p, "z": z,
                        "n_trials": int(mask.sum()),
                    })

corr_df = pd.DataFrame(all_corrs)

# Group-level t-tests
print(f"\n  {'RPE':<12} {'Condition':<12} {'Ch':<5} {'Mean r':>8} {'t':>7} "
      f"{'p':>9} {'sig':>4} {'%pos':>5}")
print("  " + "-" * 65)

group_results = []
for rpe_type in rpe_types:
    for cond_name in ["all", "high_rew", "no_rew", "low_rew", "any_rew",
                       "stay", "shift"]:
        for ch in ["FCz", "Cz"]:
            sub = corr_df[(corr_df["rpe_type"] == rpe_type) &
                           (corr_df["condition"] == cond_name) &
                           (corr_df["channel"] == ch)]
            if len(sub) >= 10:
                z_vals = sub["z"].values
                t_stat, p_val = ttest_1samp(z_vals, 0)
                sig = "**" if p_val < 0.01 else ("*" if p_val < 0.05 else (
                    "~" if p_val < 0.10 else ""))
                pct_pos = np.mean(sub["r"].values > 0) * 100
                group_results.append({
                    "rpe_type": rpe_type, "condition": cond_name,
                    "channel": ch, "mean_r": sub["r"].mean(),
                    "mean_z": sub["z"].mean(), "t": t_stat, "p": p_val,
                    "n_subjects": len(sub), "pct_positive": pct_pos,
                })
                rpe_label = "binary" if "binary" in rpe_type else "magnitude"
                if ch == "FCz" or p_val < 0.10:
                    print(f"  {rpe_label:<12} {cond_name:<12} {ch:<5} "
                          f"{sub['r'].mean():>8.4f} {t_stat:>7.2f} "
                          f"{p_val:>9.4f} {sig:>4} {pct_pos:>5.0f}%")

gresults_df = pd.DataFrame(group_results)

# =====================================================================
# STEP 5: High vs Low vs No Reward comparisons
# =====================================================================
print(f"\n{'=' * 70}")
print("STEP 5: Reward level comparisons")
print("=" * 70)

# RPE-FCz coupling: high vs low reward (paired)
for rpe_type in rpe_types:
    rpe_label = "binary" if "binary" in rpe_type else "magnitude"
    print(f"\n  {rpe_label} RPE-FCz coupling by reward level (paired t-tests):")
    for ch in ["FCz", "Cz"]:
        high_z = corr_df[(corr_df["rpe_type"] == rpe_type) &
                          (corr_df["condition"] == "high_rew") &
                          (corr_df["channel"] == ch)]
        low_z = corr_df[(corr_df["rpe_type"] == rpe_type) &
                          (corr_df["condition"] == "low_rew") &
                          (corr_df["channel"] == ch)]
        norew_z = corr_df[(corr_df["rpe_type"] == rpe_type) &
                           (corr_df["condition"] == "no_rew") &
                           (corr_df["channel"] == ch)]

        # High vs Low
        common = set(high_z["subject"]) & set(low_z["subject"])
        if len(common) >= 10:
            h = high_z[high_z["subject"].isin(common)].set_index("subject")["z"]
            l = low_z[low_z["subject"].isin(common)].set_index("subject")["z"]
            cs = sorted(common)
            t, p = ttest_rel(h.loc[cs].values, l.loc[cs].values)
            sig = "*" if p < 0.05 else ("~" if p < 0.10 else "")
            print(f"    {ch} High vs Low: t({len(cs)-1})={t:.2f}, p={p:.4f} {sig} "
                  f"(high r={high_z['r'].mean():.4f}, low r={low_z['r'].mean():.4f})")

        # High vs NoRew
        common2 = set(high_z["subject"]) & set(norew_z["subject"])
        if len(common2) >= 10:
            h2 = high_z[high_z["subject"].isin(common2)].set_index("subject")["z"]
            n2 = norew_z[norew_z["subject"].isin(common2)].set_index("subject")["z"]
            cs2 = sorted(common2)
            t2, p2 = ttest_rel(h2.loc[cs2].values, n2.loc[cs2].values)
            sig2 = "*" if p2 < 0.05 else ("~" if p2 < 0.10 else "")
            print(f"    {ch} High vs NoRew: t({len(cs2)-1})={t2:.2f}, p={p2:.4f} {sig2}")

# ERP amplitude by reward level
print(f"\n  Mean FCz amplitude by reward level:")
for vc, label in [(1, "High Rew ($0.25)"), (2, "No Rew ($0.00)"), (3, "Low Rew ($0.05)")]:
    sub = trial_df[trial_df["Vel_Cond"] == vc]
    print(f"    {label:<20} FCz={sub['FCz'].mean():.3f} (SD={sub['FCz'].std():.3f})")

# Paired t-tests on ERP
subj_erp = []
for subj in subjects:
    sdata = trial_df[trial_df["SubjectCode"] == subj]
    high = sdata[sdata["Vel_Cond"] == 1]["FCz"]
    low = sdata[sdata["Vel_Cond"] == 3]["FCz"]
    norew = sdata[sdata["Vel_Cond"] == 2]["FCz"]
    if len(high) >= 5 and len(low) >= 5 and len(norew) >= 5:
        subj_erp.append({
            "subject": subj,
            "fcz_high": high.mean(), "fcz_low": low.mean(), "fcz_norew": norew.mean(),
        })
erp_df = pd.DataFrame(subj_erp)
if len(erp_df) >= 10:
    t_hl, p_hl = ttest_rel(erp_df["fcz_high"], erp_df["fcz_low"])
    t_hn, p_hn = ttest_rel(erp_df["fcz_high"], erp_df["fcz_norew"])
    t_ln, p_ln = ttest_rel(erp_df["fcz_low"], erp_df["fcz_norew"])
    print(f"\n  Paired t-tests (FCz):")
    print(f"    High vs Low:   t={t_hl:.2f}, p={p_hl:.4f}")
    print(f"    High vs NoRew: t={t_hn:.2f}, p={p_hn:.4f}")
    print(f"    Low vs NoRew:  t={t_ln:.2f}, p={p_ln:.4f}")

# =====================================================================
# STEP 6: Stay/Shift × Reward Level × ERP
# =====================================================================
print(f"\n{'=' * 70}")
print("STEP 6: Stay/Shift × ERP with correct coding")
print("=" * 70)

print(f"\n  FCz by Outcome × Stay/Shift:")
for cond in ["rew_stay", "rew_shift", "norew_stay", "norew_shift"]:
    sub = trial_df[trial_df["cond_dir"] == cond]
    if len(sub) > 0:
        print(f"    {cond:<15} FCz={sub['FCz'].mean():>7.3f}, n={len(sub)}")

# =====================================================================
# STEP 7: Post-outcome walking speed adjustment
# =====================================================================
print(f"\n{'=' * 70}")
print("STEP 7: Post-outcome walking speed by reward level")
print("=" * 70)

rt_adj = []
for subj in subjects:
    sdata = trial_df[trial_df["SubjectCode"] == subj].sort_values("Trial")
    rts = sdata["Start_RT"].values
    rew_levels = sdata["reward_level"].values
    rpes = sdata["rpe_mag"].values if "rpe_mag" in sdata.columns else None

    for t in range(1, len(rts)):
        if rts[t] < 20 and rts[t-1] < 20:
            rt_adj.append({
                "subject": subj,
                "rt_change": rts[t] - rts[t-1],
                "current_rt": rts[t],
                "prev_reward_level": rew_levels[t-1],
                "prev_rpe_mag": rpes[t-1] if rpes is not None else np.nan,
            })

rt_adj_df = pd.DataFrame(rt_adj)

print(f"\n  Walking RT change by previous reward level:")
for level in ["high_rew", "low_rew", "no_rew"]:
    sub = rt_adj_df[rt_adj_df["prev_reward_level"] == level]
    print(f"    After {level:<10} RT change={sub['rt_change'].mean():>+.4f}s "
          f"(SD={sub['rt_change'].std():.3f})")

# RPE(mag) → RT change
rpe_rt_corrs = []
for subj in subjects:
    sub = rt_adj_df[rt_adj_df["subject"] == subj]
    if len(sub) >= 20 and sub["prev_rpe_mag"].notna().sum() >= 20:
        vals = sub.dropna(subset=["prev_rpe_mag"])
        r, p = pearsonr(vals["prev_rpe_mag"].values, vals["rt_change"].values)
        rpe_rt_corrs.append({"subject": subj, "r": r, "p": p,
                              "z": np.arctanh(np.clip(r, -0.999, 0.999))})
if rpe_rt_corrs:
    rpe_rt_df = pd.DataFrame(rpe_rt_corrs)
    t, p = ttest_1samp(rpe_rt_df["z"].values, 0)
    print(f"\n  RPE_mag(t-1) → RT_change(t): mean r={rpe_rt_df['r'].mean():.4f}, "
          f"t({len(rpe_rt_df)-1})={t:.2f}, p={p:.4f}")

# =====================================================================
# STEP 8: Figures (REPLACE existing)
# =====================================================================
print(f"\n{'=' * 70}")
print("STEP 8: Generating figures (replacing existing)")
print("=" * 70)

fig, axes = plt.subplots(3, 4, figsize=(24, 16))

# Colors for reward levels
C_HIGH = "#2ca02c"   # green
C_LOW = "#ff7f0e"    # orange
C_NOREW = "#d62728"  # red

# ── 8a: Model comparison ────────────────────────────────────────
ax = axes[0, 0]
model_counts = Counter(info_df["best_model"])
models_sorted = sorted(model_counts.keys(), key=lambda x: -model_counts[x])
ax.bar(range(len(models_sorted)), [model_counts[m] for m in models_sorted],
       color="steelblue")
ax.set_xticks(range(len(models_sorted)))
ax.set_xticklabels(models_sorted, rotation=45, ha="right", fontsize=9)
ax.set_ylabel("# Subjects")
ax.set_title("A. Best-Fitting Model per Subject")

# ── 8b: RPE-FCz by reward level (binary RPE) ────────────────────
ax = axes[0, 1]
conds_plot = ["all", "high_rew", "low_rew", "no_rew", "any_rew"]
labels_plot = ["All", "High\nRew", "Low\nRew", "No\nRew", "Any\nRew"]
colors_plot = ["gray", C_HIGH, C_LOW, C_NOREW, "steelblue"]
means, sems = [], []
for c in conds_plot:
    sub = corr_df[(corr_df["rpe_type"] == "rpe_binary") &
                   (corr_df["condition"] == c) & (corr_df["channel"] == "FCz")]
    means.append(sub["r"].mean() if len(sub) > 0 else 0)
    sems.append(sub["r"].std() / np.sqrt(len(sub)) if len(sub) > 0 else 0)
ax.bar(range(len(conds_plot)), means, yerr=sems, color=colors_plot,
       alpha=0.8, capsize=4, edgecolor="black")
ax.axhline(0, ls="--", color="black", alpha=0.3)
ax.set_xticks(range(len(conds_plot)))
ax.set_xticklabels(labels_plot, fontsize=9)
ax.set_ylabel("Mean r (RPE × FCz)")
ax.set_title("B. Binary RPE-FCz by Reward Level")

# ── 8c: RPE-FCz by reward level (magnitude RPE) ─────────────────
ax = axes[0, 2]
means_m, sems_m = [], []
for c in conds_plot:
    sub = corr_df[(corr_df["rpe_type"] == "rpe_mag") &
                   (corr_df["condition"] == c) & (corr_df["channel"] == "FCz")]
    means_m.append(sub["r"].mean() if len(sub) > 0 else 0)
    sems_m.append(sub["r"].std() / np.sqrt(len(sub)) if len(sub) > 0 else 0)
ax.bar(range(len(conds_plot)), means_m, yerr=sems_m, color=colors_plot,
       alpha=0.8, capsize=4, edgecolor="black")
ax.axhline(0, ls="--", color="black", alpha=0.3)
ax.set_xticks(range(len(conds_plot)))
ax.set_xticklabels(labels_plot, fontsize=9)
ax.set_ylabel("Mean r (RPE × FCz)")
ax.set_title("C. Magnitude RPE-FCz by Reward Level")

# ── 8d: FCz amplitude by reward level ───────────────────────────
ax = axes[0, 3]
erp_means = [erp_df["fcz_high"].mean(), erp_df["fcz_low"].mean(),
             erp_df["fcz_norew"].mean()]
erp_sems = [erp_df["fcz_high"].std() / np.sqrt(len(erp_df)),
            erp_df["fcz_low"].std() / np.sqrt(len(erp_df)),
            erp_df["fcz_norew"].std() / np.sqrt(len(erp_df))]
ax.bar([0, 1, 2], erp_means, yerr=erp_sems,
       color=[C_HIGH, C_LOW, C_NOREW], alpha=0.8, capsize=4, edgecolor="black")
ax.set_xticks([0, 1, 2])
ax.set_xticklabels(["High Rew\n($0.25)", "Low Rew\n($0.05)", "No Rew\n($0.00)"],
                    fontsize=9)
ax.set_ylabel("FCz amplitude (µV)")
ax.set_title("D. FCz by Reward Level")

# ── 8e: Grand average binned RPE × FCz ──────────────────────────
ax = axes[1, 0]
rpes_all = trial_df["rpe_binary"].values
fcz_all = trial_df["FCz"].values
mask = np.isfinite(rpes_all) & np.isfinite(fcz_all)
rpes_c, fcz_c = rpes_all[mask], fcz_all[mask]
n_bins = 10
edges = np.percentile(rpes_c, np.linspace(0, 100, n_bins + 1))
centers, bmeans, bsems = [], [], []
for b in range(n_bins):
    m = (rpes_c >= edges[b]) & (rpes_c <= edges[b + 1]) if b == n_bins - 1 \
        else (rpes_c >= edges[b]) & (rpes_c < edges[b + 1])
    if m.sum() > 0:
        centers.append(rpes_c[m].mean())
        bmeans.append(fcz_c[m].mean())
        bsems.append(fcz_c[m].std() / np.sqrt(m.sum()))
ax.errorbar(centers, bmeans, yerr=bsems, fmt="o-", color="teal",
            capsize=3, linewidth=2, markersize=6)
z = np.polyfit(centers, bmeans, 1)
xfit = np.linspace(min(centers), max(centers), 100)
ax.plot(xfit, np.polyval(z, xfit), "k--", alpha=0.5)
r_g, p_g = pearsonr(rpes_c, fcz_c)
ax.set_xlabel("RPE (binary, decile bins)")
ax.set_ylabel("Mean FCz amplitude (µV)")
ax.set_title(f"E. Grand Average RPE × FCz\n(r={r_g:.4f}, p={p_g:.4f})")

# ── 8f: Grand average magnitude RPE × FCz ───────────────────────
ax = axes[1, 1]
rpes_mag = trial_df["rpe_mag"].values
mask_m = np.isfinite(rpes_mag) & np.isfinite(fcz_all)
rm, fm = rpes_mag[mask_m], fcz_all[mask_m]
edges_m = np.percentile(rm, np.linspace(0, 100, n_bins + 1))
centers_m, bmeans_m, bsems_m = [], [], []
for b in range(n_bins):
    m = (rm >= edges_m[b]) & (rm <= edges_m[b + 1]) if b == n_bins - 1 \
        else (rm >= edges_m[b]) & (rm < edges_m[b + 1])
    if m.sum() > 0:
        centers_m.append(rm[m].mean())
        bmeans_m.append(fm[m].mean())
        bsems_m.append(fm[m].std() / np.sqrt(m.sum()))
ax.errorbar(centers_m, bmeans_m, yerr=bsems_m, fmt="s-", color="purple",
            capsize=3, linewidth=2, markersize=6)
z_m = np.polyfit(centers_m, bmeans_m, 1)
xfit_m = np.linspace(min(centers_m), max(centers_m), 100)
ax.plot(xfit_m, np.polyval(z_m, xfit_m), "k--", alpha=0.5)
r_gm, p_gm = pearsonr(rm, fm)
ax.set_xlabel("RPE (magnitude, decile bins)")
ax.set_ylabel("Mean FCz amplitude (µV)")
ax.set_title(f"F. Magnitude RPE × FCz\n(r={r_gm:.4f}, p={p_gm:.4f})")

# ── 8g: FCz by Outcome × Stay/Shift ─────────────────────────────
ax = axes[1, 2]
conds_ss = ["rew_stay", "rew_shift", "norew_stay", "norew_shift"]
ss_means = [trial_df[trial_df["cond_dir"] == c]["FCz"].mean() for c in conds_ss]
ss_sems = [trial_df[trial_df["cond_dir"] == c]["FCz"].std() /
           np.sqrt(len(trial_df[trial_df["cond_dir"] == c])) for c in conds_ss]
colors_ss = [C_HIGH, "#90EE90", C_NOREW, "#FFB6C1"]
ax.bar(range(4), ss_means, yerr=ss_sems, color=colors_ss,
       capsize=3, edgecolor="black", alpha=0.8)
ax.set_xticks(range(4))
ax.set_xticklabels(["Rew\nStay", "Rew\nShift", "NoRew\nStay", "NoRew\nShift"],
                    fontsize=9)
ax.set_ylabel("FCz amplitude (µV)")
ax.set_title("G. FCz by Outcome × Stay/Shift")

# ── 8h: Walking RT by reward level ──────────────────────────────
ax = axes[1, 3]
rt_means = []
rt_sems = []
for vc in [1, 3, 2]:
    sub = trial_df[(trial_df["Vel_Cond"] == vc) & (trial_df["Start_RT"] < 15)]
    rt_means.append(sub["Start_RT"].mean())
    rt_sems.append(sub["Start_RT"].std() / np.sqrt(len(sub)))
ax.bar([0, 1, 2], rt_means, yerr=rt_sems,
       color=[C_HIGH, C_LOW, C_NOREW], alpha=0.8, capsize=4, edgecolor="black")
ax.set_xticks([0, 1, 2])
ax.set_xticklabels(["High Rew\n($0.25)", "Low Rew\n($0.05)", "No Rew\n($0.00)"],
                    fontsize=9)
ax.set_ylabel("Mean Walking RT (sec)")
ax.set_title("H. Walking Speed by Reward Level")

# ── 8i: RPE distribution by reward level ────────────────────────
ax = axes[2, 0]
for vc, label, color in [(1, "High Rew", C_HIGH), (3, "Low Rew", C_LOW),
                           (2, "No Rew", C_NOREW)]:
    vals = trial_df[trial_df["Vel_Cond"] == vc]["rpe_mag"].dropna()
    ax.hist(vals, bins=30, alpha=0.5, color=color, label=label, edgecolor="black",
            linewidth=0.5)
ax.axvline(0, ls="--", color="black", linewidth=1.5)
ax.set_xlabel("RPE (magnitude)")
ax.set_ylabel("Count")
ax.set_title("I. RPE Distribution by Reward Level")
ax.legend()

# ── 8j: Post-outcome RT change by reward level ──────────────────
ax = axes[2, 1]
rt_changes = []
for level, color, label in [("high_rew", C_HIGH, "After High"),
                              ("low_rew", C_LOW, "After Low"),
                              ("no_rew", C_NOREW, "After NoRew")]:
    sub = rt_adj_df[rt_adj_df["prev_reward_level"] == level]
    rt_changes.append(sub["rt_change"].mean())
ax.bar([0, 1, 2], rt_changes, color=[C_HIGH, C_LOW, C_NOREW],
       alpha=0.8, edgecolor="black")
ax.axhline(0, ls="--", color="black", alpha=0.3)
ax.set_xticks([0, 1, 2])
ax.set_xticklabels(["After High\nRew", "After Low\nRew", "After No\nRew"], fontsize=9)
ax.set_ylabel("RT change (sec)")
ax.set_title("J. Post-Outcome Walking Speed Change")

# ── 8k: Topography — no-reward RPE-ERP ──────────────────────────
ax = axes[2, 2]
topo_nr = {}
for ch in channels:
    sub = corr_df[(corr_df["rpe_type"] == "rpe_binary") &
                   (corr_df["condition"] == "no_rew") & (corr_df["channel"] == ch)]
    if len(sub) > 0:
        topo_nr[ch] = sub["r"].mean()
ch_list = list(topo_nr.keys())
r_list = [topo_nr[c] for c in ch_list]
max_r = max(abs(min(r_list)), abs(max(r_list)), 0.04)
cmap_vals = plt.cm.RdBu_r([(r + max_r) / (2 * max_r) for r in r_list])
ax.barh(ch_list, r_list, color=cmap_vals, edgecolor="black")
ax.axvline(0, ls="--", color="black")
ax.set_xlabel("Mean r (RPE × ERP)")
ax.set_title("K. RPE-ERP Topography\n(No-Reward trials, binary RPE)")

# ── 8l: Learning rate distribution ───────────────────────────────
ax = axes[2, 3]
ax.hist(info_df["rw_alpha_binary"].dropna(), bins=20, color="coral",
        edgecolor="black", alpha=0.6, label="Binary")
ax.hist(info_df["rw_alpha_mag"].dropna(), bins=20, color="purple",
        edgecolor="black", alpha=0.4, label="Magnitude")
ax.set_xlabel("RW Learning Rate (α)")
ax.set_ylabel("# Subjects")
ax.set_title(f"L. Learning Rate Distribution\n(bin M={info_df['rw_alpha_binary'].mean():.3f}, "
             f"mag M={info_df['rw_alpha_mag'].mean():.3f})")
ax.legend()

plt.suptitle("T-Maze AR: RPE × ERP × Reward Magnitude (N=43)",
             fontsize=16, y=1.01)
plt.tight_layout()
fig_path = os.path.join(OUTPUT_DIR, "09_tmaze_full_pipeline.png")
plt.savefig(fig_path, dpi=150, bbox_inches="tight")
print(f"  Main figure: {fig_path}")
plt.close()

# ── Heatmap: RPE type × Channel × Condition ─────────────────────
fig2, axes2 = plt.subplots(2, 1, figsize=(14, 8))

for ax_idx, (rpe_type, rpe_label) in enumerate(
    [("rpe_binary", "Binary RPE"), ("rpe_mag", "Magnitude RPE")]):
    ax = axes2[ax_idx]
    cond_list = ["all", "high_rew", "low_rew", "no_rew", "any_rew", "stay", "shift"]
    mat_r = np.zeros((len(cond_list), len(channels)))
    mat_p = np.zeros_like(mat_r)
    for i, cond in enumerate(cond_list):
        for j, ch in enumerate(channels):
            sub = corr_df[(corr_df["rpe_type"] == rpe_type) &
                           (corr_df["condition"] == cond) & (corr_df["channel"] == ch)]
            if len(sub) >= 10:
                z_vals = sub["z"].values
                _, p_val = ttest_1samp(z_vals, 0)
                mat_r[i, j] = sub["r"].mean()
                mat_p[i, j] = p_val
    im = ax.imshow(mat_r, cmap="RdBu_r", vmin=-0.06, vmax=0.06, aspect="auto")
    ax.set_xticks(range(len(channels)))
    ax.set_xticklabels(channels, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(cond_list)))
    ax.set_yticklabels(["All", "High Rew", "Low Rew", "No Rew",
                         "Any Rew", "Stay", "Shift"], fontsize=9)
    for i in range(len(cond_list)):
        for j in range(len(channels)):
            sig = "**" if mat_p[i, j] < 0.01 else ("*" if mat_p[i, j] < 0.05 else "")
            ax.text(j, i, f"{mat_r[i, j]:.3f}{sig}", ha="center", va="center",
                    fontsize=7, color="white" if abs(mat_r[i, j]) > 0.03 else "black")
    plt.colorbar(im, ax=ax, label="Mean r", shrink=0.8)
    ax.set_title(f"{rpe_label}: Condition × Channel", fontsize=12)

plt.suptitle("Single-Trial RPE × ERP Heatmap (N=43)", fontsize=14)
plt.tight_layout()
hm_path = os.path.join(OUTPUT_DIR, "08_rpe_erp_heatmap.png")
plt.savefig(hm_path, dpi=150, bbox_inches="tight")
print(f"  Heatmap: {hm_path}")
plt.close()

# ── Save data ────────────────────────────────────────────────────
trial_df.to_csv(os.path.join(OUTPUT_DIR, "trial_data_with_rpes.csv"), index=False)
corr_df.to_csv(os.path.join(OUTPUT_DIR, "within_subject_correlations.csv"), index=False)
info_df.to_csv(os.path.join(OUTPUT_DIR, "subject_info.csv"), index=False)
gresults_df.to_csv(os.path.join(OUTPUT_DIR, "group_rpe_erp_stats.csv"), index=False)
if rpe_rt_corrs:
    pd.DataFrame(rpe_rt_corrs).to_csv(
        os.path.join(OUTPUT_DIR, "rpe_rt_correlations.csv"), index=False)

# =====================================================================
# SUMMARY
# =====================================================================
print(f"\n{'=' * 70}")
print("SUMMARY TABLE FOR PAPER")
print("=" * 70)
print(f"\n  N = {len(subjects)} subjects, {len(trial_df)} single trials")
print(f"  Reward levels: High ($0.25, n=1580), Low ($0.05, n=2372), None ($0.00, n=4011)")
print(f"  Best group model: WSLS ({Counter(info_df['best_model'])['wsls']}/{len(subjects)})")

print(f"\n  Significant/trending RPE × ERP effects (p < .15):")
for _, row in gresults_df[gresults_df["p"] < 0.15].sort_values("p").iterrows():
    sig = "**" if row["p"] < 0.01 else ("*" if row["p"] < 0.05 else "~")
    rpe_l = "bin" if "binary" in row["rpe_type"] else "mag"
    print(f"    {sig:>2} {rpe_l:<4} {row['condition']:<12} @ {row['channel']:<4} "
          f"r={row['mean_r']:.4f}, t({int(row['n_subjects'])-1})={row['t']:.2f}, "
          f"p={row['p']:.4f}")

print(f"\n  Output: {OUTPUT_DIR}")
