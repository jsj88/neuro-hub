"""
Example 7: Post-Error Slowing / Post-Reward Speeding × ERP Analysis.

Analyzes behavioral adjustment effects in T-maze data:
- Post-Error Slowing (PES): slower RTs after no-reward trials
- Post-Reward Speeding (PRS): faster RTs after reward trials
- Win-Stay/Lose-Switch rates
- Correlates all metrics with ERP components (REWP, P3, N2, P2)

Usage:
    cd ~/neuro-hub
    python examples/07_pes_prs_erp_analysis.py
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
from scipy.stats import pearsonr, spearmanr, ttest_rel, ttest_ind, wilcoxon

# ── Paths ────────────────────────────────────────────────────────
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "sample_data")
BEH_FILE = os.path.join(DATA_DIR, "RT_DATA_v9_earned_allSUBS.csv")
ERP_FILE = os.path.join(DATA_DIR, "ERP_STATS_ALL.xlsx")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "pes_prs_analysis")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =====================================================================
# STEP 1: Load data
# =====================================================================
print("=" * 60)
print("STEP 1: Loading data")
print("=" * 60)

beh = pd.read_csv(BEH_FILE)
erp = pd.read_excel(ERP_FILE)

# Map ERP subject IDs
def extract_subj_num(s):
    m = re.search(r"(\d+)", s)
    return int(m.group(1)) if m else None

erp["SubjectCode"] = erp["Subject"].apply(extract_subj_num)
subjects = sorted(beh["SubjectCode"].unique())
print(f"  Behavioral: {len(subjects)} subjects, {len(beh)} trials")
print(f"  ERP: {erp['SubjectCode'].nunique()} subjects")

# =====================================================================
# STEP 2: Compute PES/PRS metrics per subject
# =====================================================================
print(f"\n{'=' * 60}")
print("STEP 2: Computing PES/PRS behavioral metrics")
print("=" * 60)

# RT outlier threshold (trim > 99th percentile per subject)
RT_TRIM_PERCENTILE = 99

subject_metrics = []

for subj in subjects:
    s = beh[beh["SubjectCode"] == subj].sort_values("Trial").copy()
    n_trials = len(s)

    # Trim RT outliers within subject
    rt_thresh = s["Start_RT"].quantile(RT_TRIM_PERCENTILE / 100)
    s_clean = s[s["Start_RT"] <= rt_thresh].copy()

    # --- PES/PRS from Start_RT ---
    post_rew = s_clean[s_clean["PES_PRS_index"] == 2]["Start_RT"]
    post_norew = s_clean[s_clean["PES_PRS_index"] == 1]["Start_RT"]

    # PES = RT(post-no-reward) - RT(post-reward)
    pes_start = post_norew.median() - post_rew.median()
    pes_start_mean = post_norew.mean() - post_rew.mean()

    # Same for Return_RT
    post_rew_ret = s_clean[s_clean["PES_PRS_index"] == 2]["Return_RT"]
    post_norew_ret = s_clean[s_clean["PES_PRS_index"] == 1]["Return_RT"]
    pes_return = post_norew_ret.median() - post_rew_ret.median()

    # --- Win-Stay / Lose-Switch ---
    dirs = s["Direction"].values
    rews = s["Reward"].values  # 1=reward, 2=no-reward

    stays_after_rew = 0
    switches_after_norew = 0
    n_after_rew = 0
    n_after_norew = 0

    for t in range(1, len(dirs)):
        stayed = dirs[t] == dirs[t - 1]
        if rews[t - 1] == 1:  # Previous was reward
            n_after_rew += 1
            if stayed:
                stays_after_rew += 1
        else:  # Previous was no-reward
            n_after_norew += 1
            if not stayed:
                switches_after_norew += 1

    win_stay_rate = stays_after_rew / max(n_after_rew, 1)
    lose_switch_rate = switches_after_norew / max(n_after_norew, 1)
    wsls_index = (win_stay_rate + lose_switch_rate) / 2  # Combined WSLS

    # --- Sequential reward sensitivity ---
    # How much does current RT depend on previous outcome?
    outcomes = (s["Reward"] == 1).astype(int).values
    rts = s["Start_RT"].values

    # RT change after win vs loss (robust: use consecutive pairs)
    rt_changes = []
    for t in range(1, len(rts)):
        if rts[t] < rt_thresh and rts[t - 1] < rt_thresh:
            rt_changes.append({
                "prev_outcome": outcomes[t - 1],
                "rt_change": rts[t] - rts[t - 1],
                "current_rt": rts[t],
            })
    rc_df = pd.DataFrame(rt_changes)

    if len(rc_df) > 10:
        rt_change_after_win = rc_df[rc_df["prev_outcome"] == 1]["rt_change"].mean()
        rt_change_after_loss = rc_df[rc_df["prev_outcome"] == 0]["rt_change"].mean()
        seq_sensitivity = rt_change_after_loss - rt_change_after_win
    else:
        rt_change_after_win = rt_change_after_loss = seq_sensitivity = np.nan

    # --- Earned (reward magnitude) effects ---
    earned = s["Earned"].values
    mean_earned = earned[earned > 0].mean() if np.any(earned > 0) else 0

    # --- Position variability (exploration) ---
    position_std = s["Position"].std()

    subject_metrics.append({
        "subject": subj,
        "n_trials": n_trials,
        # PES/PRS
        "pes_start_median": pes_start,
        "pes_start_mean": pes_start_mean,
        "pes_return_median": pes_return,
        "rt_post_reward": post_rew.median(),
        "rt_post_norew": post_norew.median(),
        "rt_overall_median": s_clean["Start_RT"].median(),
        # Win-stay / Lose-switch
        "win_stay_rate": win_stay_rate,
        "lose_switch_rate": lose_switch_rate,
        "wsls_index": wsls_index,
        # Sequential effects
        "rt_change_after_win": rt_change_after_win,
        "rt_change_after_loss": rt_change_after_loss,
        "seq_sensitivity": seq_sensitivity,
        # Task performance
        "reward_rate": np.mean(outcomes),
        "mean_earned": mean_earned,
        "position_std": position_std,
    })

metrics_df = pd.DataFrame(subject_metrics)

# Group-level PES test
t_pes, p_pes = wilcoxon(metrics_df["pes_start_median"])
print(f"\n  Post-Error Slowing (PES):")
print(f"    Median PES = {metrics_df['pes_start_median'].median():.3f} sec")
print(f"    Mean PES   = {metrics_df['pes_start_median'].mean():.3f} sec")
print(f"    Wilcoxon signed-rank: W={t_pes:.1f}, p={p_pes:.4f}")
print(f"    Subjects with PES > 0: {(metrics_df['pes_start_median'] > 0).sum()}/{len(metrics_df)}")

print(f"\n  Win-Stay / Lose-Switch:")
print(f"    Win-Stay rate:    {metrics_df['win_stay_rate'].mean():.3f} (SD={metrics_df['win_stay_rate'].std():.3f})")
print(f"    Lose-Switch rate: {metrics_df['lose_switch_rate'].mean():.3f} (SD={metrics_df['lose_switch_rate'].std():.3f})")
print(f"    WSLS index:       {metrics_df['wsls_index'].mean():.3f}")

# =====================================================================
# STEP 3: Merge with ERP data
# =====================================================================
print(f"\n{'=' * 60}")
print("STEP 3: Merging with ERP data")
print("=" * 60)

# REWP
rewp = erp[erp["LABEL"] == "REWP"][["SubjectCode", "FCz", "Cz", "Pz", "GROUP"]].copy()
rewp.columns = ["subject", "REWP_FCz", "REWP_Cz", "REWP_Pz", "GROUP"]

# P3 REW and NREW
p3_rew = erp[(erp["LABEL"] == "P3") & (erp["CONDITION"] == "REW")][["SubjectCode", "FCz", "Cz", "Pz"]].copy()
p3_rew.columns = ["subject", "P3_REW_FCz", "P3_REW_Cz", "P3_REW_Pz"]
p3_nrew = erp[(erp["LABEL"] == "P3") & (erp["CONDITION"] == "NREW")][["SubjectCode", "FCz", "Cz", "Pz"]].copy()
p3_nrew.columns = ["subject", "P3_NREW_FCz", "P3_NREW_Cz", "P3_NREW_Pz"]

# N2 REW and NREW
n2_rew = erp[(erp["LABEL"] == "N2") & (erp["CONDITION"] == "REW")][["SubjectCode", "FCz", "Cz"]].copy()
n2_rew.columns = ["subject", "N2_REW_FCz", "N2_REW_Cz"]
n2_nrew = erp[(erp["LABEL"] == "N2") & (erp["CONDITION"] == "NREW")][["SubjectCode", "FCz", "Cz"]].copy()
n2_nrew.columns = ["subject", "N2_NREW_FCz", "N2_NREW_Cz"]

# P2 REW and NREW
p2_rew = erp[(erp["LABEL"] == "P2") & (erp["CONDITION"] == "REW")][["SubjectCode", "FCz", "Cz"]].copy()
p2_rew.columns = ["subject", "P2_REW_FCz", "P2_REW_Cz"]
p2_nrew = erp[(erp["LABEL"] == "P2") & (erp["CONDITION"] == "NREW")][["SubjectCode", "FCz", "Cz"]].copy()
p2_nrew.columns = ["subject", "P2_NREW_FCz", "P2_NREW_Cz"]

# Merge all
merged = metrics_df.merge(rewp, on="subject", how="inner")
merged = merged.merge(p3_rew, on="subject", how="left")
merged = merged.merge(p3_nrew, on="subject", how="left")
merged = merged.merge(n2_rew, on="subject", how="left")
merged = merged.merge(n2_nrew, on="subject", how="left")
merged = merged.merge(p2_rew, on="subject", how="left")
merged = merged.merge(p2_nrew, on="subject", how="left")

# Compute difference waves
merged["P3_diff_FCz"] = merged["P3_REW_FCz"] - merged["P3_NREW_FCz"]
merged["N2_diff_FCz"] = merged["N2_REW_FCz"] - merged["N2_NREW_FCz"]
merged["P2_diff_FCz"] = merged["P2_REW_FCz"] - merged["P2_NREW_FCz"]

merged.to_csv(os.path.join(OUTPUT_DIR, "pes_prs_erp_merged.csv"), index=False)
print(f"  Matched subjects: {len(merged)}")

# =====================================================================
# STEP 4: PES/PRS × ERP correlations
# =====================================================================
print(f"\n{'=' * 60}")
print("STEP 4: Behavioral adjustment × ERP correlations")
print("=" * 60)

beh_vars = {
    "pes_start_median": "PES (Start RT)",
    "pes_return_median": "PES (Return RT)",
    "win_stay_rate": "Win-Stay rate",
    "lose_switch_rate": "Lose-Switch rate",
    "wsls_index": "WSLS index",
    "seq_sensitivity": "Sequential sensitivity",
}

erp_vars = {
    "REWP_FCz": "REWP (FCz)",
    "P3_diff_FCz": "P3 diff (FCz)",
    "N2_diff_FCz": "N2 diff (FCz)",
    "P2_diff_FCz": "P2 diff (FCz)",
    "P3_REW_FCz": "P3 Reward (FCz)",
    "P3_NREW_FCz": "P3 No-Reward (FCz)",
}

print(f"\n  {'Behavioral':<25} {'ERP Variable':<22} {'r':>7} {'p':>9} {'sig':>4}")
print("  " + "-" * 70)

all_corrs = []
significant_corrs = []

for bv_key, bv_label in beh_vars.items():
    for ev_key, ev_label in erp_vars.items():
        x = merged[bv_key].values.astype(float)
        y = merged[ev_key].values.astype(float)
        mask = np.isfinite(x) & np.isfinite(y)
        if mask.sum() >= 10:
            r, p = pearsonr(x[mask], y[mask])
            rs, ps = spearmanr(x[mask], y[mask])
            sig = "**" if p < 0.01 else ("*" if p < 0.05 else ("~" if p < 0.10 else ""))
            all_corrs.append({
                "beh_var": bv_key, "erp_var": ev_key,
                "pearson_r": r, "pearson_p": p,
                "spearman_r": rs, "spearman_p": ps,
                "n": int(mask.sum()),
            })
            if p < 0.10:
                significant_corrs.append((bv_label, ev_label, r, p, sig))
            # Print REWP and P3 diff correlations
            if ev_key in ["REWP_FCz", "P3_diff_FCz"]:
                print(f"  {bv_label:<25} {ev_label:<22} {r:>7.3f} {p:>9.4f} {sig:>4}")

corr_df = pd.DataFrame(all_corrs)
corr_df.to_csv(os.path.join(OUTPUT_DIR, "pes_erp_correlations.csv"), index=False)

if significant_corrs:
    print(f"\n  Significant/trending correlations (p < .10):")
    for bv, ev, r, p, sig in significant_corrs:
        print(f"    {sig} {bv} x {ev}: r={r:.3f}, p={p:.4f}")
else:
    print(f"\n  No correlations reached p < .10")

# =====================================================================
# STEP 5: Group comparisons
# =====================================================================
print(f"\n{'=' * 60}")
print("STEP 5: Group comparisons")
print("=" * 60)

groups = sorted(merged["GROUP"].unique())
print(f"  Groups: {groups}")
print(f"  Group sizes: {merged['GROUP'].value_counts().sort_index().to_dict()}")

print(f"\n  {'Metric':<25}", end="")
for g in groups:
    print(f"  {'Group '+str(g):>10}", end="")
print(f"  {'F/t':>8} {'p':>8}")
print("  " + "-" * 75)

from scipy.stats import f_oneway, kruskal

group_test_vars = [
    "pes_start_median", "win_stay_rate", "lose_switch_rate",
    "wsls_index", "seq_sensitivity", "rt_overall_median",
    "reward_rate", "REWP_FCz", "P3_diff_FCz",
]

for var in group_test_vars:
    group_data = [merged[merged["GROUP"] == g][var].dropna().values for g in groups]
    valid = [gd for gd in group_data if len(gd) >= 3]

    if len(valid) >= 2:
        if len(valid) == 2:
            from scipy.stats import mannwhitneyu
            stat, p = mannwhitneyu(valid[0], valid[1], alternative="two-sided")
            test_name = "U"
        else:
            stat, p = kruskal(*valid)
            test_name = "H"

        sig = "**" if p < 0.01 else ("*" if p < 0.05 else ("~" if p < 0.10 else ""))
        print(f"  {var:<25}", end="")
        for g in groups:
            gd = merged[merged["GROUP"] == g][var]
            print(f"  {gd.mean():>10.3f}", end="")
        print(f"  {stat:>8.2f} {p:>7.4f} {sig}")

# =====================================================================
# STEP 6: Median-split analysis
# =====================================================================
print(f"\n{'=' * 60}")
print("STEP 6: Median-split analysis (high vs low PES)")
print("=" * 60)

median_pes = merged["pes_start_median"].median()
merged["PES_group"] = np.where(merged["pes_start_median"] > median_pes, "High PES", "Low PES")

for var in ["REWP_FCz", "P3_diff_FCz", "P3_REW_FCz", "P3_NREW_FCz",
            "win_stay_rate", "lose_switch_rate"]:
    high = merged[merged["PES_group"] == "High PES"][var].dropna()
    low = merged[merged["PES_group"] == "Low PES"][var].dropna()
    if len(high) >= 5 and len(low) >= 5:
        t, p = ttest_ind(high, low)
        sig = "*" if p < 0.05 else ("~" if p < 0.10 else "")
        print(f"  {var:<20} High PES={high.mean():>7.3f}  Low PES={low.mean():>7.3f}"
              f"  t={t:>6.2f}  p={p:.4f} {sig}")

# =====================================================================
# STEP 7: Figures
# =====================================================================
print(f"\n{'=' * 60}")
print("STEP 7: Generating figures")
print("=" * 60)

fig, axes = plt.subplots(2, 4, figsize=(22, 11))

# 7a: PES effect (paired)
ax = axes[0, 0]
ax.bar(["Post-Reward", "Post-No-Reward"],
       [metrics_df["rt_post_reward"].mean(), metrics_df["rt_post_norew"].mean()],
       yerr=[metrics_df["rt_post_reward"].sem(), metrics_df["rt_post_norew"].sem()],
       color=["green", "red"], alpha=0.7, capsize=5)
ax.set_ylabel("Median Start RT (sec)")
ax.set_title(f"Post-Error Slowing\n(PES={metrics_df['pes_start_median'].mean():.3f}s, p={p_pes:.4f})")

# 7b: PES distribution
ax = axes[0, 1]
ax.hist(metrics_df["pes_start_median"], bins=20, color="steelblue",
        edgecolor="black", alpha=0.7)
ax.axvline(0, ls="--", color="red", linewidth=2, label="No PES")
ax.axvline(metrics_df["pes_start_median"].median(), ls="-", color="black",
           linewidth=2, label=f"Median={metrics_df['pes_start_median'].median():.3f}")
ax.set_xlabel("PES (sec)")
ax.set_ylabel("# Subjects")
ax.set_title("PES Distribution Across Subjects")
ax.legend()

# 7c: Win-Stay vs Lose-Switch
ax = axes[0, 2]
ws = metrics_df["win_stay_rate"]
ls_ = metrics_df["lose_switch_rate"]
ax.scatter(ws, ls_, alpha=0.7, edgecolors="black", linewidth=0.5, color="teal")
ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
ax.set_xlabel("Win-Stay Rate")
ax.set_ylabel("Lose-Switch Rate")
ax.set_title(f"Win-Stay vs Lose-Switch\n(WS={ws.mean():.2f}, LS={ls_.mean():.2f})")
ax.set_xlim(0.2, 0.9)
ax.set_ylim(0.2, 0.9)

# 7d: PES × REWP scatter
ax = axes[0, 3]
x = merged["pes_start_median"]
y = merged["REWP_FCz"]
ax.scatter(x, y, alpha=0.7, edgecolors="black", linewidth=0.5, color="purple")
mask = np.isfinite(x) & np.isfinite(y)
if mask.sum() > 2:
    z = np.polyfit(x[mask], y[mask], 1)
    xline = np.linspace(x[mask].min(), x[mask].max(), 100)
    ax.plot(xline, np.polyval(z, xline), "k-", linewidth=2)
    r, p = pearsonr(x[mask], y[mask])
    ax.set_title(f"PES × REWP\n(r={r:.3f}, p={p:.4f})")
ax.set_xlabel("PES (sec)")
ax.set_ylabel("REWP at FCz (uV)")

# 7e: Win-Stay × REWP
ax = axes[1, 0]
x = merged["win_stay_rate"]
y = merged["REWP_FCz"]
ax.scatter(x, y, alpha=0.7, edgecolors="black", linewidth=0.5, color="coral")
mask = np.isfinite(x) & np.isfinite(y)
if mask.sum() > 2:
    z = np.polyfit(x[mask], y[mask], 1)
    xline = np.linspace(x[mask].min(), x[mask].max(), 100)
    ax.plot(xline, np.polyval(z, xline), "k-", linewidth=2)
    r, p = pearsonr(x[mask], y[mask])
    ax.set_title(f"Win-Stay × REWP\n(r={r:.3f}, p={p:.4f})")
ax.set_xlabel("Win-Stay Rate")
ax.set_ylabel("REWP at FCz (uV)")

# 7f: Lose-Switch × P3 diff
ax = axes[1, 1]
x = merged["lose_switch_rate"]
y = merged["P3_diff_FCz"]
ax.scatter(x, y, alpha=0.7, edgecolors="black", linewidth=0.5, color="orange")
mask = np.isfinite(x) & np.isfinite(y)
if mask.sum() > 2:
    z = np.polyfit(x[mask], y[mask], 1)
    xline = np.linspace(x[mask].min(), x[mask].max(), 100)
    ax.plot(xline, np.polyval(z, xline), "k-", linewidth=2)
    r, p = pearsonr(x[mask], y[mask])
    ax.set_title(f"Lose-Switch × P3 Difference\n(r={r:.3f}, p={p:.4f})")
ax.set_xlabel("Lose-Switch Rate")
ax.set_ylabel("P3 Reward-NoReward (uV)")

# 7g: Group comparison boxplots (PES by group)
ax = axes[1, 2]
group_labels = sorted(merged["GROUP"].unique())
bp_data = [merged[merged["GROUP"] == g]["pes_start_median"].values for g in group_labels]
bp = ax.boxplot(bp_data, labels=[f"Group {g}\n(n={len(d)})" for g, d in zip(group_labels, bp_data)],
                patch_artist=True)
colors_bp = ["#4ECDC4", "#FF6B6B", "#45B7D1"]
for patch, color in zip(bp["boxes"], colors_bp[:len(bp_data)]):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
ax.axhline(0, ls="--", color="black", alpha=0.3)
ax.set_ylabel("PES (sec)")
ax.set_title("PES by Group")

# 7h: WSLS index × REWP
ax = axes[1, 3]
x = merged["wsls_index"]
y = merged["REWP_FCz"]
for g in sorted(merged["GROUP"].unique()):
    mask_g = merged["GROUP"] == g
    ax.scatter(x[mask_g], y[mask_g], alpha=0.7, label=f"Group {g}",
               edgecolors="black", linewidth=0.5)
mask = np.isfinite(x) & np.isfinite(y)
if mask.sum() > 2:
    r, p = pearsonr(x[mask], y[mask])
    z = np.polyfit(x[mask], y[mask], 1)
    xline = np.linspace(x[mask].min(), x[mask].max(), 100)
    ax.plot(xline, np.polyval(z, xline), "k-", linewidth=2)
    ax.set_title(f"WSLS Index × REWP\n(r={r:.3f}, p={p:.4f})")
ax.set_xlabel("WSLS Index")
ax.set_ylabel("REWP at FCz (uV)")
ax.legend()

plt.suptitle("T-Maze: Post-Error Slowing & Behavioral Adjustments × ERP",
             fontsize=16, y=1.02)
plt.tight_layout()
fig_path = os.path.join(OUTPUT_DIR, "07_pes_prs_erp.png")
plt.savefig(fig_path, dpi=150, bbox_inches="tight")
print(f"  Main figure: {fig_path}")
plt.close()

# ── Additional: Correlation heatmap ──────────────────────────────
fig2, ax2 = plt.subplots(figsize=(10, 7))
beh_cols = ["pes_start_median", "pes_return_median", "win_stay_rate",
            "lose_switch_rate", "wsls_index", "seq_sensitivity"]
erp_cols = ["REWP_FCz", "REWP_Cz", "P3_diff_FCz", "N2_diff_FCz",
            "P2_diff_FCz", "P3_REW_FCz", "P3_NREW_FCz"]

corr_matrix = np.zeros((len(beh_cols), len(erp_cols)))
p_matrix = np.zeros_like(corr_matrix)

for i, bv in enumerate(beh_cols):
    for j, ev in enumerate(erp_cols):
        x = merged[bv].values.astype(float)
        y = merged[ev].values.astype(float)
        mask = np.isfinite(x) & np.isfinite(y)
        if mask.sum() >= 10:
            corr_matrix[i, j], p_matrix[i, j] = pearsonr(x[mask], y[mask])

im = ax2.imshow(corr_matrix, cmap="RdBu_r", vmin=-0.4, vmax=0.4, aspect="auto")
ax2.set_xticks(range(len(erp_cols)))
ax2.set_xticklabels([c.replace("_", "\n") for c in erp_cols], rotation=45, ha="right")
ax2.set_yticks(range(len(beh_cols)))
ax2.set_yticklabels([c.replace("_", " ") for c in beh_cols])

# Add text with significance markers
for i in range(len(beh_cols)):
    for j in range(len(erp_cols)):
        sig = "**" if p_matrix[i, j] < 0.01 else ("*" if p_matrix[i, j] < 0.05 else "")
        text = f"{corr_matrix[i, j]:.2f}{sig}"
        ax2.text(j, i, text, ha="center", va="center", fontsize=8,
                 color="white" if abs(corr_matrix[i, j]) > 0.25 else "black")

plt.colorbar(im, label="Pearson r")
ax2.set_title("Behavioral Adjustment × ERP Correlation Matrix", fontsize=14)
plt.tight_layout()
heatmap_path = os.path.join(OUTPUT_DIR, "07_correlation_heatmap.png")
plt.savefig(heatmap_path, dpi=150, bbox_inches="tight")
print(f"  Heatmap: {heatmap_path}")
plt.close()

# =====================================================================
# SUMMARY
# =====================================================================
print(f"\n{'=' * 60}")
print("SUMMARY")
print("=" * 60)
print(f"  PES effect: {metrics_df['pes_start_median'].mean():.3f}s (p={p_pes:.4f})")
print(f"  Win-Stay rate: {metrics_df['win_stay_rate'].mean():.3f}")
print(f"  Lose-Switch rate: {metrics_df['lose_switch_rate'].mean():.3f}")
print(f"  REWP mean: {merged['REWP_FCz'].mean():.3f} uV")
print(f"\n  Output: {OUTPUT_DIR}")
print(f"    pes_prs_erp_merged.csv     — All behavioral + ERP metrics")
print(f"    pes_erp_correlations.csv   — Full correlation table")
print(f"    07_pes_prs_erp.png         — 8-panel figure")
print(f"    07_correlation_heatmap.png  — Correlation matrix heatmap")
