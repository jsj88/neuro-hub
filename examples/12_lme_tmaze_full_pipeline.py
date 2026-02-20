"""
Example 12: Full T-Maze AR Pipeline — LME replaces correlations.

Same variables and conditions as example 09, but every test uses
    ERP ~ behv + (1 | subject) + (1 | Trial)
instead of within-subject Pearson correlations + Fisher-z + group t-tests.

The (1|subject) random intercept accounts for between-subject ERP baseline
differences. The (1|Trial) variance component accounts for shared trial-
position effects (learning, fatigue) across subjects.

Behavioral predictors tested:
  - rpe_binary     : Rescorla-Wagner RPE on binary outcome (0/1)
  - rpe_mag        : Rescorla-Wagner RPE on magnitude outcome ($0.00-$0.25)
  - abs_rpe_binary : |RPE| binary
  - abs_rpe_mag    : |RPE| magnitude
  - chosen_q_binary: Q-value of the chosen option (binary)
  - chosen_q_mag   : Q-value of the chosen option (magnitude)
  - is_reward      : binary reward indicator (1=reward, 0=noreward)
  - reward_level_num: ordinal (0=none, 1=low, 2=high)
  - Earned         : dollar amount ($0.00, $0.05, $0.25)
  - Start_RT       : walking reaction time

ERP channels: FCz, Cz, FC2, FC1, F4, F5, C4, C3, P4, P3, P8, P7, Pz

Conditions:
  all, high_rew, no_rew, low_rew, any_rew, stay, shift,
  rew_stay, rew_shift, norew_stay, norew_shift

Usage:
    cd ~/neuro-hub
    python examples/12_lme_tmaze_full_pipeline.py
"""

import sys, os, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import Counter
from statsmodels.stats.multitest import multipletests
import statsmodels.formula.api as smf
from agent.tools.model_fitting import fit_model, extract_trial_variables

# ── Paths ────────────────────────────────────────────────────────
DATA_FILE = os.path.join(os.path.dirname(__file__), "..", "sample_data",
                         "merged_clean_V3_scaled.csv")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "lme_tmaze_pipeline")
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODELS = ["random", "wsls", "rw", "q_learning", "q_dual",
          "actor_critic", "q_decay", "rw_bias", "ck", "rwck"]

CHANNELS = ["FCz", "Cz", "FC2", "FC1", "F4", "F5",
            "C4", "C3", "P4", "P3", "P8", "P7", "Pz"]

CONDITIONS = {
    "all":          lambda d: d,
    "high_rew":     lambda d: d[d["Vel_Cond"] == 1],
    "no_rew":       lambda d: d[d["Vel_Cond"] == 2],
    "low_rew":      lambda d: d[d["Vel_Cond"] == 3],
    "any_rew":      lambda d: d[d["is_reward"] == 1],
    "stay":         lambda d: d[d["stay_shift"] == "stay"],
    "shift":        lambda d: d[d["stay_shift"] == "shift"],
    "rew_stay":     lambda d: d[d["Condition_Direction_index"] == 1],
    "rew_shift":    lambda d: d[d["Condition_Direction_index"] == 2],
    "norew_stay":   lambda d: d[d["Condition_Direction_index"] == 3],
    "norew_shift":  lambda d: d[d["Condition_Direction_index"] == 4],
}

BEHAV_PREDICTORS = [
    "rpe_binary", "rpe_mag",
    "abs_rpe_binary", "abs_rpe_mag",
    "chosen_q_binary", "chosen_q_mag",
    "is_reward", "reward_level_num", "Earned", "Start_RT", "RT_DIFF",
]


# ═════════════════════════════════════════════════════════════════
# LME helper: ERP ~ behv + (1 | subject) + (1 | Trial)
# ═════════════════════════════════════════════════════════════════

def fit_lme(data, erp_col, behav_col, use_trial_re=True):
    """Fit ERP ~ behv + (1|subject) [+ (1|Trial)].

    Returns dict with estimate, se, z, p, n_obs, n_subj, aic, bic, converged.
    Returns None if the model fails or data is insufficient.
    """
    cols_needed = [erp_col, behav_col, "subject", "Trial"]
    sub = data.dropna(subset=[c for c in cols_needed if c in data.columns])

    if len(sub) < 30 or sub["subject"].nunique() < 5:
        return None

    formula = f"{erp_col} ~ {behav_col}"
    vc = {"Trial": "0 + C(Trial)"} if use_trial_re else None

    try:
        model = smf.mixedlm(formula, sub, groups=sub["subject"],
                            vc_formula=vc)
        result = model.fit(reml=True, method="powell")
    except Exception:
        # Fall back: drop trial RE
        try:
            model = smf.mixedlm(formula, sub, groups=sub["subject"])
            result = model.fit(reml=True, method="powell")
        except Exception:
            return None

    # Extract the behavioral predictor coefficient
    if behav_col not in result.params.index:
        return None

    coef = float(result.params[behav_col])
    se = float(result.bse[behav_col])
    z = float(result.tvalues[behav_col])
    p = float(result.pvalues[behav_col])

    llf = float(result.llf) if hasattr(result, "llf") else np.nan
    n_params = len(result.params)
    n_obs = len(sub)
    aic = float(result.aic) if hasattr(result, "aic") and not np.isnan(result.aic) else (
        -2 * llf + 2 * n_params if not np.isnan(llf) else np.nan)
    bic = float(result.bic) if hasattr(result, "bic") and not np.isnan(result.bic) else (
        -2 * llf + n_params * np.log(n_obs) if not np.isnan(llf) else np.nan)

    return {
        "estimate": coef, "se": se, "z": z, "p": p,
        "n_obs": n_obs, "n_subj": sub["subject"].nunique(),
        "aic": aic, "bic": bic,
        "converged": getattr(result, "converged", True),
    }


# =====================================================================
# STEP 1: Load and prepare data (identical to example 09)
# =====================================================================
print("=" * 70)
print("STEP 1: Loading data")
print("=" * 70)

df = pd.read_csv(DATA_FILE)

df["subject"] = df["SubjectCode"]
df["choice"] = df["Direction"] - 1
df["outcome_binary"] = (df["Reward"] == 1).astype(float)
df["outcome_mag"] = df["Earned"]
df["reward_level"] = df["Vel_Cond"].map({1: "high_rew", 2: "no_rew", 3: "low_rew"})
df["reward_level_num"] = df["Vel_Cond"].map({1: 2, 2: 0, 3: 1})
df["stay_shift"] = df["PES_PRS_index"].map({0: "first", 1: "stay", 2: "shift"})
df["cond_dir"] = df["Condition_Direction_index"].map({
    0: "first", 1: "rew_stay", 2: "rew_shift",
    3: "norew_stay", 4: "norew_shift"
})
df["is_reward"] = (df["Reward"] == 1).astype(int)

subjects = sorted(df["subject"].unique())
print(f"  {len(df)} trials, {len(subjects)} subjects")
print(f"  Reward levels: "
      f"High n={len(df[df.Vel_Cond==1])}, "
      f"Low n={len(df[df.Vel_Cond==3])}, "
      f"None n={len(df[df.Vel_Cond==2])}")

# =====================================================================
# STEP 2: Fit RW models, extract RPEs (identical to example 09)
# =====================================================================
print(f"\n{'=' * 70}")
print("STEP 2: Fitting models & extracting RPEs")
print("=" * 70)

all_trial_data = []
subject_info = []

for i, subj in enumerate(subjects):
    sdata = df[df["subject"] == subj].sort_values("Trial").copy()
    choices = sdata["choice"].values.astype(int)
    outcomes_bin = sdata["outcome_binary"].values.astype(float)
    outcomes_mag = sdata["outcome_mag"].values.astype(float)

    # Fit all models (binary)
    results = {}
    for model_name in MODELS:
        try:
            r = fit_model(model_name, choices, outcomes_bin,
                          n_options=2, method="de", seed=42)
            results[model_name] = r
        except Exception:
            pass

    best_name = min(results, key=lambda m: results[m].bic)
    rw_r = results.get("rw", None)

    # Binary RPE
    if rw_r:
        rw_latent = extract_trial_variables("rw", rw_r.params,
                                            choices, outcomes_bin, n_options=2)
        sdata["rpe_binary"] = rw_latent["rpes"]
        sdata["abs_rpe_binary"] = np.abs(rw_latent["rpes"])
        if "q_values" in rw_latent:
            sdata["chosen_q_binary"] = rw_latent["q_values"][
                np.arange(len(choices)), choices]

    # Magnitude RPE
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
    except Exception:
        sdata["rpe_mag"] = np.nan
        sdata["abs_rpe_mag"] = np.nan
        rw_mag_alpha = np.nan

    sdata["best_model"] = best_name
    all_trial_data.append(sdata)

    subject_info.append({
        "subject": subj, "best_model": best_name,
        "rw_alpha_binary": rw_r.params.get("alpha", np.nan) if rw_r else np.nan,
        "rw_alpha_mag": rw_mag_alpha,
        "n_trials": len(sdata),
    })

    if (i + 1) % 10 == 0 or i == 0:
        print(f"  {subj} ({i+1}/{len(subjects)}): best={best_name}")

trial_df = pd.concat(all_trial_data, ignore_index=True)
info_df = pd.DataFrame(subject_info)

# Compute RT_DIFF: previous trial's Start_RT minus current trial's Start_RT
# Positive RT_DIFF = participant sped up; Negative = slowed down
trial_df = trial_df.sort_values(["subject", "Trial"]).reset_index(drop=True)
trial_df["RT_DIFF"] = trial_df.groupby("subject")["Start_RT"].shift(1) - trial_df["Start_RT"]
# Exclude extreme outlier RT_DIFFs (first trial per subject will be NaN)
rt_diff_valid = trial_df["RT_DIFF"].dropna()
print(f"\n  RT_DIFF computed: mean={rt_diff_valid.mean():.4f}s, "
      f"std={rt_diff_valid.std():.4f}s, valid n={len(rt_diff_valid)}")

print(f"\n  Best model counts:")
for m, c in Counter(info_df["best_model"]).most_common():
    print(f"    {m:<15} {c} subjects")


# =====================================================================
# STEP 3: LME — ERP ~ behv + (1|subject) + (1|Trial)
#         for all predictor × channel × condition combinations
# =====================================================================
print(f"\n{'=' * 70}")
print("STEP 3: Running LME models: ERP ~ behv + (1|subject) + (1|Trial)")
print("=" * 70)

lme_results = []
total = len(BEHAV_PREDICTORS) * len(CHANNELS) * len(CONDITIONS)
count = 0

for behav in BEHAV_PREDICTORS:
    if behav not in trial_df.columns:
        continue
    for ch in CHANNELS:
        for cond_name, cond_fn in CONDITIONS.items():
            count += 1
            subset = cond_fn(trial_df)

            res = fit_lme(subset, ch, behav, use_trial_re=True)

            if res is not None:
                lme_results.append({
                    "predictor": behav,
                    "channel": ch,
                    "condition": cond_name,
                    **res,
                })

    done_pct = count / total * 100
    print(f"  {behav:<20} done ({done_pct:.0f}%)")

lme_df = pd.DataFrame(lme_results)
print(f"\n  Total LME models fit: {len(lme_df)}")


# =====================================================================
# STEP 4: FDR correction and summary table
# =====================================================================
print(f"\n{'=' * 70}")
print("STEP 4: FDR correction & significant results")
print("=" * 70)

# FDR across all tests
if len(lme_df) > 0:
    reject, pvals_fdr, _, _ = multipletests(lme_df["p"].values,
                                             alpha=0.05, method="fdr_bh")
    lme_df["p_fdr"] = pvals_fdr
    lme_df["sig_fdr"] = reject

    # Also uncorrected significance
    lme_df["sig_unc"] = lme_df["p"] < 0.05

    n_sig_unc = lme_df["sig_unc"].sum()
    n_sig_fdr = lme_df["sig_fdr"].sum()
    print(f"  Uncorrected p < .05: {n_sig_unc}/{len(lme_df)} "
          f"({n_sig_unc/len(lme_df)*100:.1f}%)")
    print(f"  FDR-corrected q < .05: {n_sig_fdr}/{len(lme_df)} "
          f"({n_sig_fdr/len(lme_df)*100:.1f}%)")


# ── Print key results: FCz and Cz for each predictor × condition ──
print(f"\n  {'Predictor':<20} {'Condition':<12} {'Ch':<5} "
      f"{'b':>8} {'SE':>7} {'z':>7} {'p':>9} {'p_fdr':>9} {'sig':>4}")
print("  " + "-" * 85)

for _, row in lme_df[(lme_df["channel"].isin(["FCz", "Cz"])) &
                      (lme_df["p"] < 0.10)].sort_values("p").iterrows():
    sig = "***" if row["p_fdr"] < 0.001 else (
          "**" if row["p_fdr"] < 0.01 else (
          "*" if row["p_fdr"] < 0.05 else (
          "~" if row["p"] < 0.05 else "")))
    print(f"  {row['predictor']:<20} {row['condition']:<12} {row['channel']:<5} "
          f"{row['estimate']:>8.4f} {row['se']:>7.4f} {row['z']:>7.3f} "
          f"{row['p']:>9.4f} {row['p_fdr']:>9.4f} {sig:>4}")


# =====================================================================
# STEP 5: Condition comparisons via interaction models
# =====================================================================
print(f"\n{'=' * 70}")
print("STEP 5: Reward-level and Stay/Shift interaction models")
print("=" * 70)

interaction_results = []

# 5a: RPE × Reward Level interaction
for rpe_type in ["rpe_binary", "rpe_mag"]:
    if rpe_type not in trial_df.columns:
        continue
    for ch in ["FCz", "Cz"]:
        sub = trial_df.dropna(subset=[rpe_type, ch, "reward_level"]).copy()
        if len(sub) < 100:
            continue
        try:
            formula = f"{ch} ~ {rpe_type} * C(reward_level, Treatment(reference='no_rew'))"
            model = smf.mixedlm(formula, sub, groups=sub["subject"])
            result = model.fit(reml=True, method="powell")

            row = {"model": f"{rpe_type} x reward_level", "channel": ch,
                   "formula": formula, "n_obs": len(sub)}
            for name in result.params.index:
                if name == "Intercept":
                    continue
                row[f"{name}_b"] = float(result.params[name])
                row[f"{name}_p"] = float(result.pvalues[name])
            interaction_results.append(row)

            # Print interaction terms
            rpe_label = "binary" if "binary" in rpe_type else "mag"
            print(f"\n  {rpe_label} RPE x reward_level @ {ch}:")
            for name in result.params.index:
                if name == "Intercept":
                    continue
                sig = "*" if result.pvalues[name] < 0.05 else ""
                short_name = name.replace("C(reward_level, Treatment(reference='no_rew'))", "RL")
                print(f"    {short_name:<45} b={result.params[name]:>8.4f}  "
                      f"z={result.tvalues[name]:>7.3f}  p={result.pvalues[name]:.4f}{sig}")
        except Exception as e:
            print(f"  {rpe_type} x reward_level @ {ch}: FAILED ({e})")

# 5b: RPE × Stay/Shift interaction
for rpe_type in ["rpe_binary", "rpe_mag"]:
    if rpe_type not in trial_df.columns:
        continue
    for ch in ["FCz", "Cz"]:
        sub = trial_df[trial_df["PES_PRS_index"].isin([1, 2])].dropna(
            subset=[rpe_type, ch]).copy()
        if len(sub) < 100:
            continue
        try:
            formula = f"{ch} ~ {rpe_type} * C(PES_PRS_index)"
            model = smf.mixedlm(formula, sub, groups=sub["subject"])
            result = model.fit(reml=True, method="powell")

            rpe_label = "binary" if "binary" in rpe_type else "mag"
            print(f"\n  {rpe_label} RPE x stay/shift @ {ch}:")
            for name in result.params.index:
                if name == "Intercept":
                    continue
                sig = "*" if result.pvalues[name] < 0.05 else ""
                short_name = name.replace("C(PES_PRS_index)", "SS")
                print(f"    {short_name:<35} b={result.params[name]:>8.4f}  "
                      f"z={result.tvalues[name]:>7.3f}  p={result.pvalues[name]:.4f}{sig}")
        except Exception as e:
            print(f"  {rpe_type} x stay/shift @ {ch}: FAILED ({e})")

# 5c: RPE × Velocity Condition interaction
for rpe_type in ["rpe_binary", "rpe_mag"]:
    if rpe_type not in trial_df.columns:
        continue
    for ch in ["FCz", "Cz"]:
        sub = trial_df.dropna(subset=[rpe_type, ch, "Vel_Cond"]).copy()
        if len(sub) < 100:
            continue
        try:
            formula = f"{ch} ~ {rpe_type} * C(Vel_Cond)"
            model = smf.mixedlm(formula, sub, groups=sub["subject"])
            result = model.fit(reml=True, method="powell")

            rpe_label = "binary" if "binary" in rpe_type else "mag"
            print(f"\n  {rpe_label} RPE x Vel_Cond @ {ch}:")
            for name in result.params.index:
                if name == "Intercept":
                    continue
                sig = "*" if result.pvalues[name] < 0.05 else ""
                short_name = name.replace("C(Vel_Cond)", "VC")
                print(f"    {short_name:<35} b={result.params[name]:>8.4f}  "
                      f"z={result.tvalues[name]:>7.3f}  p={result.pvalues[name]:.4f}{sig}")
        except Exception as e:
            print(f"  {rpe_type} x Vel_Cond @ {ch}: FAILED ({e})")

# 5d: RPE × Condition_Direction_index (CDI) interaction → ERP
print(f"\n  --- CDI Interaction: ERP ~ RPE * C(cond_dir) ---")
cdi_interaction_results = []
cdi_sub = trial_df[trial_df["Condition_Direction_index"].isin([1, 2, 3, 4])].copy()
for rpe_type in ["rpe_binary", "rpe_mag"]:
    if rpe_type not in cdi_sub.columns:
        continue
    for ch in ["FCz", "Cz"]:
        sub = cdi_sub.dropna(subset=[rpe_type, ch]).copy()
        if len(sub) < 100:
            continue
        try:
            formula = f"{ch} ~ {rpe_type} * C(cond_dir, Treatment(reference='norew_stay'))"
            model = smf.mixedlm(formula, sub, groups=sub["subject"])
            result = model.fit(reml=True, method="powell")

            row = {"model": f"{rpe_type} x cond_dir", "channel": ch,
                   "formula": formula, "n_obs": len(sub)}
            for name in result.params.index:
                if name == "Intercept":
                    continue
                row[f"{name}_b"] = float(result.params[name])
                row[f"{name}_p"] = float(result.pvalues[name])
            cdi_interaction_results.append(row)
            interaction_results.append(row)

            rpe_label = "binary" if "binary" in rpe_type else "mag"
            print(f"\n  {rpe_label} RPE x CDI @ {ch}:")
            for name in result.params.index:
                if name == "Intercept":
                    continue
                sig = "*" if result.pvalues[name] < 0.05 else ""
                short_name = name.replace(
                    "C(cond_dir, Treatment(reference='norew_stay'))", "CDI")
                print(f"    {short_name:<50} b={result.params[name]:>8.4f}  "
                      f"z={result.tvalues[name]:>7.3f}  p={result.pvalues[name]:.4f}{sig}")
        except Exception as e:
            print(f"  {rpe_type} x CDI @ {ch}: FAILED ({e})")

# 5e: RPE × CDI interaction → RT_DIFF
print(f"\n  --- CDI Interaction: RT_DIFF ~ RPE * C(cond_dir) ---")
for rpe_type in ["rpe_binary", "rpe_mag"]:
    if rpe_type not in cdi_sub.columns:
        continue
    sub = cdi_sub.dropna(subset=[rpe_type, "RT_DIFF"]).copy()
    if len(sub) < 100:
        continue
    try:
        formula = f"RT_DIFF ~ {rpe_type} * C(cond_dir, Treatment(reference='norew_stay'))"
        model = smf.mixedlm(formula, sub, groups=sub["subject"])
        result = model.fit(reml=True, method="powell")

        row = {"model": f"{rpe_type} x cond_dir -> RT_DIFF", "channel": "RT_DIFF",
               "formula": formula, "n_obs": len(sub)}
        for name in result.params.index:
            if name == "Intercept":
                continue
            row[f"{name}_b"] = float(result.params[name])
            row[f"{name}_p"] = float(result.pvalues[name])
        interaction_results.append(row)

        rpe_label = "binary" if "binary" in rpe_type else "mag"
        print(f"\n  {rpe_label} RPE x CDI -> RT_DIFF:")
        for name in result.params.index:
            if name == "Intercept":
                continue
            sig = "*" if result.pvalues[name] < 0.05 else ""
            short_name = name.replace(
                "C(cond_dir, Treatment(reference='norew_stay'))", "CDI")
            print(f"    {short_name:<50} b={result.params[name]:>8.4f}  "
                  f"z={result.tvalues[name]:>7.3f}  p={result.pvalues[name]:.4f}{sig}")
    except Exception as e:
        print(f"  {rpe_type} x CDI -> RT_DIFF: FAILED ({e})")


# =====================================================================
# STEP 6: Post-outcome RT adjustment (LME)
# =====================================================================
print(f"\n{'=' * 70}")
print("STEP 6: Post-outcome walking speed (LME)")
print("=" * 70)

# Build RT change dataframe
rt_rows = []
for subj in subjects:
    sdata = trial_df[trial_df["subject"] == subj].sort_values("Trial")
    rts = sdata["Start_RT"].values
    trials = sdata["Trial"].values

    for t in range(1, len(rts)):
        if rts[t] < 20 and rts[t - 1] < 20:
            row = {
                "subject": subj,
                "Trial": int(trials[t]),
                "rt_change": rts[t] - rts[t - 1],
                "current_rt": rts[t],
                "prev_reward_level": sdata.iloc[t - 1]["reward_level"],
            }
            for col in ["rpe_binary", "rpe_mag", "abs_rpe_binary", "abs_rpe_mag"]:
                if col in sdata.columns:
                    row[f"prev_{col}"] = sdata.iloc[t - 1][col]
            rt_rows.append(row)

rt_df = pd.DataFrame(rt_rows)

# RPE → RT change LME
for rpe_col in ["prev_rpe_binary", "prev_rpe_mag",
                "prev_abs_rpe_binary", "prev_abs_rpe_mag"]:
    if rpe_col not in rt_df.columns:
        continue
    res = fit_lme(rt_df, "rt_change", rpe_col, use_trial_re=False)
    if res:
        sig = "*" if res["p"] < 0.05 else ("~" if res["p"] < 0.10 else "")
        print(f"  {rpe_col:<25} -> RT change: b={res['estimate']:>8.4f}, "
              f"z={res['z']:>7.3f}, p={res['p']:.4f}{sig}")

# RT change by previous reward level
for level in ["high_rew", "low_rew", "no_rew"]:
    sub = rt_df[rt_df["prev_reward_level"] == level]
    print(f"  After {level:<10}: mean RT change = {sub['rt_change'].mean():>+.4f}s "
          f"(n={len(sub)})")


# =====================================================================
# STEP 6b: RT_DIFF as outcome — RT_DIFF ~ behv + (1|subject) + (1|Trial)
# =====================================================================
print(f"\n{'=' * 70}")
print("STEP 6b: RT_DIFF as outcome: RT_DIFF ~ behv + (1|subject) + (1|Trial)")
print("=" * 70)

RTDIFF_PREDICTORS = [
    "rpe_binary", "rpe_mag", "abs_rpe_binary", "abs_rpe_mag",
    "chosen_q_binary", "chosen_q_mag", "is_reward", "reward_level_num",
    "Earned", "Start_RT",
]

rtdiff_results = []
for behav in RTDIFF_PREDICTORS:
    if behav not in trial_df.columns:
        continue
    for cond_name, cond_fn in CONDITIONS.items():
        subset = cond_fn(trial_df)
        res = fit_lme(subset, "RT_DIFF", behav, use_trial_re=True)
        if res is not None:
            rtdiff_results.append({
                "predictor": behav, "condition": cond_name, **res,
            })

rtdiff_df = pd.DataFrame(rtdiff_results)

if len(rtdiff_df) > 0:
    reject_rt, pvals_fdr_rt, _, _ = multipletests(
        rtdiff_df["p"].values, alpha=0.05, method="fdr_bh")
    rtdiff_df["p_fdr"] = pvals_fdr_rt
    rtdiff_df["sig_fdr"] = reject_rt
    rtdiff_df["sig_unc"] = rtdiff_df["p"] < 0.05

    n_sig_unc_rt = rtdiff_df["sig_unc"].sum()
    n_sig_fdr_rt = rtdiff_df["sig_fdr"].sum()
    print(f"  RT_DIFF models fit: {len(rtdiff_df)}")
    print(f"  Uncorrected p < .05: {n_sig_unc_rt}/{len(rtdiff_df)}")
    print(f"  FDR-corrected q < .05: {n_sig_fdr_rt}/{len(rtdiff_df)}")

    # Print key RT_DIFF results
    print(f"\n  {'Predictor':<20} {'Condition':<12} "
          f"{'b':>8} {'SE':>7} {'z':>7} {'p':>9} {'p_fdr':>9} {'sig':>4}")
    print("  " + "-" * 75)
    for _, row in rtdiff_df[rtdiff_df["p"] < 0.10].sort_values("p").iterrows():
        sig = "***" if row["p_fdr"] < 0.001 else (
              "**" if row["p_fdr"] < 0.01 else (
              "*" if row["p_fdr"] < 0.05 else (
              "~" if row["p"] < 0.05 else "")))
        print(f"  {row['predictor']:<20} {row['condition']:<12} "
              f"{row['estimate']:>8.4f} {row['se']:>7.4f} {row['z']:>7.3f} "
              f"{row['p']:>9.4f} {row['p_fdr']:>9.4f} {sig:>4}")

    # CDI-specific RT_DIFF summary
    print(f"\n  RT_DIFF by CDI condition:")
    for cdi in ["rew_stay", "rew_shift", "norew_stay", "norew_shift"]:
        cdi_data = trial_df[trial_df["cond_dir"] == cdi]["RT_DIFF"].dropna()
        print(f"    {cdi:<15}: mean={cdi_data.mean():>+.4f}s, "
              f"std={cdi_data.std():.4f}s, n={len(cdi_data)}")


# =====================================================================
# STEP 7: Figures
# =====================================================================
print(f"\n{'=' * 70}")
print("STEP 7: Generating figures")
print("=" * 70)

C_HIGH = "#2ca02c"
C_LOW = "#ff7f0e"
C_NOREW = "#d62728"

fig, axes = plt.subplots(3, 4, figsize=(24, 16))

# ── 7a: Model comparison ────────────────────────────────────────
ax = axes[0, 0]
model_counts = Counter(info_df["best_model"])
models_sorted = sorted(model_counts.keys(), key=lambda x: -model_counts[x])
ax.bar(range(len(models_sorted)), [model_counts[m] for m in models_sorted],
       color="steelblue")
ax.set_xticks(range(len(models_sorted)))
ax.set_xticklabels(models_sorted, rotation=45, ha="right", fontsize=9)
ax.set_ylabel("# Subjects")
ax.set_title("A. Best-Fitting Model per Subject")

# ── 7b: LME coefficients — RPE_binary × FCz by condition ────────
ax = axes[0, 1]
conds_plot = ["all", "high_rew", "low_rew", "no_rew", "any_rew"]
labels_plot = ["All", "High\nRew", "Low\nRew", "No\nRew", "Any\nRew"]
colors_plot = ["gray", C_HIGH, C_LOW, C_NOREW, "steelblue"]
coefs, ses = [], []
for c in conds_plot:
    row = lme_df[(lme_df["predictor"] == "rpe_binary") &
                  (lme_df["condition"] == c) & (lme_df["channel"] == "FCz")]
    if len(row) > 0:
        coefs.append(row.iloc[0]["estimate"])
        ses.append(row.iloc[0]["se"])
    else:
        coefs.append(0)
        ses.append(0)
ax.bar(range(len(conds_plot)), coefs, yerr=ses, color=colors_plot,
       alpha=0.8, capsize=4, edgecolor="black")
ax.axhline(0, ls="--", color="black", alpha=0.3)
ax.set_xticks(range(len(conds_plot)))
ax.set_xticklabels(labels_plot, fontsize=9)
ax.set_ylabel("LME coefficient (b)")
ax.set_title("B. Binary RPE → FCz (LME)")

# ── 7c: LME coefficients — RPE_mag × FCz by condition ───────────
ax = axes[0, 2]
coefs_m, ses_m = [], []
for c in conds_plot:
    row = lme_df[(lme_df["predictor"] == "rpe_mag") &
                  (lme_df["condition"] == c) & (lme_df["channel"] == "FCz")]
    if len(row) > 0:
        coefs_m.append(row.iloc[0]["estimate"])
        ses_m.append(row.iloc[0]["se"])
    else:
        coefs_m.append(0)
        ses_m.append(0)
ax.bar(range(len(conds_plot)), coefs_m, yerr=ses_m, color=colors_plot,
       alpha=0.8, capsize=4, edgecolor="black")
ax.axhline(0, ls="--", color="black", alpha=0.3)
ax.set_xticks(range(len(conds_plot)))
ax.set_xticklabels(labels_plot, fontsize=9)
ax.set_ylabel("LME coefficient (b)")
ax.set_title("C. Magnitude RPE → FCz (LME)")

# ── 7d: FCz LME coefficient across all predictors (all trials) ──
ax = axes[0, 3]
pred_coefs = []
pred_names = []
pred_pvals = []
for behav in BEHAV_PREDICTORS:
    row = lme_df[(lme_df["predictor"] == behav) &
                  (lme_df["condition"] == "all") & (lme_df["channel"] == "FCz")]
    if len(row) > 0:
        pred_coefs.append(row.iloc[0]["estimate"])
        pred_pvals.append(row.iloc[0]["p"])
        pred_names.append(behav.replace("_binary", "\n(bin)").replace("_mag", "\n(mag)"))
x = np.arange(len(pred_names))
colors_pred = ["steelblue" if p < 0.05 else "lightgray" for p in pred_pvals]
ax.barh(x, pred_coefs, color=colors_pred, edgecolor="navy", height=0.6)
for i, p in enumerate(pred_pvals):
    if p < 0.05:
        ax.text(pred_coefs[i], i, f" p={p:.3f}", va="center", fontsize=7)
ax.set_yticks(x)
ax.set_yticklabels(pred_names, fontsize=8)
ax.axvline(0, ls="--", color="black")
ax.set_xlabel("LME coefficient (b)")
ax.set_title("D. All Predictors → FCz (all trials)")

# ── 7e: Binned RPE × FCz (grand average) ────────────────────────
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
z_fit = np.polyfit(centers, bmeans, 1)
xfit = np.linspace(min(centers), max(centers), 100)
ax.plot(xfit, np.polyval(z_fit, xfit), "k--", alpha=0.5)
# Get LME p-value for annotation
lme_row = lme_df[(lme_df["predictor"] == "rpe_binary") &
                   (lme_df["condition"] == "all") & (lme_df["channel"] == "FCz")]
if len(lme_row) > 0:
    lme_b = lme_row.iloc[0]["estimate"]
    lme_p = lme_row.iloc[0]["p"]
    ax.set_title(f"E. Binary RPE × FCz (binned)\nLME: b={lme_b:.4f}, p={lme_p:.4f}")
else:
    ax.set_title("E. Binary RPE × FCz (binned)")
ax.set_xlabel("RPE (binary, decile bins)")
ax.set_ylabel("Mean FCz amplitude (uV)")

# ── 7f: Magnitude RPE × FCz (binned) ────────────────────────────
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
lme_row_m = lme_df[(lme_df["predictor"] == "rpe_mag") &
                     (lme_df["condition"] == "all") & (lme_df["channel"] == "FCz")]
if len(lme_row_m) > 0:
    ax.set_title(f"F. Mag RPE × FCz (binned)\nLME: b={lme_row_m.iloc[0]['estimate']:.4f}, "
                 f"p={lme_row_m.iloc[0]['p']:.4f}")
ax.set_xlabel("RPE (magnitude, decile bins)")
ax.set_ylabel("Mean FCz amplitude (uV)")

# ── 7g: FCz by Outcome × Stay/Shift ────────────────────────────
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
ax.set_ylabel("FCz amplitude (uV)")
ax.set_title("G. FCz by Outcome x Stay/Shift")

# ── 7h: Heatmap — binary RPE → all channels × conditions ────────
ax = axes[1, 3]
cond_list = list(CONDITIONS.keys())
mat = np.full((len(cond_list), len(CHANNELS)), np.nan)
mat_p = np.ones_like(mat)
for i, cond in enumerate(cond_list):
    for j, ch in enumerate(CHANNELS):
        row = lme_df[(lme_df["predictor"] == "rpe_binary") &
                      (lme_df["condition"] == cond) & (lme_df["channel"] == ch)]
        if len(row) > 0:
            mat[i, j] = row.iloc[0]["estimate"]
            mat_p[i, j] = row.iloc[0]["p"]
vmax = np.nanmax(np.abs(mat[np.isfinite(mat)])) if np.any(np.isfinite(mat)) else 1
im = ax.imshow(mat, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
ax.set_xticks(range(len(CHANNELS)))
ax.set_xticklabels(CHANNELS, rotation=45, ha="right", fontsize=8)
ax.set_yticks(range(len(cond_list)))
ax.set_yticklabels(cond_list, fontsize=9)
for i in range(len(cond_list)):
    for j in range(len(CHANNELS)):
        if np.isfinite(mat[i, j]):
            sig = "*" if mat_p[i, j] < 0.05 else ""
            color = "white" if abs(mat[i, j]) > vmax * 0.5 else "black"
            ax.text(j, i, f"{mat[i, j]:.2f}{sig}", ha="center", va="center",
                    fontsize=6, color=color)
plt.colorbar(im, ax=ax, label="LME b", shrink=0.8)
ax.set_title("H. Binary RPE LME (b) Heatmap")

# ── 7i: RPE distribution by reward level ────────────────────────
ax = axes[2, 0]
for vc, label, color in [(1, "High", C_HIGH), (3, "Low", C_LOW), (2, "None", C_NOREW)]:
    vals = trial_df[trial_df["Vel_Cond"] == vc]["rpe_mag"].dropna()
    ax.hist(vals, bins=30, alpha=0.5, color=color, label=label, edgecolor="black",
            linewidth=0.5)
ax.axvline(0, ls="--", color="black")
ax.set_xlabel("RPE (magnitude)")
ax.set_ylabel("Count")
ax.set_title("I. RPE Distribution by Reward Level")
ax.legend()

# ── 7j: Heatmap — magnitude RPE → all channels × conditions ─────
ax = axes[2, 1]
mat_m = np.full((len(cond_list), len(CHANNELS)), np.nan)
mat_mp = np.ones_like(mat_m)
for i, cond in enumerate(cond_list):
    for j, ch in enumerate(CHANNELS):
        row = lme_df[(lme_df["predictor"] == "rpe_mag") &
                      (lme_df["condition"] == cond) & (lme_df["channel"] == ch)]
        if len(row) > 0:
            mat_m[i, j] = row.iloc[0]["estimate"]
            mat_mp[i, j] = row.iloc[0]["p"]
vmax_m = np.nanmax(np.abs(mat_m[np.isfinite(mat_m)])) if np.any(np.isfinite(mat_m)) else 1
im2 = ax.imshow(mat_m, cmap="RdBu_r", vmin=-vmax_m, vmax=vmax_m, aspect="auto")
ax.set_xticks(range(len(CHANNELS)))
ax.set_xticklabels(CHANNELS, rotation=45, ha="right", fontsize=8)
ax.set_yticks(range(len(cond_list)))
ax.set_yticklabels(cond_list, fontsize=9)
for i in range(len(cond_list)):
    for j in range(len(CHANNELS)):
        if np.isfinite(mat_m[i, j]):
            sig = "*" if mat_mp[i, j] < 0.05 else ""
            color = "white" if abs(mat_m[i, j]) > vmax_m * 0.5 else "black"
            ax.text(j, i, f"{mat_m[i, j]:.2f}{sig}", ha="center", va="center",
                    fontsize=6, color=color)
plt.colorbar(im2, ax=ax, label="LME b", shrink=0.8)
ax.set_title("J. Magnitude RPE LME (b) Heatmap")

# ── 7k: Walking RT by reward level ──────────────────────────────
ax = axes[2, 2]
rt_means = []
rt_sems = []
for vc in [1, 3, 2]:
    sub = trial_df[(trial_df["Vel_Cond"] == vc) & (trial_df["Start_RT"] < 15)]
    rt_means.append(sub["Start_RT"].mean())
    rt_sems.append(sub["Start_RT"].std() / np.sqrt(len(sub)))
ax.bar([0, 1, 2], rt_means, yerr=rt_sems,
       color=[C_HIGH, C_LOW, C_NOREW], alpha=0.8, capsize=4, edgecolor="black")
ax.set_xticks([0, 1, 2])
ax.set_xticklabels(["High ($0.25)", "Low ($0.05)", "None ($0.00)"], fontsize=9)
ax.set_ylabel("Mean Walking RT (sec)")
ax.set_title("K. Walking Speed by Reward Level")

# ── 7l: Learning rate distribution ──────────────────────────────
ax = axes[2, 3]
ax.hist(info_df["rw_alpha_binary"].dropna(), bins=20, color="coral",
        edgecolor="black", alpha=0.6, label="Binary")
ax.hist(info_df["rw_alpha_mag"].dropna(), bins=20, color="purple",
        edgecolor="black", alpha=0.4, label="Magnitude")
ax.set_xlabel("RW Learning Rate (alpha)")
ax.set_ylabel("# Subjects")
ax.set_title(f"L. Learning Rate Distribution\n"
             f"(bin M={info_df['rw_alpha_binary'].mean():.3f}, "
             f"mag M={info_df['rw_alpha_mag'].mean():.3f})")
ax.legend()

plt.suptitle("T-Maze AR: LME Neural-Behavioral Fusion (N=43)\n"
             "ERP ~ behv + (1|subject) + (1|Trial)",
             fontsize=16, y=1.01)
plt.tight_layout()
fig_path = os.path.join(OUTPUT_DIR, "12_lme_tmaze_pipeline.png")
plt.savefig(fig_path, dpi=150, bbox_inches="tight")
print(f"  Main figure: {fig_path}")
plt.close()

# ── Second figure: CDI and RT_DIFF analyses ──────────────────────
fig2, axes2 = plt.subplots(3, 4, figsize=(24, 16))

# ── 2a: FCz by CDI condition ────────────────────────────────────
ax = axes2[0, 0]
cdi_labels = ["Rew\nStay", "Rew\nShift", "NoRew\nStay", "NoRew\nShift"]
cdi_codes = ["rew_stay", "rew_shift", "norew_stay", "norew_shift"]
cdi_colors = [C_HIGH, "#90EE90", C_NOREW, "#FFB6C1"]
cdi_means_fcz = [trial_df[trial_df["cond_dir"] == c]["FCz"].mean() for c in cdi_codes]
cdi_sems_fcz = [trial_df[trial_df["cond_dir"] == c]["FCz"].std() /
                np.sqrt(len(trial_df[trial_df["cond_dir"] == c])) for c in cdi_codes]
ax.bar(range(4), cdi_means_fcz, yerr=cdi_sems_fcz, color=cdi_colors,
       capsize=4, edgecolor="black", alpha=0.8)
ax.axhline(0, ls="--", color="black", alpha=0.3)
ax.set_xticks(range(4))
ax.set_xticklabels(cdi_labels, fontsize=9)
ax.set_ylabel("FCz amplitude (uV)")
ax.set_title("A. FCz by Condition_Direction_index")

# ── 2b: Cz by CDI condition ────────────────────────────────────
ax = axes2[0, 1]
cdi_means_cz = [trial_df[trial_df["cond_dir"] == c]["Cz"].mean() for c in cdi_codes]
cdi_sems_cz = [trial_df[trial_df["cond_dir"] == c]["Cz"].std() /
               np.sqrt(len(trial_df[trial_df["cond_dir"] == c])) for c in cdi_codes]
ax.bar(range(4), cdi_means_cz, yerr=cdi_sems_cz, color=cdi_colors,
       capsize=4, edgecolor="black", alpha=0.8)
ax.axhline(0, ls="--", color="black", alpha=0.3)
ax.set_xticks(range(4))
ax.set_xticklabels(cdi_labels, fontsize=9)
ax.set_ylabel("Cz amplitude (uV)")
ax.set_title("B. Cz by Condition_Direction_index")

# ── 2c: RT_DIFF by CDI condition ────────────────────────────────
ax = axes2[0, 2]
cdi_means_rt = [trial_df[trial_df["cond_dir"] == c]["RT_DIFF"].dropna().mean() for c in cdi_codes]
cdi_sems_rt = [trial_df[trial_df["cond_dir"] == c]["RT_DIFF"].dropna().std() /
               np.sqrt(len(trial_df[trial_df["cond_dir"] == c]["RT_DIFF"].dropna())) for c in cdi_codes]
ax.bar(range(4), cdi_means_rt, yerr=cdi_sems_rt, color=cdi_colors,
       capsize=4, edgecolor="black", alpha=0.8)
ax.axhline(0, ls="--", color="black", alpha=0.3)
ax.set_xticks(range(4))
ax.set_xticklabels(cdi_labels, fontsize=9)
ax.set_ylabel("RT_DIFF (prev - current, sec)")
ax.set_title("C. RT_DIFF by Condition_Direction_index")

# ── 2d: RT_DIFF LME coefficients — all predictors (all trials) ──
ax = axes2[0, 3]
if len(rtdiff_df) > 0:
    rt_all = rtdiff_df[rtdiff_df["condition"] == "all"].copy()
    if len(rt_all) > 0:
        rt_all = rt_all.sort_values("p")
        pred_names_rt = rt_all["predictor"].values
        pred_coefs_rt = rt_all["estimate"].values
        pred_pvals_rt = rt_all["p"].values
        colors_rt = ["steelblue" if p < 0.05 else "lightgray" for p in pred_pvals_rt]
        y_pos = np.arange(len(pred_names_rt))
        ax.barh(y_pos, pred_coefs_rt, color=colors_rt, edgecolor="navy", height=0.6)
        for i, p in enumerate(pred_pvals_rt):
            if p < 0.05:
                ax.text(pred_coefs_rt[i], i, f" p={p:.3f}", va="center", fontsize=7)
        ax.set_yticks(y_pos)
        short = [n.replace("_binary", "\n(bin)").replace("_mag", "\n(mag)") for n in pred_names_rt]
        ax.set_yticklabels(short, fontsize=8)
        ax.axvline(0, ls="--", color="black")
ax.set_xlabel("LME coefficient (b)")
ax.set_title("D. All Predictors -> RT_DIFF (all trials)")

# ── 2e: Binary RPE → FCz by CDI condition (LME b) ───────────────
ax = axes2[1, 0]
cdi_conds = ["rew_stay", "rew_shift", "norew_stay", "norew_shift"]
cdi_coefs_b = []
cdi_ses_b = []
for c in cdi_conds:
    row = lme_df[(lme_df["predictor"] == "rpe_binary") &
                  (lme_df["condition"] == c) & (lme_df["channel"] == "FCz")]
    if len(row) > 0:
        cdi_coefs_b.append(row.iloc[0]["estimate"])
        cdi_ses_b.append(row.iloc[0]["se"])
    else:
        cdi_coefs_b.append(0)
        cdi_ses_b.append(0)
ax.bar(range(4), cdi_coefs_b, yerr=cdi_ses_b, color=cdi_colors,
       capsize=4, edgecolor="black", alpha=0.8)
ax.axhline(0, ls="--", color="black", alpha=0.3)
ax.set_xticks(range(4))
ax.set_xticklabels(cdi_labels, fontsize=9)
ax.set_ylabel("LME coefficient (b)")
ax.set_title("E. Binary RPE -> FCz (LME) by CDI")

# ── 2f: Magnitude RPE → FCz by CDI condition (LME b) ────────────
ax = axes2[1, 1]
cdi_coefs_m = []
cdi_ses_m = []
for c in cdi_conds:
    row = lme_df[(lme_df["predictor"] == "rpe_mag") &
                  (lme_df["condition"] == c) & (lme_df["channel"] == "FCz")]
    if len(row) > 0:
        cdi_coefs_m.append(row.iloc[0]["estimate"])
        cdi_ses_m.append(row.iloc[0]["se"])
    else:
        cdi_coefs_m.append(0)
        cdi_ses_m.append(0)
ax.bar(range(4), cdi_coefs_m, yerr=cdi_ses_m, color=cdi_colors,
       capsize=4, edgecolor="black", alpha=0.8)
ax.axhline(0, ls="--", color="black", alpha=0.3)
ax.set_xticks(range(4))
ax.set_xticklabels(cdi_labels, fontsize=9)
ax.set_ylabel("LME coefficient (b)")
ax.set_title("F. Magnitude RPE -> FCz (LME) by CDI")

# ── 2g: Binary RPE → RT_DIFF by CDI condition (LME b) ───────────
ax = axes2[1, 2]
if len(rtdiff_df) > 0:
    cdi_coefs_rtb = []
    cdi_ses_rtb = []
    for c in cdi_conds:
        row = rtdiff_df[(rtdiff_df["predictor"] == "rpe_binary") &
                         (rtdiff_df["condition"] == c)]
        if len(row) > 0:
            cdi_coefs_rtb.append(row.iloc[0]["estimate"])
            cdi_ses_rtb.append(row.iloc[0]["se"])
        else:
            cdi_coefs_rtb.append(0)
            cdi_ses_rtb.append(0)
    ax.bar(range(4), cdi_coefs_rtb, yerr=cdi_ses_rtb, color=cdi_colors,
           capsize=4, edgecolor="black", alpha=0.8)
    ax.axhline(0, ls="--", color="black", alpha=0.3)
    ax.set_xticks(range(4))
    ax.set_xticklabels(cdi_labels, fontsize=9)
ax.set_ylabel("LME coefficient (b)")
ax.set_title("G. Binary RPE -> RT_DIFF (LME) by CDI")

# ── 2h: Magnitude RPE → RT_DIFF by CDI condition (LME b) ────────
ax = axes2[1, 3]
if len(rtdiff_df) > 0:
    cdi_coefs_rtm = []
    cdi_ses_rtm = []
    for c in cdi_conds:
        row = rtdiff_df[(rtdiff_df["predictor"] == "rpe_mag") &
                         (rtdiff_df["condition"] == c)]
        if len(row) > 0:
            cdi_coefs_rtm.append(row.iloc[0]["estimate"])
            cdi_ses_rtm.append(row.iloc[0]["se"])
        else:
            cdi_coefs_rtm.append(0)
            cdi_ses_rtm.append(0)
    ax.bar(range(4), cdi_coefs_rtm, yerr=cdi_ses_rtm, color=cdi_colors,
           capsize=4, edgecolor="black", alpha=0.8)
    ax.axhline(0, ls="--", color="black", alpha=0.3)
    ax.set_xticks(range(4))
    ax.set_xticklabels(cdi_labels, fontsize=9)
ax.set_ylabel("LME coefficient (b)")
ax.set_title("H. Magnitude RPE -> RT_DIFF (LME) by CDI")

# ── 2i: Heatmap — RT_DIFF LME (b) across predictors × conditions ─
ax = axes2[2, 0]
if len(rtdiff_df) > 0:
    rt_cond_list = list(CONDITIONS.keys())
    rt_pred_list = RTDIFF_PREDICTORS
    mat_rt = np.full((len(rt_cond_list), len(rt_pred_list)), np.nan)
    mat_rtp = np.ones_like(mat_rt)
    for i, cond in enumerate(rt_cond_list):
        for j, pred in enumerate(rt_pred_list):
            row = rtdiff_df[(rtdiff_df["predictor"] == pred) &
                             (rtdiff_df["condition"] == cond)]
            if len(row) > 0:
                mat_rt[i, j] = row.iloc[0]["estimate"]
                mat_rtp[i, j] = row.iloc[0]["p"]
    vmax_rt = np.nanmax(np.abs(mat_rt[np.isfinite(mat_rt)])) if np.any(np.isfinite(mat_rt)) else 1
    im_rt = ax.imshow(mat_rt, cmap="RdBu_r", vmin=-vmax_rt, vmax=vmax_rt, aspect="auto")
    ax.set_xticks(range(len(rt_pred_list)))
    short_preds = [p.replace("_binary", "\n(bin)").replace("_mag", "\n(mag)")
                   for p in rt_pred_list]
    ax.set_xticklabels(short_preds, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(len(rt_cond_list)))
    ax.set_yticklabels(rt_cond_list, fontsize=8)
    for i in range(len(rt_cond_list)):
        for j in range(len(rt_pred_list)):
            if np.isfinite(mat_rt[i, j]):
                sig = "*" if mat_rtp[i, j] < 0.05 else ""
                color = "white" if abs(mat_rt[i, j]) > vmax_rt * 0.5 else "black"
                ax.text(j, i, f"{mat_rt[i, j]:.3f}{sig}", ha="center", va="center",
                        fontsize=5, color=color)
    plt.colorbar(im_rt, ax=ax, label="LME b", shrink=0.8)
ax.set_title("I. RT_DIFF LME Heatmap (pred x cond)")

# ── 2j: Heatmap — binary RPE LME for CDI conditions only ─────────
ax = axes2[2, 1]
cdi_only = ["rew_stay", "rew_shift", "norew_stay", "norew_shift"]
mat_cdi = np.full((len(cdi_only), len(CHANNELS)), np.nan)
mat_cdip = np.ones_like(mat_cdi)
for i, cond in enumerate(cdi_only):
    for j, ch in enumerate(CHANNELS):
        row = lme_df[(lme_df["predictor"] == "rpe_binary") &
                      (lme_df["condition"] == cond) & (lme_df["channel"] == ch)]
        if len(row) > 0:
            mat_cdi[i, j] = row.iloc[0]["estimate"]
            mat_cdip[i, j] = row.iloc[0]["p"]
vmax_cdi = np.nanmax(np.abs(mat_cdi[np.isfinite(mat_cdi)])) if np.any(np.isfinite(mat_cdi)) else 1
im_cdi = ax.imshow(mat_cdi, cmap="RdBu_r", vmin=-vmax_cdi, vmax=vmax_cdi, aspect="auto")
ax.set_xticks(range(len(CHANNELS)))
ax.set_xticklabels(CHANNELS, rotation=45, ha="right", fontsize=8)
ax.set_yticks(range(len(cdi_only)))
ax.set_yticklabels(cdi_only, fontsize=9)
for i in range(len(cdi_only)):
    for j in range(len(CHANNELS)):
        if np.isfinite(mat_cdi[i, j]):
            sig = "*" if mat_cdip[i, j] < 0.05 else ""
            color = "white" if abs(mat_cdi[i, j]) > vmax_cdi * 0.5 else "black"
            ax.text(j, i, f"{mat_cdi[i, j]:.2f}{sig}", ha="center", va="center",
                    fontsize=6, color=color)
plt.colorbar(im_cdi, ax=ax, label="LME b", shrink=0.8)
ax.set_title("J. Binary RPE -> ERP (CDI conditions)")

# ── 2k: Heatmap — magnitude RPE LME for CDI conditions ──────────
ax = axes2[2, 2]
mat_cdim = np.full((len(cdi_only), len(CHANNELS)), np.nan)
mat_cdimp = np.ones_like(mat_cdim)
for i, cond in enumerate(cdi_only):
    for j, ch in enumerate(CHANNELS):
        row = lme_df[(lme_df["predictor"] == "rpe_mag") &
                      (lme_df["condition"] == cond) & (lme_df["channel"] == ch)]
        if len(row) > 0:
            mat_cdim[i, j] = row.iloc[0]["estimate"]
            mat_cdimp[i, j] = row.iloc[0]["p"]
vmax_cdim = np.nanmax(np.abs(mat_cdim[np.isfinite(mat_cdim)])) if np.any(np.isfinite(mat_cdim)) else 1
im_cdim = ax.imshow(mat_cdim, cmap="RdBu_r", vmin=-vmax_cdim, vmax=vmax_cdim, aspect="auto")
ax.set_xticks(range(len(CHANNELS)))
ax.set_xticklabels(CHANNELS, rotation=45, ha="right", fontsize=8)
ax.set_yticks(range(len(cdi_only)))
ax.set_yticklabels(cdi_only, fontsize=9)
for i in range(len(cdi_only)):
    for j in range(len(CHANNELS)):
        if np.isfinite(mat_cdim[i, j]):
            sig = "*" if mat_cdimp[i, j] < 0.05 else ""
            color = "white" if abs(mat_cdim[i, j]) > vmax_cdim * 0.5 else "black"
            ax.text(j, i, f"{mat_cdim[i, j]:.2f}{sig}", ha="center", va="center",
                    fontsize=6, color=color)
plt.colorbar(im_cdim, ax=ax, label="LME b", shrink=0.8)
ax.set_title("K. Magnitude RPE -> ERP (CDI conditions)")

# ── 2l: RT_DIFF distribution by CDI ─────────────────────────────
ax = axes2[2, 3]
for cdi_code, label, color in zip(cdi_codes, cdi_labels,
                                   [C_HIGH, "#90EE90", C_NOREW, "#FFB6C1"]):
    vals = trial_df[trial_df["cond_dir"] == cdi_code]["RT_DIFF"].dropna()
    ax.hist(vals, bins=30, alpha=0.4, color=color,
            label=label.replace("\n", " "), edgecolor="black", linewidth=0.3)
ax.axvline(0, ls="--", color="black")
ax.set_xlabel("RT_DIFF (prev - current RT, sec)")
ax.set_ylabel("Count")
ax.set_title("L. RT_DIFF Distribution by CDI")
ax.legend(fontsize=8)

plt.suptitle("T-Maze AR: CDI & RT_DIFF Analysis (N=43)\n"
             "Condition_Direction_index: rew_stay / rew_shift / norew_stay / norew_shift",
             fontsize=14, y=1.01)
plt.tight_layout()
fig2_path = os.path.join(OUTPUT_DIR, "12_lme_cdi_rtdiff.png")
plt.savefig(fig2_path, dpi=150, bbox_inches="tight")
print(f"  CDI/RT_DIFF figure: {fig2_path}")
plt.close()


# =====================================================================
# STEP 8: Save all results
# =====================================================================
print(f"\n{'=' * 70}")
print("STEP 8: Saving results")
print("=" * 70)

lme_df.to_csv(os.path.join(OUTPUT_DIR, "lme_all_results.csv"), index=False)
trial_df.to_csv(os.path.join(OUTPUT_DIR, "trial_data_with_rpes.csv"), index=False)
info_df.to_csv(os.path.join(OUTPUT_DIR, "subject_info.csv"), index=False)
if interaction_results:
    pd.DataFrame(interaction_results).to_csv(
        os.path.join(OUTPUT_DIR, "interaction_models.csv"), index=False)
if len(rtdiff_df) > 0:
    rtdiff_df.to_csv(os.path.join(OUTPUT_DIR, "rtdiff_lme_results.csv"), index=False)

print(f"  lme_all_results.csv       ({len(lme_df)} models)")
print(f"  trial_data_with_rpes.csv  ({len(trial_df)} trials)")
print(f"  subject_info.csv          ({len(info_df)} subjects)")
if len(rtdiff_df) > 0:
    print(f"  rtdiff_lme_results.csv    ({len(rtdiff_df)} models)")


# =====================================================================
# SUMMARY
# =====================================================================
print(f"\n{'=' * 70}")
print("SUMMARY TABLE FOR PAPER")
print("=" * 70)
print(f"\n  N = {len(subjects)} subjects, {len(trial_df)} single trials")
print(f"  Model: ERP ~ behv + (1 | subject) + (1 | Trial)")
print(f"  Total LME models fit: {len(lme_df)}")
if len(lme_df) > 0:
    print(f"  Uncorrected p < .05: {lme_df['sig_unc'].sum()}")
    print(f"  FDR-corrected q < .05: {lme_df['sig_fdr'].sum()}")

    print(f"\n  FDR-significant effects:")
    for _, row in lme_df[lme_df["sig_fdr"]].sort_values("p").iterrows():
        print(f"    {row['predictor']:<20} {row['condition']:<12} @ {row['channel']:<5} "
              f"b={row['estimate']:.4f}, z={row['z']:.2f}, "
              f"p={row['p']:.4f}, q={row['p_fdr']:.4f}")

    if not lme_df["sig_fdr"].any():
        print(f"\n  Strongest uncorrected effects (p < .05):")
        top = lme_df[lme_df["sig_unc"]].sort_values("p").head(10)
        for _, row in top.iterrows():
            print(f"    {row['predictor']:<20} {row['condition']:<12} @ {row['channel']:<5} "
                  f"b={row['estimate']:.4f}, z={row['z']:.2f}, p={row['p']:.4f}")

# RT_DIFF summary
if len(rtdiff_df) > 0:
    print(f"\n  ── RT_DIFF as Outcome ──")
    print(f"  RT_DIFF models fit: {len(rtdiff_df)}")
    print(f"  Uncorrected p < .05: {rtdiff_df['sig_unc'].sum()}")
    print(f"  FDR-corrected q < .05: {rtdiff_df['sig_fdr'].sum()}")
    if rtdiff_df["sig_fdr"].any():
        print(f"\n  FDR-significant RT_DIFF effects:")
        for _, row in rtdiff_df[rtdiff_df["sig_fdr"]].sort_values("p").iterrows():
            print(f"    {row['predictor']:<20} {row['condition']:<12} "
                  f"b={row['estimate']:.4f}, z={row['z']:.2f}, "
                  f"p={row['p']:.4f}, q={row['p_fdr']:.4f}")

    # CDI-specific RT_DIFF summary
    print(f"\n  RT_DIFF descriptive stats by CDI:")
    for cdi in ["rew_stay", "rew_shift", "norew_stay", "norew_shift"]:
        cdi_data = trial_df[trial_df["cond_dir"] == cdi]["RT_DIFF"].dropna()
        print(f"    {cdi:<15}: M={cdi_data.mean():>+.4f}s, SD={cdi_data.std():.4f}s, n={len(cdi_data)}")

print(f"\n  Output directory: {OUTPUT_DIR}")
print("=" * 70)
