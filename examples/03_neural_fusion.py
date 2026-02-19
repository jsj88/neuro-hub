"""
Example 3: Neural data fusion — linking RPEs to EEG (REWP).

Demonstrates the full pipeline:
1. Simulate behavioral data with a reward model
2. Fit the model and extract trial-level RPEs
3. Simulate EEG with RPE-coupled REWP signal
4. Correlate RPEs with REWP amplitudes
5. Run temporal decoding to find when reward info emerges

Usage:
    cd ~/neuro-hub
    python examples/03_neural_fusion.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from agent.tools.simulate import simulate_q_dual
from agent.tools.model_fitting import fit_model, extract_trial_variables

# ── Configuration ──────────────────────────────────────────────
REWARD_PROBS = np.array([0.7, 0.3])
N_TRIALS = 200
SEED = 42
SFREQ = 200.0         # Hz
TMIN, TMAX = -0.2, 0.8  # seconds
REWP_TMIN, REWP_TMAX = 0.240, 0.340  # REWP window
REWP_AMP = 3.0        # uV
RPE_COUPLING = 5.0    # RPE-to-REWP scaling
NOISE_STD = 10.0       # uV
N_CHANNELS = 5         # frontocentral subset
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "examples")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Step 1: Simulate behavioral data ──────────────────────────
print("Step 1: Simulating behavioral data...")
sim = simulate_q_dual(
    N_TRIALS, 2, REWARD_PROBS,
    alpha_pos=0.5, alpha_neg=0.1, beta=5.0, seed=SEED,
)
print(f"  Optimal choice: {np.mean(sim['choices'] == 0) * 100:.1f}%")

# ── Step 2: Fit model and extract RPEs ────────────────────────
print("Step 2: Fitting model and extracting RPEs...")
fit = fit_model("q_dual", sim["choices"], sim["outcomes"],
                n_options=2, method="multistart", n_starts=10)
latent = extract_trial_variables("q_dual", fit.params,
                                  sim["choices"], sim["outcomes"])
rpes = latent["rpes"]
print(f"  Best-fit params: {fit.params}")

# ── Step 3: Simulate EEG with RPE-coupled REWP ────────────────
print("Step 3: Simulating EEG data with RPE-coupled REWP...")
rng = np.random.default_rng(SEED)
n_times = int((TMAX - TMIN) * SFREQ)
times = np.linspace(TMIN, TMAX, n_times)

# Create labels: reward (1) vs no-reward (0)
labels = (sim["outcomes"] > 0).astype(int)

# Generate noisy EEG
eeg_data = rng.normal(0, NOISE_STD, (N_TRIALS, N_CHANNELS, n_times))

# Add REWP signal to reward trials, modulated by RPE
rewp_mask = (times >= REWP_TMIN) & (times <= REWP_TMAX)
for i in range(N_TRIALS):
    if labels[i] == 1:
        amp = REWP_AMP + RPE_COUPLING * rpes[i]
        eeg_data[i, :, rewp_mask] += amp

print(f"  EEG shape: {eeg_data.shape}")
print(f"  Reward trials: {labels.sum()}/{N_TRIALS}")

# ── Step 4: Extract REWP amplitudes and correlate with RPEs ───
print("Step 4: Correlating RPEs with REWP amplitudes...")
# Mean amplitude at FCz (channel 0) in REWP window
rewp_amplitudes = eeg_data[:, 0, rewp_mask].mean(axis=1)

# Full correlation (all trials)
r_all, p_all = pearsonr(rpes, rewp_amplitudes)
print(f"  All trials:     r={r_all:.3f}, p={p_all:.4f}")

# Reward trials only
rew_idx = labels == 1
r_rew, p_rew = pearsonr(rpes[rew_idx], rewp_amplitudes[rew_idx])
print(f"  Reward trials:  r={r_rew:.3f}, p={p_rew:.4f}")

# ── Step 5: Time-resolved analysis ────────────────────────────
print("Step 5: Time-resolved RPE-EEG correlation...")
r_timecourse = np.zeros(n_times)
p_timecourse = np.zeros(n_times)
for ti in range(n_times):
    r_timecourse[ti], p_timecourse[ti] = pearsonr(
        rpes, eeg_data[:, 0, ti]
    )

# ── Visualize ─────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 5a: RPE-REWP scatter
ax = axes[0, 0]
ax.scatter(rpes[rew_idx], rewp_amplitudes[rew_idx],
           alpha=0.5, color="green", label="Reward")
ax.scatter(rpes[~rew_idx], rewp_amplitudes[~rew_idx],
           alpha=0.5, color="red", label="No reward")
z = np.polyfit(rpes, rewp_amplitudes, 1)
xline = np.linspace(rpes.min(), rpes.max(), 100)
ax.plot(xline, np.polyval(z, xline), "k-", linewidth=2)
ax.set_xlabel("Prediction Error (RPE)")
ax.set_ylabel("REWP Amplitude (uV)")
ax.set_title(f"RPE-REWP Correlation (r={r_all:.3f}, p={p_all:.4f})")
ax.legend()

# 5b: Time-resolved correlation
ax = axes[0, 1]
ax.plot(times * 1000, r_timecourse, color="blue", linewidth=1.5)
ax.fill_between(times * 1000, 0, r_timecourse,
                where=p_timecourse < 0.05, alpha=0.3, color="blue",
                label="p < 0.05")
ax.axhline(0, ls="--", color="black", alpha=0.3)
ax.axvline(REWP_TMIN * 1000, ls=":", color="red", alpha=0.5)
ax.axvline(REWP_TMAX * 1000, ls=":", color="red", alpha=0.5,
           label="REWP window")
ax.set_xlabel("Time (ms)")
ax.set_ylabel("Pearson r (RPE vs EEG)")
ax.set_title("Time-Resolved RPE-EEG Correlation")
ax.legend()

# 5c: ERP by condition
ax = axes[1, 0]
erp_rew = eeg_data[rew_idx, 0, :].mean(axis=0)
erp_norew = eeg_data[~rew_idx, 0, :].mean(axis=0)
ax.plot(times * 1000, erp_rew, color="green", label="Reward", linewidth=2)
ax.plot(times * 1000, erp_norew, color="red", label="No Reward", linewidth=2)
ax.plot(times * 1000, erp_rew - erp_norew, color="black",
        label="Difference (REWP)", linewidth=2, linestyle="--")
ax.axvspan(REWP_TMIN * 1000, REWP_TMAX * 1000, alpha=0.1, color="yellow")
ax.set_xlabel("Time (ms)")
ax.set_ylabel("Amplitude (uV)")
ax.set_title("ERP at FCz: Reward vs No-Reward")
ax.legend()

# 5d: RPE histogram by outcome
ax = axes[1, 1]
ax.hist(rpes[rew_idx], bins=20, alpha=0.6, color="green", label="Reward trials")
ax.hist(rpes[~rew_idx], bins=20, alpha=0.6, color="red", label="No-reward trials")
ax.axvline(0, ls="--", color="black")
ax.set_xlabel("Prediction Error")
ax.set_ylabel("Count")
ax.set_title("RPE Distribution by Outcome")
ax.legend()

plt.suptitle("Neural Data Fusion: RPE-REWP Pipeline", fontsize=16, y=1.02)
plt.tight_layout()
out_path = os.path.join(OUTPUT_DIR, "03_neural_fusion.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"\nFigure saved: {out_path}")
plt.close()
