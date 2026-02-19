"""
Example 1: Simulate behavioral data and fit reward learning models.

This script demonstrates the core workflow:
1. Simulate a 2-armed bandit task with known parameters
2. Fit all 12 models to the data
3. Compare models via BIC
4. Visualize results

Usage:
    cd ~/neuro-hub
    python examples/01_simulate_and_fit.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from agent.tools.simulate import simulate_rw, simulate_q_dual, simulate_actor_critic
from agent.tools.model_fitting import fit_model, MODEL_SPECS, extract_trial_variables

# ── Configuration ──────────────────────────────────────────────
REWARD_PROBS = np.array([0.7, 0.3])  # 2-armed bandit
N_TRIALS = 200
SEED = 42
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "examples")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Step 1: Simulate data from Q-learning dual LR ─────────────
print("Step 1: Simulating Q-learning (dual LR) agent...")
true_params = {"alpha_pos": 0.5, "alpha_neg": 0.1, "beta": 5.0}
sim = simulate_q_dual(
    N_TRIALS, 2, REWARD_PROBS,
    alpha_pos=true_params["alpha_pos"],
    alpha_neg=true_params["alpha_neg"],
    beta=true_params["beta"],
    seed=SEED,
)
print(f"  Optimal choice rate: {np.mean(sim['choices'] == 0) * 100:.1f}%")
print(f"  Mean reward: {np.mean(sim['outcomes']):.3f}")

# ── Step 2: Fit all bandit models ──────────────────────────────
print("\nStep 2: Fitting all models...")
bandit_models = ["random", "wsls", "rw", "q_learning", "q_dual",
                 "actor_critic", "q_decay", "rw_bias", "ck", "rwck"]

results = []
for model_name in bandit_models:
    r = fit_model(model_name, sim["choices"], sim["outcomes"],
                  n_options=2, method="multistart", n_starts=10)
    results.append(r)
    print(f"  {r.model:<15} NLL={r.nll:>7.2f}  BIC={r.bic:>7.2f}  {r.params}")

# ── Step 3: Model comparison ──────────────────────────────────
results.sort(key=lambda x: x.bic)
best = results[0]
print(f"\nStep 3: Model comparison")
print(f"  Best model: {best.model} (BIC={best.bic:.2f})")
print(f"  Recovered params: {best.params}")
print(f"  True params: {true_params}")

# ── Step 4: Extract trial-level latent variables ───────────────
print("\nStep 4: Extracting trial-level RPEs from best model...")
latent = extract_trial_variables(best.model, best.params,
                                  sim["choices"], sim["outcomes"])
rpes = latent["rpes"]
print(f"  RPE range: [{rpes.min():.3f}, {rpes.max():.3f}]")
print(f"  Mean |RPE|: {np.mean(np.abs(rpes)):.3f}")

# ── Step 5: Visualize ─────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 5a: Q-values over trials
ax = axes[0, 0]
if "q_values" in latent:
    q = latent["q_values"]
    ax.plot(q[:, 0], label="Q(option 0) [p=0.7]", color="blue")
    ax.plot(q[:, 1], label="Q(option 1) [p=0.3]", color="red")
    ax.axhline(0.7, ls="--", color="blue", alpha=0.3)
    ax.axhline(0.3, ls="--", color="red", alpha=0.3)
ax.set_xlabel("Trial")
ax.set_ylabel("Q-value")
ax.set_title("Value Evolution")
ax.legend()

# 5b: RPEs
ax = axes[0, 1]
ax.plot(rpes, alpha=0.5, color="green")
ax.axhline(0, ls="--", color="black", alpha=0.3)
ax.set_xlabel("Trial")
ax.set_ylabel("Prediction Error")
ax.set_title("Trial-by-Trial RPEs")

# 5c: BIC comparison
ax = axes[1, 0]
names = [r.model for r in results]
bics = [r.bic for r in results]
colors = ["green" if n == best.model else "gray" for n in names]
ax.barh(names, bics, color=colors)
ax.set_xlabel("BIC (lower = better)")
ax.set_title("Model Comparison")

# 5d: Choice accuracy
ax = axes[1, 1]
window = 20
rolling = np.convolve(sim["choices"] == 0,
                      np.ones(window) / window, mode="valid")
ax.plot(rolling, color="purple")
ax.axhline(1.0, ls="--", color="black", alpha=0.3)
ax.set_xlabel("Trial")
ax.set_ylabel("P(optimal)")
ax.set_title(f"Choice Accuracy ({window}-trial window)")
ax.set_ylim(0, 1.05)

plt.tight_layout()
out_path = os.path.join(OUTPUT_DIR, "01_simulate_and_fit.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"\nFigure saved: {out_path}")
plt.close()
