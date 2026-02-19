"""
Example 2: Parameter recovery analysis.

Validates that a model's parameters can be reliably recovered from
simulated data. Essential step before applying to real data.

Usage:
    cd ~/neuro-hub
    python examples/02_parameter_recovery.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from agent.tools.simulate import simulate_rw, simulate_q_dual
from agent.tools.model_fitting import fit_model

# ── Configuration ──────────────────────────────────────────────
REWARD_PROBS = np.array([0.7, 0.3])
N_TRIALS = 200
N_SIMS = 50
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "examples")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def run_recovery(model_name, sim_func, param_ranges, n_sims=N_SIMS):
    """
    Run parameter recovery: sample params, simulate, fit, compare.

    Parameters
    ----------
    model_name : str
    sim_func : callable(seed, **params) -> dict with 'choices', 'outcomes'
    param_ranges : dict of param_name -> (low, high)
    """
    rng = np.random.default_rng(42)
    param_names = list(param_ranges.keys())

    true_vals = {p: [] for p in param_names}
    rec_vals = {p: [] for p in param_names}

    for i in range(n_sims):
        # Sample random true parameters
        params = {}
        for p, (lo, hi) in param_ranges.items():
            params[p] = rng.uniform(lo, hi)
            true_vals[p].append(params[p])

        # Simulate
        sim = sim_func(seed=i, **params)

        # Fit
        result = fit_model(model_name, sim["choices"], sim["outcomes"],
                           n_options=2, method="multistart", n_starts=5, seed=i)

        for p in param_names:
            rec_vals[p].append(result.params[p])

        if (i + 1) % 10 == 0:
            print(f"  {model_name}: {i + 1}/{n_sims} simulations complete")

    return true_vals, rec_vals


# ── Recovery 1: RW model ──────────────────────────────────────
print("=== Parameter Recovery: Rescorla-Wagner ===")
true_rw, rec_rw = run_recovery(
    "rw",
    lambda seed, alpha, beta: simulate_rw(
        N_TRIALS, 2, REWARD_PROBS, alpha=alpha, beta=beta, seed=seed),
    {"alpha": (0.05, 0.95), "beta": (0.5, 15.0)},
)

# ── Recovery 2: Q-dual model ─────────────────────────────────
print("\n=== Parameter Recovery: Q-Learning Dual LR ===")
true_qd, rec_qd = run_recovery(
    "q_dual",
    lambda seed, alpha_pos, alpha_neg, beta: simulate_q_dual(
        N_TRIALS, 2, REWARD_PROBS,
        alpha_pos=alpha_pos, alpha_neg=alpha_neg, beta=beta, seed=seed),
    {"alpha_pos": (0.05, 0.95), "alpha_neg": (0.05, 0.95), "beta": (0.5, 15.0)},
)

# ── Plot results ──────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# RW recovery
for i, param in enumerate(["alpha", "beta"]):
    ax = axes[0, i]
    t = true_rw[param]
    r = rec_rw[param]
    ax.scatter(t, r, alpha=0.5, edgecolors="black", linewidth=0.5)
    lims = [min(min(t), min(r)), max(max(t), max(r))]
    ax.plot(lims, lims, "k--", alpha=0.3)
    corr, pval = pearsonr(t, r)
    ax.set_xlabel(f"True {param}")
    ax.set_ylabel(f"Recovered {param}")
    ax.set_title(f"RW: {param} (r={corr:.3f}, p={pval:.1e})")
axes[0, 2].axis("off")
axes[0, 2].text(0.5, 0.5, "RW Model\n2 parameters\nalpha, beta",
                ha="center", va="center", fontsize=14)

# Q-dual recovery
for i, param in enumerate(["alpha_pos", "alpha_neg", "beta"]):
    ax = axes[1, i]
    t = true_qd[param]
    r = rec_qd[param]
    ax.scatter(t, r, alpha=0.5, edgecolors="black", linewidth=0.5, color="coral")
    lims = [min(min(t), min(r)), max(max(t), max(r))]
    ax.plot(lims, lims, "k--", alpha=0.3)
    corr, pval = pearsonr(t, r)
    ax.set_xlabel(f"True {param}")
    ax.set_ylabel(f"Recovered {param}")
    ax.set_title(f"Q-dual: {param} (r={corr:.3f}, p={pval:.1e})")

plt.suptitle("Parameter Recovery Analysis", fontsize=16, y=1.02)
plt.tight_layout()
out_path = os.path.join(OUTPUT_DIR, "02_parameter_recovery.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"\nFigure saved: {out_path}")
plt.close()

# ── Print summary ─────────────────────────────────────────────
print("\n=== Summary ===")
for name, tv, rv in [("RW", true_rw, rec_rw), ("Q-dual", true_qd, rec_qd)]:
    print(f"\n{name}:")
    for p in tv:
        r, pval = pearsonr(tv[p], rv[p])
        bias = np.mean(np.array(rv[p]) - np.array(tv[p]))
        print(f"  {p:<12} r={r:.3f}  p={pval:.1e}  bias={bias:+.3f}")
