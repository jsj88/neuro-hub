"""
Example 5: Run the Neuro-Coscientist agent in offline mode.

Demonstrates the autonomous agent running a complete analysis pipeline
WITHOUT needing an LLM API key (offline/deterministic mode).

Usage:
    cd ~/neuro-hub
    python examples/05_agent_demo.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.coscientist import NeuroCoscientist

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "agent_demo")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Initialize agent (offline = no LLM needed) ────────────────
agent = NeuroCoscientist(
    provider="openai",
    model="gpt-4o",
    max_steps=15,
    verbose=True,
    output_dir=OUTPUT_DIR,
)

print("Available tools:")
for name, tool in agent.tools.items():
    print(f"  {name}")

print("\n" + "=" * 60)
print("Available models in SIMULATE_BEHAVIOR:")
print(f"  {agent.tools['SIMULATE_BEHAVIOR'].MODEL_REGISTRY}")
print("=" * 60)

# ── Run offline: simulate + fit + compare ──────────────────────
print("\n\n=== RUNNING OFFLINE PIPELINE ===\n")
result = agent.run_offline(
    "Simulate behavioral data with RW model, then fit and compare models"
)
print("\n--- Agent Output ---")
print(result)

# ── Direct tool usage examples ─────────────────────────────────
print("\n\n=== DIRECT TOOL EXAMPLES ===\n")

# 1. Simulate
print("--- Simulating Q-dual ---")
sim_result = agent.tools["SIMULATE_BEHAVIOR"]({
    "model": "q_dual",
    "params": {"alpha_pos": 0.5, "alpha_neg": 0.1, "beta": 5.0},
    "n_trials": 200,
    "n_subjects": 3,
    "reward_probs": [0.7, 0.3],
    "output_dir": OUTPUT_DIR,
})
print(sim_result)

# 2. Fit
print("\n--- Fitting RW to simulated data ---")
csv_path = os.path.join(OUTPUT_DIR, "sim_q_dual_3subj.csv")
if os.path.exists(csv_path):
    fit_result = agent.tools["FIT_MODEL"]({
        "model": "rw",
        "data_path": csv_path,
        "subject": 0,
    })
    print(fit_result)

# 3. Compare
print("\n--- Comparing models ---")
if os.path.exists(csv_path):
    compare_result = agent.tools["COMPARE_MODELS"]({
        "models": ["random", "wsls", "rw", "q_dual", "actor_critic", "ck", "rwck"],
        "data_path": csv_path,
        "subject": 0,
    })
    print(compare_result)

# 4. Parameter recovery
print("\n--- Parameter recovery (RW) ---")
recovery_result = agent.tools["PARAMETER_RECOVERY"]({
    "model": "rw",
    "true_params": {"alpha": 0.3, "beta": 5.0},
    "n_trials": 200,
    "n_simulations": 20,
})
print(recovery_result)

print("\n=== DONE ===")
print(f"Output saved to: {OUTPUT_DIR}")
