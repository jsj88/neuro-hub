"""Quick smoke test for the Neuro-Coscientist agent tools."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

print("=" * 60)
print("Neuro-Coscientist — Smoke Test")
print("=" * 60)

# 1. Test behavioral simulation
print("\n[1] Testing SimulateBehavior (RW)...")
from agent.tools.simulate import simulate_rw
res = simulate_rw(200, 2, np.array([0.7, 0.3]), alpha=0.3, beta=5.0)
print(f"    Choices shape: {res['choices'].shape}")
print(f"    Mean reward: {res['outcomes'].mean():.3f}")
print(f"    RPE range: [{res['rpes'].min():.3f}, {res['rpes'].max():.3f}]")
print("    OK")

# 2. Test ORL simulation
print("\n[2] Testing SimulateBehavior (ORL)...")
from agent.tools.simulate import simulate_orl, _default_igt_payoffs
payoffs = _default_igt_payoffs(100)
res_orl = simulate_orl(100, payoffs, A_rew=0.3, A_pun=0.1, K=1.0, beta_f=2.0, beta_p=1.0)
print(f"    Choices shape: {res_orl['choices'].shape}")
print(f"    Mean outcome: {res_orl['outcomes'].mean():.1f}")
print("    OK")

# 3. Test model fitting
print("\n[3] Testing FitModel (RW)...")
from agent.tools.model_fitting import fit_model
fit_res = fit_model("rw", res["choices"], res["outcomes"], n_options=2, method="de")
print(f"    Recovered: alpha={fit_res.params['alpha']:.3f}, beta={fit_res.params['beta']:.3f}")
print(f"    NLL={fit_res.nll:.2f}, BIC={fit_res.bic:.2f}")
print("    OK")

# 4. Test model comparison
print("\n[4] Testing CompareModels...")
from agent.tools.model_fitting import fit_model as fm
results = []
for m in ["rw", "ck", "rwck"]:
    r = fm(m, res["choices"], res["outcomes"], n_options=2)
    results.append(r)
    print(f"    {m}: BIC={r.bic:.2f}")
best = min(results, key=lambda r: r.bic)
print(f"    Winner: {best.model}")
print("    OK")

# 5. Test SimulateNeural tool
print("\n[5] Testing SimulateNeural...")
from agent.tools.simulate import SimulateNeural
sim_neural = SimulateNeural()
result_str = sim_neural({"n_epochs": 100, "n_channels": 64, "rewp_amplitude": 3.0})
print(f"    {result_str[:100]}...")
print("    OK")

# 6. Test tool registry
print("\n[6] Testing tool registry...")
from agent.tools.base import Stop
from agent.tools.simulate import SimulateBehavior
from agent.tools.model_fitting import FitModel, ParameterRecovery, CompareModels
from agent.tools.correlate import CorrelateNeuroModel
from agent.tools.plot import PlotAndSave

tools = [SimulateBehavior(), SimulateNeural(), FitModel(), ParameterRecovery(),
         CompareModels(), CorrelateNeuroModel(), PlotAndSave(), Stop()]
for t in tools:
    print(f"    {t.command_name}: registered")
print("    OK")

# 7. Test system prompt
print("\n[7] Testing system prompt build...")
from agent.prompts.system_prompt import build_system_prompt
tool_dict = {t.command_name: t for t in tools}
prompt = build_system_prompt(tool_dict)
print(f"    System prompt length: {len(prompt)} chars")
print(f"    Tools listed: {len(tool_dict)}")
print("    OK")

print("\n" + "=" * 60)
print("All tests passed!")
print("=" * 60)
