"""
BehavioralPipeline skill — multi-step behavioral modeling pipeline.

Chain: SIMULATE_BEHAVIOR -> FIT_MODEL (per model) -> COMPARE_MODELS -> PLOT
Optional: PARAMETER_RECOVERY if run_parameter_recovery=True
"""

import os
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .base import BaseSkill, SkillResult
from ..tools.simulate import simulate_rw, simulate_ck, simulate_rwck
from ..tools.model_fitting import fit_model, extract_trial_variables, MODEL_SPECS


# Map model names to simulator functions (bandit models only)
_BANDIT_SIMULATORS = {
    "rw": simulate_rw,
    "ck": simulate_ck,
    "rwck": simulate_rwck,
}

# Default parameters per model for simulation
_DEFAULT_PARAMS = {
    "rw": {"alpha": 0.3, "beta": 5.0},
    "ck": {"alpha_c": 0.3, "beta_c": 3.0},
    "rwck": {"alpha": 0.3, "beta": 5.0, "alpha_c": 0.2, "beta_c": 2.0},
}


class BehavioralPipeline(BaseSkill):
    """
    Multi-step behavioral modeling pipeline.

    Steps:
    1. Simulate behavioral data from a generating model
    2. Fit each candidate model via MLE
    3. Compare models via BIC
    4. Plot model comparison (bar chart)
    5. (Optional) Run parameter recovery on the generating model
    """

    def __init__(self):
        super().__init__("BEHAVIORAL_PIPELINE")

    @property
    def description(self) -> str:
        return (
            "Run a complete behavioral modeling pipeline: simulate -> fit -> compare -> plot.\n"
            "Input JSON keys:\n"
            "  generating_model: str — model to generate data ('rw', 'ck', 'rwck'; default 'rw')\n"
            "  generating_params: dict — params for generating model (defaults per model)\n"
            "  models_to_compare: list — models to fit and compare (default ['rw', 'ck', 'rwck'])\n"
            "  n_trials: int — trials per subject (default 200)\n"
            "  n_subjects: int — number of subjects (default 1)\n"
            "  n_options: int — number of options (default 2)\n"
            "  reward_probs: list — per-option reward probability (default [0.7, 0.3])\n"
            "  seed: int — random seed (default 42)\n"
            "  run_parameter_recovery: bool — run recovery on generating model (default False)\n"
            "  n_recovery_sims: int — number of recovery simulations (default 50)\n"
            "  output_dir: str — output directory (default './results')\n"
            "Returns: best model, fitted params, RPEs, CSVs, and figures."
        )

    def run_direct(self, **kwargs) -> SkillResult:
        self._reset_steps()

        # Parse config
        gen_model = kwargs.get("generating_model", "rw")
        gen_params = kwargs.get("generating_params", _DEFAULT_PARAMS.get(gen_model, {}))
        models_to_compare = kwargs.get("models_to_compare", ["rw", "ck", "rwck"])
        n_trials = kwargs.get("n_trials", 200)
        n_subjects = kwargs.get("n_subjects", 1)
        n_options = kwargs.get("n_options", 2)
        reward_probs = np.array(kwargs.get("reward_probs", [0.7, 0.3]))
        seed = kwargs.get("seed", 42)
        run_recovery = kwargs.get("run_parameter_recovery", False)
        n_recovery_sims = kwargs.get("n_recovery_sims", 50)
        output_dir = kwargs.get("output_dir", "./results")

        os.makedirs(output_dir, exist_ok=True)
        fig_dir = os.path.join(output_dir, "figures")
        os.makedirs(fig_dir, exist_ok=True)

        result = SkillResult(skill_name="BehavioralPipeline")

        # ── Step 1: Simulate behavioral data ──
        sim_fn = _BANDIT_SIMULATORS.get(gen_model)
        if sim_fn is None:
            result.error = f"Unknown generating model '{gen_model}'. Supported: {list(_BANDIT_SIMULATORS.keys())}"
            result.steps = self._steps
            return result

        all_dfs = []
        all_sim_data = []

        for subj in range(n_subjects):
            step = self._run_step(
                "SIMULATE_BEHAVIOR",
                f"Simulate {gen_model} subject {subj}",
                sim_fn,
                n_trials, n_options, reward_probs, **gen_params, seed=seed + subj,
            )
            if step.status == "error":
                result.error = f"Simulation failed: {step.error}"
                result.steps = self._steps
                return result

            sim_data = step.output
            all_sim_data.append(sim_data)

            df = pd.DataFrame({
                "subject": subj,
                "trial": np.arange(n_trials),
                "choice": sim_data["choices"],
                "outcome": sim_data["outcomes"],
            })
            if "rpes" in sim_data:
                df["rpe"] = sim_data["rpes"]
            if "q_values" in sim_data:
                for k in range(sim_data["q_values"].shape[1]):
                    df[f"Q_{k}"] = sim_data["q_values"][:, k]
            all_dfs.append(df)

        combined_df = pd.concat(all_dfs, ignore_index=True)
        csv_path = os.path.join(output_dir, f"sim_{gen_model}_{n_subjects}subj.csv")
        combined_df.to_csv(csv_path, index=False)
        result.csv_paths.append(csv_path)

        # ── Step 2: Fit each candidate model ──
        fit_results = {}
        for model_name in models_to_compare:
            if model_name not in MODEL_SPECS:
                continue

            # Fit on subject 0 (or combined)
            choices = combined_df[combined_df["subject"] == 0]["choice"].values.astype(int)
            outcomes = combined_df[combined_df["subject"] == 0]["outcome"].values.astype(float)

            step = self._run_step(
                "FIT_MODEL",
                f"Fit {model_name} model",
                fit_model,
                model_name, choices, outcomes, n_options,
            )
            if step.status == "success":
                fit_results[model_name] = step.output

        if not fit_results:
            result.error = "All model fits failed"
            result.steps = self._steps
            return result

        # ── Step 3: Compare models by BIC ──
        ranked = sorted(fit_results.values(), key=lambda r: r.bic)
        best = ranked[0]
        best_bic = best.bic

        comparison_lines = []
        for i, r in enumerate(ranked):
            comparison_lines.append(
                f"  {i+1}. {r.model}: BIC={r.bic:.2f} (DBIC={r.bic - best_bic:+.2f}), "
                f"params={r.params}"
            )
        comparison_str = "\n".join(comparison_lines)

        step = SkillResult  # just record the comparison step manually
        comp_step = self._run_step(
            "COMPARE_MODELS",
            f"Compare {len(fit_results)} models by BIC",
            lambda: ranked,
        )

        # ── Step 4: Plot model comparison ──
        def _plot_comparison():
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            models = [r.model for r in ranked]
            bics = [r.bic for r in ranked]
            colors = ["#2ecc71" if i == 0 else "#95a5a6" for i in range(len(models))]

            fig, ax = plt.subplots(figsize=(8, 5))
            ax.bar(models, bics, color=colors, edgecolor="black", linewidth=0.5)
            ax.set_ylabel("BIC")
            ax.set_title(f"Model Comparison — Best: {best.model}")
            plt.tight_layout()

            path = os.path.join(fig_dir, "model_comparison.png")
            fig.savefig(path, dpi=300, bbox_inches="tight")
            plt.close(fig)
            return path

        plot_step = self._run_step("PLOT", "Plot model comparison bar chart", _plot_comparison)
        if plot_step.status == "success":
            result.figures.append(plot_step.output)

        # ── Step 5: Extract trial variables from best model ──
        choices_s0 = combined_df[combined_df["subject"] == 0]["choice"].values.astype(int)
        outcomes_s0 = combined_df[combined_df["subject"] == 0]["outcome"].values.astype(float)

        trial_vars_step = self._run_step(
            "EXTRACT_TRIAL_VARIABLES",
            f"Extract RPEs from best model ({best.model})",
            extract_trial_variables,
            best.model, best.params, choices_s0, outcomes_s0, n_options,
        )
        rpes = None
        if trial_vars_step.status == "success" and "rpes" in trial_vars_step.output:
            rpes = trial_vars_step.output["rpes"]

        # ── Step 6 (optional): Parameter recovery ──
        recovery_results = None
        if run_recovery and gen_model in _BANDIT_SIMULATORS:
            def _run_recovery():
                recovered = {name: [] for name in MODEL_SPECS[gen_model]["param_names"]}
                for i in range(n_recovery_sims):
                    sim = _BANDIT_SIMULATORS[gen_model](
                        n_trials, n_options, reward_probs, **gen_params, seed=1000 + i,
                    )
                    fr = fit_model(gen_model, sim["choices"], sim["outcomes"], n_options, seed=i)
                    for name in recovered:
                        recovered[name].append(fr.params[name])
                return recovered

            rec_step = self._run_step(
                "PARAMETER_RECOVERY",
                f"Parameter recovery for {gen_model} ({n_recovery_sims} sims)",
                _run_recovery,
            )
            if rec_step.status == "success":
                recovery_results = rec_step.output

        # ── Build result ──
        result.success = True
        result.results = {
            "best_model": best.model,
            "best_bic": best.bic,
            "best_params": best.params,
            "all_fits": {r.model: {"params": r.params, "bic": r.bic, "nll": r.nll} for r in ranked},
            "rpes": rpes.tolist() if rpes is not None else None,
            "choices": choices_s0.tolist(),
            "outcomes": outcomes_s0.tolist(),
            "n_trials": n_trials,
            "n_options": n_options,
            "generating_model": gen_model,
            "generating_params": gen_params,
        }
        if recovery_results:
            result.results["parameter_recovery"] = {
                name: {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
                for name, vals in recovery_results.items()
            }

        result.summary = (
            f"Generated {n_trials} trials from {gen_model} ({gen_params}).\n"
            f"Best model: {best.model} (BIC={best.bic:.2f})\n"
            f"Ranking:\n{comparison_str}"
        )
        result.steps = self._steps

        return result
