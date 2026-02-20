"""
FullPipeline skill — end-to-end simulated neuroscience pipeline.

Chain: BehavioralPipeline -> NeuralPipeline -> NeuroModelFusion -> summary report
"""

import os
from typing import Any, Dict

from .base import BaseSkill, SkillResult
from .behavioral import BehavioralPipeline
from .neural import NeuralPipeline
from .fusion import NeuroModelFusion


class FullPipeline(BaseSkill):
    """
    End-to-end pipeline: behavior + neural + fusion.

    Steps:
    1. BehavioralPipeline: simulate -> fit -> compare -> extract RPEs
    2. NeuralPipeline: simulate EEG (coupled to RPEs) -> decode -> REWP
    3. NeuroModelFusion: correlate RPEs with REWP amplitude -> scatter plot
    """

    def __init__(self):
        super().__init__("FULL_PIPELINE")
        self._behavioral = BehavioralPipeline()
        self._neural = NeuralPipeline()
        self._fusion = NeuroModelFusion()

    @property
    def description(self) -> str:
        return (
            "Run the full simulated pipeline: behavioral modeling -> neural analysis -> fusion.\n"
            "Input JSON keys:\n"
            "  generating_model: str — behavioral model (default 'rw')\n"
            "  generating_params: dict — model parameters\n"
            "  models_to_compare: list — candidate models (default ['rw', 'ck', 'rwck'])\n"
            "  n_trials: int — trials (default 200)\n"
            "  n_subjects: int — subjects (default 1)\n"
            "  n_options: int — options (default 2)\n"
            "  reward_probs: list — reward probabilities (default [0.7, 0.3])\n"
            "  rewp_amplitude: float — REWP signal strength (default 3.0)\n"
            "  rpe_coupling: float — RPE-to-REWP coupling (default 2.0)\n"
            "  noise_std: float — EEG noise (default 10.0)\n"
            "  classifier: str — 'svm' or 'lda' (default 'svm')\n"
            "  correlation_method: str — 'pearson', 'spearman', 'regression' (default 'pearson')\n"
            "  seed: int — random seed (default 42)\n"
            "  output_dir: str — output directory (default './results')\n"
            "Returns: combined results from all 3 stages, all figures, comprehensive summary."
        )

    def run_direct(self, **kwargs) -> SkillResult:
        self._reset_steps()

        # Shared config
        output_dir = kwargs.get("output_dir", "./results")
        seed = kwargs.get("seed", 42)
        n_trials = kwargs.get("n_trials", 200)
        rpe_coupling = kwargs.get("rpe_coupling", 2.0)

        os.makedirs(output_dir, exist_ok=True)

        result = SkillResult(skill_name="FullPipeline")

        # ══════════════════════════════════════════════════════════════
        # Stage 1: Behavioral Pipeline
        # ══════════════════════════════════════════════════════════════
        beh_step = self._run_step(
            "BEHAVIORAL_PIPELINE",
            "Stage 1: Behavioral modeling pipeline",
            self._behavioral.run_direct,
            generating_model=kwargs.get("generating_model", "rw"),
            generating_params=kwargs.get("generating_params", {"alpha": 0.3, "beta": 5.0}),
            models_to_compare=kwargs.get("models_to_compare", ["rw", "ck", "rwck"]),
            n_trials=n_trials,
            n_subjects=kwargs.get("n_subjects", 1),
            n_options=kwargs.get("n_options", 2),
            reward_probs=kwargs.get("reward_probs", [0.7, 0.3]),
            seed=seed,
            run_parameter_recovery=kwargs.get("run_parameter_recovery", False),
            output_dir=output_dir,
        )

        if beh_step.status == "error":
            result.error = f"Behavioral pipeline failed: {beh_step.error}"
            result.steps = self._steps
            return result

        beh_result: SkillResult = beh_step.output
        if not beh_result.success:
            result.error = f"Behavioral pipeline failed: {beh_result.error}"
            result.steps = self._steps
            return result

        rpes = beh_result.results.get("rpes")
        result.figures.extend(beh_result.figures)
        result.csv_paths.extend(beh_result.csv_paths)

        # ══════════════════════════════════════════════════════════════
        # Stage 2: Neural Pipeline
        # ══════════════════════════════════════════════════════════════
        neural_step = self._run_step(
            "NEURAL_PIPELINE",
            "Stage 2: Neural analysis pipeline",
            self._neural.run_direct,
            source="simulated",
            n_epochs=n_trials,
            n_channels=kwargs.get("n_channels", 64),
            rewp_amplitude=kwargs.get("rewp_amplitude", 3.0),
            rpe_coupling=rpe_coupling,
            rpes=rpes,
            noise_std=kwargs.get("noise_std", 10.0),
            classifier=kwargs.get("classifier", "svm"),
            cv_folds=kwargs.get("cv_folds", 5),
            seed=seed,
            output_dir=output_dir,
        )

        if neural_step.status == "error":
            result.error = f"Neural pipeline failed: {neural_step.error}"
            result.steps = self._steps
            return result

        neural_result: SkillResult = neural_step.output
        if not neural_result.success:
            result.error = f"Neural pipeline failed: {neural_result.error}"
            result.steps = self._steps
            return result

        simulated_eeg = neural_result.results.get("simulated_eeg")
        result.figures.extend(neural_result.figures)

        # ══════════════════════════════════════════════════════════════
        # Stage 3: Neuro-Model Fusion
        # ══════════════════════════════════════════════════════════════
        fusion_step = self._run_step(
            "NEURO_MODEL_FUSION",
            "Stage 3: Neuro-model fusion (RPE-REWP correlation)",
            self._fusion.run_direct,
            rpes=rpes,
            simulated_eeg=simulated_eeg,
            method=kwargs.get("correlation_method", "pearson"),
            channels=kwargs.get("channels", ["FCz"]),
            time_window=kwargs.get("time_window", [0.240, 0.340]),
            output_dir=output_dir,
        )

        if fusion_step.status == "error":
            result.error = f"Fusion pipeline failed: {fusion_step.error}"
            result.steps = self._steps
            return result

        fusion_result: SkillResult = fusion_step.output
        if not fusion_result.success:
            result.error = f"Fusion pipeline failed: {fusion_result.error}"
            result.steps = self._steps
            return result

        result.figures.extend(fusion_result.figures)

        # ══════════════════════════════════════════════════════════════
        # Compile final result
        # ══════════════════════════════════════════════════════════════
        result.success = True
        result.results = {
            "behavioral": beh_result.results,
            "neural": neural_result.results,
            "fusion": fusion_result.results,
        }

        # Build comprehensive summary
        best_model = beh_result.results.get("best_model", "?")
        best_bic = beh_result.results.get("best_bic", 0)
        corr = fusion_result.results.get("correlation", {})

        summary_lines = [
            "=== Full Pipeline Summary ===",
            "",
            f"Behavioral: Best model = {best_model} (BIC = {best_bic:.2f})",
            f"  Generating model: {kwargs.get('generating_model', 'rw')}",
            f"  N trials: {n_trials}",
        ]

        if "decoding" in neural_result.results:
            dec = neural_result.results["decoding"]
            summary_lines.append(
                f"Neural: Peak decoding = {dec['peak_score']:.3f} at {dec['peak_time_ms']:.0f}ms"
            )
        if "rewp" in neural_result.results:
            rewp = neural_result.results["rewp"]
            summary_lines.append(
                f"  REWP in window: {'YES' if rewp.get('in_rewp_window') else 'NO'}"
            )

        if corr:
            sig = "YES" if fusion_result.results.get("significant") else "NO"
            summary_lines.append(
                f"Fusion: r = {corr.get('r', 0):.4f}, p = {corr.get('p', 1):.2e} (significant: {sig})"
            )

        summary_lines.append(f"\nFigures: {len(result.figures)}")
        summary_lines.append(f"CSVs: {len(result.csv_paths)}")

        result.summary = "\n".join(summary_lines)
        result.steps = self._steps

        return result
