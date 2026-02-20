"""
NeuralPipeline skill — multi-step neural analysis pipeline.

Chain: SIMULATE_NEURAL (or load .fif) -> RUN_TEMPORAL_DECODING -> RUN_REWP -> PLOT
"""

import os
import sys
from typing import Any, Dict, List, Optional

import numpy as np

from .base import BaseSkill, SkillResult
from ..tools.simulate import SimulateNeural

# Add tmaze-analysis to path for classification imports
_TMAZE_ROOT = os.path.join(os.path.dirname(__file__), "..", "..", "tmaze-analysis")
if _TMAZE_ROOT not in sys.path:
    sys.path.insert(0, os.path.abspath(_TMAZE_ROOT))


class NeuralPipeline(BaseSkill):
    """
    Multi-step neural analysis pipeline.

    Steps:
    1. Simulate EEG data (or load from .fif file)
    2. Run time-resolved temporal decoding
    3. Run REWP-focused analysis
    4. Plot decoding time course
    """

    def __init__(self):
        super().__init__("NEURAL_PIPELINE")
        self._sim_tool = SimulateNeural()

    @property
    def description(self) -> str:
        return (
            "Run a complete neural analysis pipeline: simulate/load EEG -> decode -> REWP -> plot.\n"
            "Input JSON keys:\n"
            "  source: str — 'simulated' or 'file' (default 'simulated')\n"
            "  file_path: str — path to .fif file (if source='file')\n"
            "  n_epochs: int — number of epochs for simulation (default 200)\n"
            "  n_channels: int — number of channels (default 64)\n"
            "  rewp_amplitude: float — REWP signal strength in uV (default 3.0)\n"
            "  rpe_coupling: float — RPE-to-REWP scaling (default 0.0)\n"
            "  rpes: list — trial-level RPEs for REWP modulation\n"
            "  noise_std: float — Gaussian noise std (default 10.0)\n"
            "  classifier: str — 'svm' or 'lda' (default 'svm')\n"
            "  cv_folds: int — cross-validation folds (default 5)\n"
            "  seed: int — random seed (default 42)\n"
            "  output_dir: str — output directory (default './results')\n"
            "Returns: decoding results, REWP results, simulated_eeg dict, figures."
        )

    def run_direct(self, **kwargs) -> SkillResult:
        self._reset_steps()

        source = kwargs.get("source", "simulated")
        file_path = kwargs.get("file_path", None)
        n_epochs = kwargs.get("n_epochs", 200)
        n_channels = kwargs.get("n_channels", 64)
        rewp_amplitude = kwargs.get("rewp_amplitude", 3.0)
        rpe_coupling = kwargs.get("rpe_coupling", 0.0)
        rpes = kwargs.get("rpes", None)
        noise_std = kwargs.get("noise_std", 10.0)
        classifier = kwargs.get("classifier", "svm")
        cv_folds = kwargs.get("cv_folds", 5)
        seed = kwargs.get("seed", 42)
        output_dir = kwargs.get("output_dir", "./results")

        os.makedirs(output_dir, exist_ok=True)
        fig_dir = os.path.join(output_dir, "figures")
        os.makedirs(fig_dir, exist_ok=True)

        result = SkillResult(skill_name="NeuralPipeline")
        simulated_eeg = None

        # ── Step 1: Get EEG data ──
        if source == "simulated":
            sim_params = {
                "n_epochs": n_epochs,
                "n_channels": n_channels,
                "rewp_amplitude": rewp_amplitude,
                "rpe_coupling": rpe_coupling,
                "rpes": rpes,
                "noise_std": noise_std,
                "seed": seed,
            }

            step = self._run_step(
                "SIMULATE_NEURAL",
                f"Simulate {n_epochs} EEG epochs (REWP={rewp_amplitude}uV, coupling={rpe_coupling})",
                self._sim_tool,
                sim_params,
            )
            if step.status == "error":
                result.error = f"Neural simulation failed: {step.error}"
                result.steps = self._steps
                return result

            simulated_eeg = self._sim_tool._last_data

        elif source == "file":
            if not file_path:
                result.error = "file_path required when source='file'"
                result.steps = self._steps
                return result

            def _load_file():
                from io.loaders import TMazeEEGLoader
                loader = TMazeEEGLoader(
                    condition_mapping={"MazeReward": 1, "MazeNoReward": 0},
                    tmin=-0.2, tmax=0.8,
                )
                return loader.load(file_path)

            step = self._run_step("LOAD_EEG", f"Load EEG from {file_path}", _load_file)
            if step.status == "error":
                result.error = f"EEG loading failed: {step.error}"
                result.steps = self._steps
                return result
            eeg_data_obj = step.output
        else:
            result.error = f"Unknown source '{source}'"
            result.steps = self._steps
            return result

        # Build TMazeEEGData object for classification pipeline
        def _build_eeg():
            from core.containers import TMazeEEGData
            if simulated_eeg is not None:
                return TMazeEEGData(
                    data=simulated_eeg["data"],
                    times=simulated_eeg["times"],
                    labels=simulated_eeg["labels"],
                    condition_names=["NoReward", "Reward"],
                    channels=simulated_eeg["channels"],
                    sfreq=simulated_eeg["sfreq"],
                    subject_id="sim-001",
                )
            return eeg_data_obj

        build_step = self._run_step("BUILD_EEG_DATA", "Build TMazeEEGData container", _build_eeg)
        if build_step.status == "error":
            result.error = f"EEG data build failed: {build_step.error}"
            result.steps = self._steps
            return result
        eeg_data = build_step.output

        # ── Step 2: Temporal decoding ──
        decoding_result = None

        def _run_decoding():
            from classification.temporal import temporal_decoding, find_significant_times
            dec = temporal_decoding(
                eeg_data,
                classifier_type=classifier,
                scoring="roc_auc",
                cv=cv_folds,
                n_jobs=-1,
                verbose=False,
            )
            sig_mask, clusters = find_significant_times(dec, chance_level=0.5, alpha=0.05)
            return {
                "decoding": dec,
                "sig_mask": sig_mask,
                "clusters": clusters,
            }

        dec_step = self._run_step(
            "RUN_TEMPORAL_DECODING",
            f"Time-resolved {classifier.upper()} decoding ({cv_folds}-fold CV)",
            _run_decoding,
        )
        if dec_step.status == "success":
            decoding_result = dec_step.output

        # ── Step 3: REWP-focused analysis ──
        rewp_result = None

        def _run_rewp():
            from classification.temporal import rewp_temporal_analysis
            return rewp_temporal_analysis(
                eeg_data,
                fcz_only=True,
                classifier_type=classifier,
                cv=cv_folds,
                n_jobs=-1,
            )

        rewp_step = self._run_step(
            "RUN_REWP",
            f"REWP temporal analysis (FCz, {classifier.upper()})",
            _run_rewp,
        )
        if rewp_step.status == "success":
            rewp_result = rewp_step.output

        # ── Step 4: Plot temporal decoding time course ──
        if decoding_result is not None:
            dec = decoding_result["decoding"]

            def _plot_decoding():
                import matplotlib
                matplotlib.use("Agg")
                import matplotlib.pyplot as plt

                times_ms = eeg_data.times * 1000
                scores = dec.scores_mean if hasattr(dec, "scores_mean") else dec.scores
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.plot(times_ms, scores, "b-", linewidth=2)
                if hasattr(dec, "scores_std"):
                    ax.fill_between(
                        times_ms,
                        scores - dec.scores_std,
                        scores + dec.scores_std,
                        alpha=0.2,
                    )
                ax.axhline(0.5, color="k", linestyle="--", alpha=0.5, label="Chance")
                ax.axvspan(240, 340, alpha=0.15, color="orange", label="REWP window")
                ax.set_xlabel("Time (ms)")
                ax.set_ylabel("AUC")
                ax.set_title(f"Temporal Decoding — Peak: {dec.peak_score:.3f} at {dec.peak_time*1000:.0f}ms")
                ax.legend()
                plt.tight_layout()

                path = os.path.join(fig_dir, "temporal_decoding.png")
                fig.savefig(path, dpi=300, bbox_inches="tight")
                plt.close(fig)
                return path

            plot_step = self._run_step("PLOT", "Plot temporal decoding time course", _plot_decoding)
            if plot_step.status == "success":
                result.figures.append(plot_step.output)

        # ── Build result ──
        result.success = True
        result.results = {
            "simulated_eeg": simulated_eeg,
            "source": source,
        }

        if decoding_result is not None:
            dec = decoding_result["decoding"]
            clusters = decoding_result["clusters"]
            result.results["decoding"] = {
                "peak_score": float(dec.peak_score),
                "peak_time_ms": float(dec.peak_time * 1000),
                "onset_time_ms": float(dec.onset_time * 1000) if dec.onset_time else None,
                "significant_clusters": [[c[0], c[1]] for c in clusters] if clusters else [],
            }

        if rewp_result is not None:
            in_rewp = (
                rewp_result.peak_time is not None
                and 0.240 <= rewp_result.peak_time <= 0.340
            )
            result.results["rewp"] = {
                "peak_score": float(rewp_result.peak_score),
                "peak_time_ms": float(rewp_result.peak_time * 1000) if rewp_result.peak_time else None,
                "in_rewp_window": in_rewp,
                "significant_clusters": (
                    [[c[0], c[1]] for c in rewp_result.significant_clusters]
                    if rewp_result.significant_clusters else []
                ),
            }

        # Summary
        summary_parts = [f"Source: {source}"]
        if simulated_eeg:
            summary_parts.append(
                f"EEG: {simulated_eeg['n_epochs']} epochs x {simulated_eeg['n_channels']} channels"
            )
        if decoding_result:
            dec = decoding_result["decoding"]
            summary_parts.append(
                f"Decoding peak: {dec.peak_score:.3f} at {dec.peak_time*1000:.0f}ms"
            )
        if rewp_result and rewp_result.peak_time:
            summary_parts.append(
                f"REWP peak: {rewp_result.peak_score:.3f} at {rewp_result.peak_time*1000:.0f}ms"
            )

        result.summary = "\n".join(summary_parts)
        result.steps = self._steps

        return result
