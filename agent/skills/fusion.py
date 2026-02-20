"""
NeuroModelFusion skill — correlate behavioral model variables with neural signals.

Chain: extract RPEs -> CORRELATE (pearson/spearman/regression) -> PLOT (scatter)
"""

import os
from typing import Any, Dict, List, Optional

import numpy as np
from scipy import stats

from .base import BaseSkill, SkillResult


class NeuroModelFusion(BaseSkill):
    """
    Multi-step neuro-model fusion pipeline.

    Steps:
    1. Extract REWP amplitude from simulated EEG in the REWP window
    2. Correlate model-derived RPEs with REWP amplitude
    3. Plot scatter with regression line
    """

    def __init__(self):
        super().__init__("NEURO_MODEL_FUSION")

    @property
    def description(self) -> str:
        return (
            "Fuse behavioral model outputs with neural signals: extract RPEs -> correlate -> plot.\n"
            "Input JSON keys:\n"
            "  rpes: list — trial-level RPEs from behavioral model fitting\n"
            "  choices: list — trial-level choices\n"
            "  outcomes: list — trial-level outcomes\n"
            "  simulated_eeg: dict — EEG data dict from NeuralPipeline (with 'data', 'times', etc.)\n"
            "  method: str — 'pearson', 'spearman', or 'regression' (default 'pearson')\n"
            "  channels: list — channels for REWP extraction (default ['FCz'])\n"
            "  time_window: list — [tmin, tmax] in seconds (default [0.24, 0.34])\n"
            "  output_dir: str — output directory (default './results')\n"
            "Returns: correlation stats (r, p, r-squared), regression coefficients, scatter figure."
        )

    def run_direct(self, **kwargs) -> SkillResult:
        self._reset_steps()

        rpes = kwargs.get("rpes")
        simulated_eeg = kwargs.get("simulated_eeg")
        method = kwargs.get("method", "pearson")
        channels = kwargs.get("channels", ["FCz"])
        time_window = kwargs.get("time_window", [0.240, 0.340])
        output_dir = kwargs.get("output_dir", "./results")

        os.makedirs(output_dir, exist_ok=True)
        fig_dir = os.path.join(output_dir, "figures")
        os.makedirs(fig_dir, exist_ok=True)

        result = SkillResult(skill_name="NeuroModelFusion")

        if rpes is None:
            result.error = "No RPEs provided. Run BehavioralPipeline first."
            result.steps = self._steps
            return result

        if simulated_eeg is None:
            result.error = "No simulated_eeg provided. Run NeuralPipeline first."
            result.steps = self._steps
            return result

        model_data = np.array(rpes)

        # ── Step 1: Extract REWP amplitude from EEG ──
        def _extract_rewp():
            data = simulated_eeg["data"]       # (n_epochs, n_channels, n_times)
            times = simulated_eeg["times"]
            ch_names = simulated_eeg["channels"]

            # Find channel indices
            ch_idx = [i for i, ch in enumerate(ch_names) if ch in channels]
            if not ch_idx:
                # Fallback to first 5 (frontocentral)
                ch_idx = list(range(min(5, data.shape[1])))

            # Time mask for REWP window
            t_mask = (times >= time_window[0]) & (times <= time_window[1])

            # Mean amplitude across channels and time window
            rewp_amp = data[:, ch_idx, :][:, :, t_mask].mean(axis=(1, 2))
            return rewp_amp

        extract_step = self._run_step(
            "EXTRACT_REWP",
            f"Extract REWP amplitude ({channels}, [{time_window[0]:.3f}-{time_window[1]:.3f}]s)",
            _extract_rewp,
        )
        if extract_step.status == "error":
            result.error = f"REWP extraction failed: {extract_step.error}"
            result.steps = self._steps
            return result

        neural_data = extract_step.output

        # Align lengths
        n = min(len(model_data), len(neural_data))
        model_data = model_data[:n]
        neural_data = neural_data[:n]

        # Remove NaN/Inf
        valid = np.isfinite(model_data) & np.isfinite(neural_data)
        model_data = model_data[valid]
        neural_data = neural_data[valid]

        if len(model_data) < 5:
            result.error = "Fewer than 5 valid data points for correlation."
            result.steps = self._steps
            return result

        # ── Step 2: Correlate ──
        def _correlate():
            if method == "pearson":
                r, p = stats.pearsonr(model_data, neural_data)
                return {"method": "Pearson", "r": float(r), "p": float(p), "r_squared": float(r**2)}
            elif method == "spearman":
                r, p = stats.spearmanr(model_data, neural_data)
                return {"method": "Spearman", "r": float(r), "p": float(p), "r_squared": float(r**2)}
            elif method == "regression":
                slope, intercept, r, p, se = stats.linregress(model_data, neural_data)
                return {
                    "method": "Regression",
                    "slope": float(slope),
                    "intercept": float(intercept),
                    "r": float(r),
                    "p": float(p),
                    "r_squared": float(r**2),
                    "se": float(se),
                }
            else:
                raise ValueError(f"Unknown method '{method}'")

        corr_step = self._run_step(
            "CORRELATE",
            f"{method.capitalize()} correlation: RPE vs REWP amplitude (n={len(model_data)})",
            _correlate,
        )
        if corr_step.status == "error":
            result.error = f"Correlation failed: {corr_step.error}"
            result.steps = self._steps
            return result

        corr_stats = corr_step.output

        # ── Step 3: Plot scatter ──
        def _plot_scatter():
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(8, 6))
            ax.scatter(model_data, neural_data, alpha=0.5, edgecolors="black", linewidths=0.3)

            # Regression line
            z = np.polyfit(model_data, neural_data, 1)
            p_line = np.poly1d(z)
            x_sorted = np.sort(model_data)
            ax.plot(x_sorted, p_line(x_sorted), "r-", linewidth=2)

            ax.set_xlabel("RPE (model)")
            ax.set_ylabel("REWP amplitude (uV)")

            r_val = corr_stats["r"]
            p_val = corr_stats["p"]
            sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "n.s."
            ax.set_title(f"RPE-REWP Correlation: r={r_val:.3f}, p={p_val:.2e} {sig}")
            plt.tight_layout()

            path = os.path.join(fig_dir, "rpe_rewp_correlation.png")
            fig.savefig(path, dpi=300, bbox_inches="tight")
            plt.close(fig)
            return path

        plot_step = self._run_step("PLOT", "Plot RPE-REWP scatter with regression", _plot_scatter)
        if plot_step.status == "success":
            result.figures.append(plot_step.output)

        # ── Build result ──
        result.success = True
        result.results = {
            "correlation": corr_stats,
            "n_datapoints": int(len(model_data)),
            "significant": corr_stats["p"] < 0.05,
        }

        sig_str = "YES" if corr_stats["p"] < 0.05 else "NO"
        result.summary = (
            f"{corr_stats['method']} correlation: RPE vs REWP amplitude\n"
            f"  r = {corr_stats['r']:.4f}, p = {corr_stats['p']:.2e}, r^2 = {corr_stats['r_squared']:.4f}\n"
            f"  n = {len(model_data)}\n"
            f"  Significant (p < .05): {sig_str}"
        )
        if method == "regression" and "slope" in corr_stats:
            result.summary += (
                f"\n  slope = {corr_stats['slope']:.4f} (SE = {corr_stats['se']:.4f})"
            )

        result.steps = self._steps
        return result
