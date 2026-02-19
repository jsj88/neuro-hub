"""
Neuro-model correlation tools.

CorrelateNeuroModel: Links trial-level model variables (RPE, Q-values)
with neural signals (EEG REWP amplitude, fMRI ROI betas).
"""

from typing import Any, Dict
import numpy as np
from scipy import stats

from .base import BaseTool


class CorrelateNeuroModel(BaseTool):
    """Agent tool: correlate model-derived variables with neural signals."""

    def __init__(self):
        super().__init__("CORRELATE_NEURO_MODEL")

    @property
    def description(self) -> str:
        return (
            "Correlate trial-level computational model variables with neural signals.\n"
            "Input JSON keys:\n"
            "  model_var: str — model variable name ('rpes', 'chosen_q', 'ev_values', etc.)\n"
            "  model_var_data: list — trial-level model variable values\n"
            "  neural_var: str — 'rewp_amplitude', 'decoding_score', or 'fmri_roi'\n"
            "  neural_data: list — trial-level neural values (optional if source='simulated')\n"
            "  source: str — 'simulated' or 'provided'\n"
            "  method: str — 'pearson', 'spearman', or 'regression' (default 'pearson')\n"
            "  channels: list — EEG channels for amplitude extraction (default ['FCz'])\n"
            "  time_window: list — [tmin, tmax] in seconds (default [0.24, 0.34])\n"
            "Returns: correlation r, p-value, regression stats."
        )

    def __call__(self, params: Dict[str, Any]) -> str:
        model_var_name = params.get("model_var", "rpes")
        model_data = np.array(params.get("model_var_data", []))
        neural_var = params.get("neural_var", "rewp_amplitude")
        neural_data = params.get("neural_data", None)
        method = params.get("method", "pearson")
        source = params.get("source", "provided")
        channels = params.get("channels", ["FCz"])
        time_window = params.get("time_window", [0.240, 0.340])

        # If source is simulated, extract REWP amplitude from simulated EEG
        if source == "simulated":
            sim_data = params.get("_simulated_data")
            if sim_data is None:
                return "ERROR: No simulated data. Run SIMULATE_NEURAL first."

            data = sim_data["data"]      # (n_epochs, n_channels, n_times)
            times = sim_data["times"]
            ch_names = sim_data["channels"]

            # Find channel indices
            ch_idx = [i for i, ch in enumerate(ch_names) if ch in channels]
            if not ch_idx:
                ch_idx = list(range(min(5, data.shape[1])))

            # Find time indices
            t_mask = (times >= time_window[0]) & (times <= time_window[1])

            # Mean amplitude in REWP window across selected channels
            neural_data = data[:, ch_idx, :][:, :, t_mask].mean(axis=(1, 2))

        elif neural_data is not None:
            neural_data = np.array(neural_data)
        else:
            return "ERROR: Provide neural_data or use source='simulated'."

        # Align lengths
        n = min(len(model_data), len(neural_data))
        model_data = model_data[:n]
        neural_data = neural_data[:n]

        # Remove NaN / Inf
        valid = np.isfinite(model_data) & np.isfinite(neural_data)
        model_data = model_data[valid]
        neural_data = neural_data[valid]

        if len(model_data) < 5:
            return "ERROR: Fewer than 5 valid data points for correlation."

        # Compute correlation
        if method == "pearson":
            r, p = stats.pearsonr(model_data, neural_data)
            corr_type = "Pearson"
        elif method == "spearman":
            r, p = stats.spearmanr(model_data, neural_data)
            corr_type = "Spearman"
        elif method == "regression":
            from scipy.stats import linregress
            slope, intercept, r, p, se = linregress(model_data, neural_data)
            return (
                f"Linear Regression: {model_var_name} → {neural_var}\n"
                f"  slope = {slope:.4f} (SE = {se:.4f})\n"
                f"  intercept = {intercept:.4f}\n"
                f"  r = {r:.4f}, r² = {r**2:.4f}\n"
                f"  p = {p:.2e}\n"
                f"  n = {len(model_data)}\n"
                f"  Significant (p < .05): {'YES ✓' if p < 0.05 else 'NO ✗'}"
            )
        else:
            return f"ERROR: Unknown method '{method}'. Use 'pearson', 'spearman', or 'regression'."

        sig_str = "YES ✓" if p < 0.05 else "NO ✗"

        return (
            f"{corr_type} Correlation: {model_var_name} ↔ {neural_var}\n"
            f"  r = {r:.4f}\n"
            f"  p = {p:.2e}\n"
            f"  n = {len(model_data)}\n"
            f"  Significant (p < .05): {sig_str}\n"
            f"  Effect size (r²): {r**2:.4f}"
        )
