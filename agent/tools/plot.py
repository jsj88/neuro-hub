"""
Plotting tool for the Neuro-Coscientist agent.

PlotAndSave: generates and saves publication-quality figures
for parameter recovery, model comparison, temporal decoding, and correlations.
"""

import os
from typing import Any, Dict
import numpy as np

from .base import BaseTool


class PlotAndSave(BaseTool):
    """Agent tool: generate and save figures."""

    def __init__(self):
        super().__init__("PLOT")

    @property
    def description(self) -> str:
        return (
            "Generate and save a figure.\n"
            "Input JSON keys:\n"
            "  plot_type: str — one of:\n"
            "    'model_comparison' — bar chart of BIC values\n"
            "    'parameter_recovery' — scatter: true vs recovered params\n"
            "    'temporal_decoding' — time course of decoding accuracy\n"
            "    'correlation' — scatter of model_var vs neural_var\n"
            "    'learning_curve' — Q-values over trials\n"
            "  data: dict — plot-specific data (see examples)\n"
            "  output_path: str — where to save the figure (default ./results/figures/)\n"
            "  title: str — figure title\n"
            "  figsize: list — [width, height] in inches (default [8, 5])\n"
            "Returns: path to saved figure."
        )

    def __call__(self, params: Dict[str, Any]) -> str:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plot_type = params.get("plot_type", "")
        data = params.get("data", {})
        title = params.get("title", "")
        figsize = tuple(params.get("figsize", [8, 5]))
        output_dir = params.get("output_path", "./results/figures")
        os.makedirs(output_dir, exist_ok=True)

        fig, ax = plt.subplots(figsize=figsize)

        if plot_type == "model_comparison":
            models = data.get("models", [])
            bics = data.get("bics", [])
            colors = ["#2ecc71" if i == 0 else "#95a5a6" for i in range(len(models))]
            ax.bar(models, bics, color=colors, edgecolor="black", linewidth=0.5)
            ax.set_ylabel("BIC")
            ax.set_title(title or "Model Comparison (BIC)")
            fname = "model_comparison.png"

        elif plot_type == "parameter_recovery":
            true_vals = data.get("true", [])
            recovered = data.get("recovered", [])
            param_name = data.get("param_name", "parameter")
            ax.scatter(true_vals, recovered, alpha=0.6, edgecolors="black", linewidths=0.5)
            lims = [min(min(true_vals), min(recovered)), max(max(true_vals), max(recovered))]
            ax.plot(lims, lims, "k--", alpha=0.5, label="identity")
            ax.set_xlabel(f"True {param_name}")
            ax.set_ylabel(f"Recovered {param_name}")
            ax.set_title(title or f"Parameter Recovery: {param_name}")
            ax.legend()
            fname = f"param_recovery_{param_name}.png"

        elif plot_type == "temporal_decoding":
            times_ms = np.array(data.get("times", [])) * 1000
            scores = np.array(data.get("scores", []))
            scores_std = np.array(data.get("scores_std", []))
            chance = data.get("chance", 0.5)

            ax.plot(times_ms, scores, "b-", linewidth=2)
            if len(scores_std) > 0:
                ax.fill_between(times_ms, scores - scores_std, scores + scores_std, alpha=0.2)
            ax.axhline(chance, color="k", linestyle="--", alpha=0.5, label="Chance")
            ax.axvspan(240, 340, alpha=0.15, color="orange", label="REWP window")
            ax.set_xlabel("Time (ms)")
            ax.set_ylabel("AUC")
            ax.set_title(title or "Temporal Decoding")
            ax.legend()
            fname = "temporal_decoding.png"

        elif plot_type == "correlation":
            x = np.array(data.get("x", []))
            y = np.array(data.get("y", []))
            x_label = data.get("x_label", "Model variable")
            y_label = data.get("y_label", "Neural signal")
            r_val = data.get("r", None)
            p_val = data.get("p", None)

            ax.scatter(x, y, alpha=0.5, edgecolors="black", linewidths=0.3)
            if len(x) > 2:
                z = np.polyfit(x, y, 1)
                p_line = np.poly1d(z)
                x_sorted = np.sort(x)
                ax.plot(x_sorted, p_line(x_sorted), "r-", linewidth=2)
            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label)
            stat_str = ""
            if r_val is not None:
                stat_str = f"r={r_val:.3f}"
            if p_val is not None:
                stat_str += f", p={p_val:.2e}"
            ax.set_title(title or f"Correlation ({stat_str})")
            fname = "correlation.png"

        elif plot_type == "learning_curve":
            trials = np.arange(len(data.get("q_values", [[]])[0]))
            q_vals = data.get("q_values", [])
            labels = data.get("option_labels", [f"Option {i}" for i in range(len(q_vals))])

            for i, q in enumerate(q_vals):
                ax.plot(trials, q, linewidth=1.5, label=labels[i])
            ax.set_xlabel("Trial")
            ax.set_ylabel("Q-value")
            ax.set_title(title or "Learning Curves (Q-values)")
            ax.legend()
            fname = "learning_curve.png"

        else:
            plt.close(fig)
            return f"ERROR: Unknown plot_type '{plot_type}'. Use: model_comparison, parameter_recovery, temporal_decoding, correlation, learning_curve"

        plt.tight_layout()
        out_path = os.path.join(output_dir, fname)
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

        return f"Figure saved: {out_path}"
