"""
Effect size meta-analysis using PyMARE.

This module provides a wrapper around PyMARE for traditional
random-effects meta-analysis of effect sizes.
"""

from typing import Dict, Any, Optional, List
from pathlib import Path
import numpy as np


class EffectSizeMetaAnalysis:
    """
    Effect size meta-analysis using random effects models.

    Supports DerSimonian-Laird, REML, and other estimators via PyMARE.

    Example:
        >>> from core import MetaAnalysisDataset
        >>> dataset = MetaAnalysisDataset.load("my_studies.json")
        >>> ma = EffectSizeMetaAnalysis(dataset)
        >>> results = ma.run()
        >>> print(f"Combined effect: {results['combined_effect']:.3f}")
    """

    def __init__(self, dataset: "MetaAnalysisDataset"):
        """
        Initialize effect size meta-analysis.

        Args:
            dataset: MetaAnalysisDataset containing studies with effect sizes
        """
        self.dataset = dataset
        self.results = None
        self._check_pymare()

    def _check_pymare(self):
        """Check that PyMARE is installed."""
        try:
            import pymare
        except ImportError:
            raise ImportError(
                "PyMARE is required for effect size meta-analysis. "
                "Install with: pip install pymare"
            )

    def run(
        self,
        method: str = "DL",
        outcome_filter: Optional[str] = None,
        ci_level: float = 0.95
    ) -> Dict[str, Any]:
        """
        Run effect size meta-analysis.

        Args:
            method: Estimation method:
                - "DL": DerSimonian-Laird (default, widely used)
                - "REML": Restricted maximum likelihood
                - "HE": Hedges estimator
                - "FE": Fixed effects (no heterogeneity)
            outcome_filter: Only include effect sizes with this outcome name
            ci_level: Confidence interval level (default 0.95)

        Returns:
            Dictionary containing:
                - "combined_effect": Pooled effect size
                - "combined_se": Standard error of combined effect
                - "combined_ci": Confidence interval tuple
                - "tau_squared": Between-study variance
                - "i_squared": Heterogeneity percentage
                - "q_statistic": Cochran's Q
                - "q_pvalue": P-value for Q test
                - "study_weights": Weight for each study
                - "study_effects": Effect size for each study
        """
        from pymare import Dataset as PyMAREDataset
        from pymare.estimators import (
            DerSimonianLaird,
            Hedges,
            VarianceBasedLikelihoodEstimator
        )

        # Get effect sizes DataFrame
        es_df = self.dataset.to_effect_sizes_df()

        if outcome_filter:
            es_df = es_df[
                es_df["outcome_name"].str.contains(outcome_filter, case=False, na=False)
            ]

        if es_df.empty:
            raise ValueError("No effect sizes found in dataset")

        # Check for missing variance
        if es_df["variance"].isna().any():
            raise ValueError(
                "Some effect sizes have missing variance. "
                "Ensure all EffectSize objects have variance computed."
            )

        n_studies = len(es_df)
        print(f"Running {method} meta-analysis on {n_studies} effect sizes...")

        # Extract arrays
        y = es_df["effect_size"].values
        v = es_df["variance"].values

        # Create PyMARE dataset
        pymare_ds = PyMAREDataset(y=y, v=v)

        # Select estimator
        if method == "DL":
            estimator = DerSimonianLaird()
        elif method == "REML":
            estimator = VarianceBasedLikelihoodEstimator(method="REML")
        elif method == "HE":
            estimator = Hedges()
        elif method == "FE":
            # Fixed effects: tau^2 = 0
            estimator = DerSimonianLaird()
            # Will manually set tau^2 = 0
        else:
            raise ValueError(f"Unknown method: {method}")

        # Fit model
        self.results = estimator.fit_dataset(pymare_ds)

        # Extract results
        return self._build_results_dict(y, v, es_df, ci_level, method)

    def _build_results_dict(
        self,
        y: np.ndarray,
        v: np.ndarray,
        es_df,
        ci_level: float,
        method: str
    ) -> Dict[str, Any]:
        """Build results dictionary from PyMARE results."""
        from scipy import stats

        # Get estimates from results
        combined_effect = float(self.results.fe_params[0])

        # Get tau-squared (between-study variance)
        tau_squared = float(self.results.tau2) if hasattr(self.results, 'tau2') else 0.0

        if method == "FE":
            tau_squared = 0.0

        # Compute combined SE
        # Weight = 1 / (v_i + tau^2)
        weights = 1.0 / (v + tau_squared)
        weights = weights / weights.sum()
        combined_var = 1.0 / np.sum(1.0 / (v + tau_squared))
        combined_se = np.sqrt(combined_var)

        # Compute CI
        z_crit = stats.norm.ppf((1 + ci_level) / 2)
        ci_lower = combined_effect - z_crit * combined_se
        ci_upper = combined_effect + z_crit * combined_se

        # Compute heterogeneity statistics
        q_stat, i_squared, q_pvalue = self._compute_heterogeneity(y, v, combined_effect)

        return {
            "combined_effect": combined_effect,
            "combined_se": combined_se,
            "combined_ci": (ci_lower, ci_upper),
            "combined_z": combined_effect / combined_se,
            "combined_pvalue": 2 * (1 - stats.norm.cdf(abs(combined_effect / combined_se))),
            "tau_squared": tau_squared,
            "tau": np.sqrt(tau_squared),
            "i_squared": i_squared,
            "q_statistic": q_stat,
            "q_pvalue": q_pvalue,
            "n_studies": len(y),
            "study_weights": weights.tolist(),
            "study_effects": y.tolist(),
            "study_ids": es_df["study_id"].tolist(),
            "method": method,
            "ci_level": ci_level
        }

    def _compute_heterogeneity(
        self,
        y: np.ndarray,
        v: np.ndarray,
        combined_effect: float
    ) -> tuple:
        """Compute Q statistic and I-squared."""
        from scipy import stats

        k = len(y)

        # Fixed-effect weights
        w = 1.0 / v

        # Cochran's Q
        q = np.sum(w * (y - combined_effect)**2)

        # Degrees of freedom
        df = k - 1

        # P-value for Q
        q_pvalue = 1 - stats.chi2.cdf(q, df) if df > 0 else 1.0

        # I-squared
        if q > df:
            i_squared = ((q - df) / q) * 100
        else:
            i_squared = 0.0

        return q, i_squared, q_pvalue

    def forest_plot(
        self,
        output_path: Optional[str] = None,
        title: str = "Forest Plot",
        effect_label: str = "Effect Size (d)",
        show_weights: bool = True,
        figsize: tuple = (10, None)
    ):
        """
        Generate a forest plot.

        Args:
            output_path: Path to save figure
            title: Plot title
            effect_label: Label for x-axis
            show_weights: Show study weights
            figsize: Figure size (width, height). Height auto-calculated if None.

        Returns:
            matplotlib Figure
        """
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches

        if self.results is None:
            raise ValueError("Run analysis first with .run()")

        es_df = self.dataset.to_effect_sizes_df()
        n_studies = len(es_df)

        # Calculate figure height if not specified
        height = figsize[1] or (n_studies * 0.5 + 2)
        fig, ax = plt.subplots(figsize=(figsize[0], height))

        # Get results
        results = self._build_results_dict(
            es_df["effect_size"].values,
            es_df["variance"].values,
            es_df,
            0.95,
            "DL"
        )

        y_positions = list(range(n_studies))

        # Plot individual studies
        for i, (_, row) in enumerate(es_df.iterrows()):
            es = row["effect_size"]
            se = np.sqrt(row["variance"])
            ci_low = es - 1.96 * se
            ci_high = es + 1.96 * se
            weight = results["study_weights"][i]

            # CI line
            ax.hlines(i, ci_low, ci_high, colors='black', linewidth=1)

            # Point estimate (square, size proportional to weight)
            marker_size = weight * 500 + 20
            ax.scatter(es, i, s=marker_size, marker='s', color='black', zorder=3)

            # Study label
            label = f"{row['study_id']} ({row['year']})" if 'year' in row else row['study_id']
            ax.text(-0.1, i, label, ha='right', va='center', transform=ax.get_yaxis_transform())

            # Effect size and CI text
            ci_text = f"{es:.2f} [{ci_low:.2f}, {ci_high:.2f}]"
            ax.text(1.02, i, ci_text, ha='left', va='center', transform=ax.get_yaxis_transform())

        # Plot combined effect (diamond)
        combined = results["combined_effect"]
        ci = results["combined_ci"]
        diamond_y = -1

        diamond = mpatches.FancyBboxPatch(
            (ci[0], diamond_y - 0.3),
            ci[1] - ci[0], 0.6,
            boxstyle="round,pad=0",
            facecolor='black',
            edgecolor='black'
        )
        # Actually draw as polygon for diamond shape
        diamond_x = [ci[0], combined, ci[1], combined]
        diamond_y_coords = [diamond_y, diamond_y + 0.3, diamond_y, diamond_y - 0.3]
        ax.fill(diamond_x, diamond_y_coords, color='black')

        ax.text(-0.1, diamond_y, "Combined", ha='right', va='center',
                fontweight='bold', transform=ax.get_yaxis_transform())
        ax.text(1.02, diamond_y, f"{combined:.2f} [{ci[0]:.2f}, {ci[1]:.2f}]",
                ha='left', va='center', fontweight='bold', transform=ax.get_yaxis_transform())

        # Vertical line at null effect
        ax.axvline(x=0, color='gray', linestyle='--', linewidth=0.5)

        # Formatting
        ax.set_ylim(-2, n_studies)
        ax.set_yticks([])
        ax.set_xlabel(effect_label)
        ax.set_title(title)

        # Add heterogeneity stats
        het_text = f"I² = {results['i_squared']:.1f}%, Q = {results['q_statistic']:.1f}, p = {results['q_pvalue']:.3f}"
        ax.text(0.5, -0.1, het_text, ha='center', transform=ax.transAxes)

        plt.tight_layout()

        if output_path:
            fig.savefig(output_path, dpi=300, bbox_inches='tight')

        return fig

    def summary(self) -> str:
        """Generate text summary of analysis."""
        if self.results is None:
            return "Analysis not yet run. Call .run() first."

        es_df = self.dataset.to_effect_sizes_df()
        results = self._build_results_dict(
            es_df["effect_size"].values,
            es_df["variance"].values,
            es_df,
            0.95,
            "DL"
        )

        lines = [
            "Effect Size Meta-Analysis Summary",
            "=" * 40,
            f"Dataset: {self.dataset.name}",
            f"Studies: {results['n_studies']}",
            "",
            "Combined Effect:",
            f"  Effect size: {results['combined_effect']:.3f}",
            f"  95% CI: [{results['combined_ci'][0]:.3f}, {results['combined_ci'][1]:.3f}]",
            f"  Z = {results['combined_z']:.2f}, p = {results['combined_pvalue']:.4f}",
            "",
            "Heterogeneity:",
            f"  Q = {results['q_statistic']:.2f}, p = {results['q_pvalue']:.4f}",
            f"  I² = {results['i_squared']:.1f}%",
            f"  τ² = {results['tau_squared']:.4f}",
        ]

        return "\n".join(lines)
