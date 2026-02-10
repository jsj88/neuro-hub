"""
Forest and funnel plot visualization for effect size meta-analysis.

Provides publication-quality plots for presenting meta-analysis results.
"""

from typing import Optional, List, Tuple, Dict, Any
from pathlib import Path
import numpy as np


class ForestPlotter:
    """
    Create forest plots for effect size meta-analysis.
    
    Example:
        >>> plotter = ForestPlotter()
        >>> fig = plotter.plot(
        ...     effects=[0.5, 0.3, 0.7, 0.4],
        ...     variances=[0.04, 0.05, 0.03, 0.06],
        ...     study_labels=["Smith 2020", "Jones 2019", "Chen 2021", "Kim 2022"]
        ... )
        >>> fig.savefig("forest_plot.png", dpi=300)
    """
    
    def __init__(self, output_dir: Optional[str] = None):
        """
        Initialize plotter.
        
        Args:
            output_dir: Default directory for saving figures
        """
        self.output_dir = Path(output_dir) if output_dir else None
    
    def plot(
        self,
        effects: List[float],
        variances: List[float],
        study_labels: Optional[List[str]] = None,
        combined_effect: Optional[float] = None,
        combined_ci: Optional[Tuple[float, float]] = None,
        weights: Optional[List[float]] = None,
        title: str = "Forest Plot",
        effect_label: str = "Effect Size",
        ci_level: float = 0.95,
        show_weights: bool = True,
        sort_by: Optional[str] = None,
        output_path: Optional[str] = None,
        figsize: Tuple[int, int] = (10, None)
    ):
        """
        Create a forest plot.
        
        Args:
            effects: Effect sizes for each study
            variances: Variance of each effect size
            study_labels: Labels for each study
            combined_effect: Pooled effect size
            combined_ci: Confidence interval for combined effect
            weights: Study weights (auto-calculated if None)
            title: Plot title
            effect_label: Label for x-axis
            ci_level: Confidence level for intervals
            show_weights: Display weight column
            sort_by: Sort studies by "effect", "weight", or None
            output_path: Path to save figure
            figsize: Figure size (height auto-calculated if None)
            
        Returns:
            matplotlib Figure
        """
        import matplotlib.pyplot as plt
        from scipy import stats
        
        n_studies = len(effects)
        effects = np.array(effects)
        variances = np.array(variances)
        se = np.sqrt(variances)
        
        # Z value for CI
        z_crit = stats.norm.ppf((1 + ci_level) / 2)
        
        # Calculate CIs
        ci_lower = effects - z_crit * se
        ci_upper = effects + z_crit * se
        
        # Calculate weights if not provided
        if weights is None:
            weights = 1.0 / variances
            weights = weights / weights.sum()
        weights = np.array(weights)
        
        # Default labels
        if study_labels is None:
            study_labels = [f"Study {i+1}" for i in range(n_studies)]
        
        # Sort if requested
        if sort_by == "effect":
            order = np.argsort(effects)
        elif sort_by == "weight":
            order = np.argsort(weights)
        else:
            order = np.arange(n_studies)
        
        effects = effects[order]
        ci_lower = ci_lower[order]
        ci_upper = ci_upper[order]
        weights = weights[order]
        study_labels = [study_labels[i] for i in order]
        
        # Calculate combined effect if not provided
        if combined_effect is None:
            combined_effect = np.sum(weights * effects)
        if combined_ci is None:
            combined_se = np.sqrt(1.0 / np.sum(1.0 / variances))
            combined_ci = (
                combined_effect - z_crit * combined_se,
                combined_effect + z_crit * combined_se
            )
        
        # Figure size
        height = figsize[1] or (n_studies * 0.5 + 2)
        fig, ax = plt.subplots(figsize=(figsize[0], height))
        
        y_positions = np.arange(n_studies)
        
        # Plot individual studies
        for i in range(n_studies):
            # CI line
            ax.hlines(i, ci_lower[i], ci_upper[i], colors='black', linewidth=1.5)
            
            # Point estimate (square sized by weight)
            marker_size = weights[i] * 1000 + 30
            ax.scatter(effects[i], i, s=marker_size, marker='s', 
                      color='steelblue', edgecolors='black', zorder=3)
        
        # Plot combined effect (diamond)
        diamond_y = -1.5
        diamond_x = [combined_ci[0], combined_effect, combined_ci[1], combined_effect]
        diamond_y_coords = [diamond_y, diamond_y + 0.3, diamond_y, diamond_y - 0.3]
        ax.fill(diamond_x, diamond_y_coords, color='darkred', edgecolor='black')
        
        # Null effect line
        ax.axvline(x=0, color='gray', linestyle='--', linewidth=1, alpha=0.7)
        
        # Labels
        for i, label in enumerate(study_labels):
            ax.text(-0.02, i, label, ha='right', va='center', 
                   transform=ax.get_yaxis_transform(), fontsize=10)
        
        ax.text(-0.02, diamond_y, "Combined", ha='right', va='center',
               transform=ax.get_yaxis_transform(), fontsize=10, fontweight='bold')
        
        # Effect size and CI text on right
        for i in range(n_studies):
            ci_text = f"{effects[i]:.2f} [{ci_lower[i]:.2f}, {ci_upper[i]:.2f}]"
            ax.text(1.02, i, ci_text, ha='left', va='center',
                   transform=ax.get_yaxis_transform(), fontsize=9)
        
        ci_text = f"{combined_effect:.2f} [{combined_ci[0]:.2f}, {combined_ci[1]:.2f}]"
        ax.text(1.02, diamond_y, ci_text, ha='left', va='center',
               transform=ax.get_yaxis_transform(), fontsize=9, fontweight='bold')
        
        # Weight column
        if show_weights:
            for i in range(n_studies):
                ax.text(1.25, i, f"{weights[i]*100:.1f}%", ha='left', va='center',
                       transform=ax.get_yaxis_transform(), fontsize=9)
            ax.text(1.25, n_studies + 0.5, "Weight", ha='left', va='center',
                   transform=ax.get_yaxis_transform(), fontsize=9, fontweight='bold')
        
        # Formatting
        ax.set_ylim(-2.5, n_studies + 0.5)
        ax.set_yticks([])
        ax.set_xlabel(effect_label, fontsize=11)
        ax.set_title(title, fontsize=12, fontweight='bold')
        
        # Header
        ax.text(-0.02, n_studies + 0.5, "Study", ha='right', va='center',
               transform=ax.get_yaxis_transform(), fontsize=9, fontweight='bold')
        ax.text(1.02, n_studies + 0.5, f"{effect_label} [{int(ci_level*100)}% CI]", 
               ha='left', va='center', transform=ax.get_yaxis_transform(), 
               fontsize=9, fontweight='bold')
        
        plt.tight_layout()
        
        if output_path:
            self._save_figure(fig, output_path)
        
        return fig
    
    def _save_figure(self, fig, path: str):
        """Save figure to file."""
        output_path = Path(path)
        if self.output_dir:
            output_path = self.output_dir / output_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches='tight')


class FunnelPlotter:
    """
    Create funnel plots for publication bias assessment.
    
    Example:
        >>> plotter = FunnelPlotter()
        >>> fig = plotter.plot(
        ...     effects=[0.5, 0.3, 0.7, 0.4],
        ...     se=[0.2, 0.22, 0.17, 0.25]
        ... )
    """
    
    def __init__(self, output_dir: Optional[str] = None):
        self.output_dir = Path(output_dir) if output_dir else None
    
    def plot(
        self,
        effects: List[float],
        se: List[float],
        combined_effect: Optional[float] = None,
        title: str = "Funnel Plot",
        effect_label: str = "Effect Size",
        show_contours: bool = True,
        contour_levels: List[float] = [0.01, 0.05, 0.10],
        output_path: Optional[str] = None,
        figsize: Tuple[int, int] = (8, 6)
    ):
        """
        Create a funnel plot.
        
        Args:
            effects: Effect sizes
            se: Standard errors
            combined_effect: Pooled effect (auto-calculated if None)
            title: Plot title
            effect_label: Label for x-axis
            show_contours: Show significance contours
            contour_levels: Alpha levels for contours
            output_path: Path to save figure
            figsize: Figure size
            
        Returns:
            matplotlib Figure
        """
        import matplotlib.pyplot as plt
        from scipy import stats
        
        effects = np.array(effects)
        se = np.array(se)
        
        # Calculate combined effect if not provided
        if combined_effect is None:
            weights = 1.0 / (se ** 2)
            combined_effect = np.sum(weights * effects) / np.sum(weights)
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot studies
        ax.scatter(effects, se, s=60, alpha=0.7, edgecolors='black', zorder=3)
        
        # Combined effect line
        ax.axvline(x=combined_effect, color='red', linestyle='-', 
                  linewidth=2, label='Combined effect')
        
        # Pseudo-confidence interval funnel
        se_range = np.linspace(0.001, max(se) * 1.2, 100)
        
        if show_contours:
            for alpha in contour_levels:
                z = stats.norm.ppf(1 - alpha/2)
                ax.plot(combined_effect - z * se_range, se_range, 
                       'k--', alpha=0.3, linewidth=1)
                ax.plot(combined_effect + z * se_range, se_range, 
                       'k--', alpha=0.3, linewidth=1)
        
        # Fill funnel
        z_95 = stats.norm.ppf(0.975)
        ax.fill_betweenx(se_range, 
                         combined_effect - z_95 * se_range,
                         combined_effect + z_95 * se_range,
                         alpha=0.1, color='gray')
        
        # Formatting
        ax.set_xlabel(effect_label, fontsize=11)
        ax.set_ylabel("Standard Error", fontsize=11)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.invert_yaxis()  # SE increases downward
        ax.legend(loc='lower right')
        
        # Add interpretation note
        ax.text(0.02, 0.02, 
                "Asymmetry may indicate publication bias",
                transform=ax.transAxes, fontsize=8, 
                style='italic', alpha=0.7)
        
        plt.tight_layout()
        
        if output_path:
            self._save_figure(fig, output_path)
        
        return fig
    
    def trim_and_fill(
        self,
        effects: List[float],
        se: List[float],
        side: str = "right",
        output_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Perform trim-and-fill analysis for publication bias.
        
        Args:
            effects: Effect sizes
            se: Standard errors
            side: Side to impute ("left" or "right")
            output_path: Path to save figure
            
        Returns:
            Dictionary with adjusted estimate and imputed studies
        """
        import matplotlib.pyplot as plt
        from scipy import stats
        
        effects = np.array(effects)
        se = np.array(se)
        
        # Calculate initial combined effect
        weights = 1.0 / (se ** 2)
        combined = np.sum(weights * effects) / np.sum(weights)
        
        # Simple trim-and-fill implementation
        # Center effects around combined
        centered = effects - combined
        
        # Find asymmetric studies
        if side == "right":
            asymmetric = centered > 0
        else:
            asymmetric = centered < 0
        
        # Mirror asymmetric studies
        imputed_effects = []
        imputed_se = []
        
        for i, is_asym in enumerate(asymmetric):
            if is_asym:
                mirror_effect = combined - (effects[i] - combined)
                imputed_effects.append(mirror_effect)
                imputed_se.append(se[i])
        
        # Recalculate with imputed studies
        all_effects = np.concatenate([effects, imputed_effects])
        all_se = np.concatenate([se, imputed_se])
        
        all_weights = 1.0 / (all_se ** 2)
        adjusted_combined = np.sum(all_weights * all_effects) / np.sum(all_weights)
        adjusted_se = np.sqrt(1.0 / np.sum(all_weights))
        
        # Create plot
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Original studies
        ax.scatter(effects, se, s=60, alpha=0.7, edgecolors='black', 
                  label='Original studies', zorder=3)
        
        # Imputed studies
        if imputed_effects:
            ax.scatter(imputed_effects, imputed_se, s=60, alpha=0.5, 
                      marker='s', edgecolors='black', facecolors='none',
                      label='Imputed studies', zorder=3)
        
        # Lines
        ax.axvline(x=combined, color='blue', linestyle='--', 
                  linewidth=1.5, label=f'Original: {combined:.3f}')
        ax.axvline(x=adjusted_combined, color='red', linestyle='-', 
                  linewidth=1.5, label=f'Adjusted: {adjusted_combined:.3f}')
        
        # Funnel
        se_range = np.linspace(0.001, max(all_se) * 1.2, 100)
        z_95 = stats.norm.ppf(0.975)
        ax.fill_betweenx(se_range,
                         adjusted_combined - z_95 * se_range,
                         adjusted_combined + z_95 * se_range,
                         alpha=0.1, color='gray')
        
        ax.set_xlabel("Effect Size", fontsize=11)
        ax.set_ylabel("Standard Error", fontsize=11)
        ax.set_title("Trim-and-Fill Analysis", fontsize=12, fontweight='bold')
        ax.invert_yaxis()
        ax.legend(loc='lower right', fontsize=9)
        
        plt.tight_layout()
        
        if output_path:
            output_path = Path(output_path)
            if self.output_dir:
                output_path = self.output_dir / output_path
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(output_path, dpi=300, bbox_inches='tight')
        
        return {
            "original_combined": combined,
            "adjusted_combined": adjusted_combined,
            "adjusted_se": adjusted_se,
            "n_imputed": len(imputed_effects),
            "imputed_effects": imputed_effects,
            "figure": fig
        }
    
    def _save_figure(self, fig, path: str):
        """Save figure to file."""
        output_path = Path(path)
        if self.output_dir:
            output_path = self.output_dir / output_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
