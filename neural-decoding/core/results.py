"""
DecodingResults - Container for classification results.

Stores accuracy, predictions, confusion matrix, and other metrics
from neural decoding analyses.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
import numpy as np
import json
from pathlib import Path


@dataclass
class DecodingResults:
    """
    Container for neural decoding results.
    
    Stores classification accuracy, predictions, confusion matrix,
    feature importances, and cross-validation scores.
    
    Attributes:
        accuracy: Overall classification accuracy
        accuracy_per_class: Accuracy for each class
        confusion_matrix: Confusion matrix (n_classes x n_classes)
        predictions: Predicted labels for each sample
        probabilities: Class probabilities (n_samples x n_classes)
        feature_importances: Importance of each feature
        cv_scores: Accuracy for each CV fold
        permutation_pvalue: P-value from permutation test
        model: Trained classifier object
        class_names: Names of classes
        metadata: Additional information
    
    Example:
        >>> results = decoder.cross_validate(dataset)
        >>> print(f"Accuracy: {results.accuracy:.1%}")
        >>> results.plot_confusion_matrix()
    """
    
    accuracy: float
    accuracy_per_class: Optional[Dict[str, float]] = None
    confusion_matrix: Optional[np.ndarray] = None
    predictions: Optional[np.ndarray] = None
    true_labels: Optional[np.ndarray] = None
    probabilities: Optional[np.ndarray] = None
    feature_importances: Optional[np.ndarray] = None
    cv_scores: Optional[List[float]] = None
    permutation_pvalue: Optional[float] = None
    permutation_scores: Optional[np.ndarray] = None
    model: Optional[Any] = None
    class_names: Optional[List[str]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def cv_mean(self) -> Optional[float]:
        """Mean cross-validation accuracy."""
        if self.cv_scores is None:
            return None
        return np.mean(self.cv_scores)
    
    @property
    def cv_std(self) -> Optional[float]:
        """Standard deviation of cross-validation accuracy."""
        if self.cv_scores is None:
            return None
        return np.std(self.cv_scores)
    
    @property
    def n_folds(self) -> int:
        """Number of CV folds."""
        if self.cv_scores is None:
            return 0
        return len(self.cv_scores)
    
    @property
    def is_significant(self) -> Optional[bool]:
        """Whether result is statistically significant (p < 0.05)."""
        if self.permutation_pvalue is None:
            return None
        return self.permutation_pvalue < 0.05
    
    def precision(self, class_idx: int = None) -> float:
        """
        Calculate precision.
        
        Args:
            class_idx: Class index (None for macro average)
        """
        if self.confusion_matrix is None:
            raise ValueError("No confusion matrix available")
        
        cm = self.confusion_matrix
        
        if class_idx is not None:
            tp = cm[class_idx, class_idx]
            fp = cm[:, class_idx].sum() - tp
            return tp / (tp + fp) if (tp + fp) > 0 else 0.0
        
        # Macro average
        precisions = []
        for i in range(cm.shape[0]):
            precisions.append(self.precision(i))
        return np.mean(precisions)
    
    def recall(self, class_idx: int = None) -> float:
        """
        Calculate recall (sensitivity).
        
        Args:
            class_idx: Class index (None for macro average)
        """
        if self.confusion_matrix is None:
            raise ValueError("No confusion matrix available")
        
        cm = self.confusion_matrix
        
        if class_idx is not None:
            tp = cm[class_idx, class_idx]
            fn = cm[class_idx, :].sum() - tp
            return tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        # Macro average
        recalls = []
        for i in range(cm.shape[0]):
            recalls.append(self.recall(i))
        return np.mean(recalls)
    
    def f1_score(self, class_idx: int = None) -> float:
        """
        Calculate F1 score.
        
        Args:
            class_idx: Class index (None for macro average)
        """
        p = self.precision(class_idx)
        r = self.recall(class_idx)
        return 2 * (p * r) / (p + r) if (p + r) > 0 else 0.0
    
    def get_top_features(self, n: int = 20) -> List[tuple]:
        """
        Get top N features by importance.
        
        Args:
            n: Number of features to return
            
        Returns:
            List of (feature_index, importance) tuples
        """
        if self.feature_importances is None:
            raise ValueError("No feature importances available")
        
        indices = np.argsort(np.abs(self.feature_importances))[::-1][:n]
        return [(int(i), float(self.feature_importances[i])) for i in indices]
    
    def summary(self) -> str:
        """Generate text summary of results."""
        lines = [
            "Decoding Results Summary",
            "=" * 40,
            f"Accuracy: {self.accuracy:.1%}",
        ]
        
        if self.cv_scores is not None:
            lines.append(f"CV Accuracy: {self.cv_mean:.1%} (+/- {self.cv_std:.1%})")
            lines.append(f"CV Folds: {self.n_folds}")
        
        if self.permutation_pvalue is not None:
            sig = "significant" if self.is_significant else "not significant"
            lines.append(f"Permutation p-value: {self.permutation_pvalue:.4f} ({sig})")
        
        if self.accuracy_per_class:
            lines.append("")
            lines.append("Per-class accuracy:")
            for name, acc in self.accuracy_per_class.items():
                lines.append(f"  {name}: {acc:.1%}")
        
        if self.confusion_matrix is not None:
            lines.append("")
            lines.append(f"Precision: {self.precision():.1%}")
            lines.append(f"Recall: {self.recall():.1%}")
            lines.append(f"F1 Score: {self.f1_score():.1%}")
        
        return "\n".join(lines)
    
    def plot_confusion_matrix(
        self,
        normalize: bool = True,
        cmap: str = "Blues",
        figsize: tuple = (8, 6),
        output_path: Optional[str] = None
    ):
        """
        Plot confusion matrix.
        
        Args:
            normalize: Normalize by row (true labels)
            cmap: Colormap name
            figsize: Figure size
            output_path: Path to save figure
            
        Returns:
            matplotlib Figure
        """
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        if self.confusion_matrix is None:
            raise ValueError("No confusion matrix available")
        
        cm = self.confusion_matrix.copy()
        
        if normalize:
            cm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
            fmt = ".2f"
            title = "Normalized Confusion Matrix"
        else:
            fmt = "d"
            title = "Confusion Matrix"
        
        fig, ax = plt.subplots(figsize=figsize)
        
        labels = self.class_names or [str(i) for i in range(cm.shape[0])]
        
        sns.heatmap(
            cm, annot=True, fmt=fmt, cmap=cmap,
            xticklabels=labels, yticklabels=labels,
            ax=ax
        )
        
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(title)
        
        plt.tight_layout()
        
        if output_path:
            fig.savefig(output_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_cv_scores(
        self,
        figsize: tuple = (8, 4),
        output_path: Optional[str] = None
    ):
        """
        Plot cross-validation scores.
        
        Args:
            figsize: Figure size
            output_path: Path to save figure
            
        Returns:
            matplotlib Figure
        """
        import matplotlib.pyplot as plt
        
        if self.cv_scores is None:
            raise ValueError("No CV scores available")
        
        fig, ax = plt.subplots(figsize=figsize)
        
        folds = range(1, len(self.cv_scores) + 1)
        ax.bar(folds, self.cv_scores, color='steelblue', edgecolor='black')
        ax.axhline(y=self.cv_mean, color='red', linestyle='--', 
                  label=f'Mean: {self.cv_mean:.1%}')
        
        ax.set_xlabel("Fold")
        ax.set_ylabel("Accuracy")
        ax.set_title("Cross-Validation Scores")
        ax.set_xticks(folds)
        ax.legend()
        ax.set_ylim(0, 1)
        
        plt.tight_layout()
        
        if output_path:
            fig.savefig(output_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def save(self, path: str):
        """Save results to JSON file."""
        path = Path(path)
        
        data = {
            "accuracy": self.accuracy,
            "accuracy_per_class": self.accuracy_per_class,
            "cv_scores": list(self.cv_scores) if self.cv_scores else None,
            "cv_mean": self.cv_mean,
            "cv_std": self.cv_std,
            "permutation_pvalue": self.permutation_pvalue,
            "precision": self.precision() if self.confusion_matrix is not None else None,
            "recall": self.recall() if self.confusion_matrix is not None else None,
            "f1_score": self.f1_score() if self.confusion_matrix is not None else None,
            "class_names": self.class_names,
            "metadata": self.metadata
        }
        
        # Save confusion matrix separately
        if self.confusion_matrix is not None:
            np.save(path.with_suffix(".confusion.npy"), self.confusion_matrix)
            data["confusion_matrix_path"] = str(path.with_suffix(".confusion.npy"))
        
        if self.feature_importances is not None:
            np.save(path.with_suffix(".importance.npy"), self.feature_importances)
            data["feature_importances_path"] = str(path.with_suffix(".importance.npy"))
        
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
    
    def __repr__(self) -> str:
        cv_str = f", cv_mean={self.cv_mean:.1%}" if self.cv_scores else ""
        pval_str = f", p={self.permutation_pvalue:.4f}" if self.permutation_pvalue else ""
        return f"DecodingResults(accuracy={self.accuracy:.1%}{cv_str}{pval_str})"
