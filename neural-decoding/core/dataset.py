"""
DecodingDataset - Container for neural decoding data.

This module provides a unified data container for fMRI, EEG, 
and behavioral data used in classification analyses.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Union
import numpy as np
import json
from pathlib import Path


@dataclass
class DecodingDataset:
    """
    Container for neural decoding analysis data.
    
    Holds features (X), labels (y), and metadata for classification.
    Designed for pre-processed, analysis-ready data.
    
    Attributes:
        X: Feature matrix (n_samples, n_features)
        y: Class labels (n_samples,)
        groups: Group identifiers for CV (run, subject, etc.)
        feature_names: Names for each feature
        class_names: Names for each class
        metadata: Additional information (subject, task, etc.)
        modality: Data type ("fmri", "eeg", "behavior", "multimodal")
    
    Example:
        >>> dataset = DecodingDataset(
        ...     X=np.random.randn(100, 5000),
        ...     y=np.array([0, 1] * 50),
        ...     groups=np.repeat([1, 2, 3, 4, 5], 20),
        ...     class_names=["face", "house"],
        ...     modality="fmri"
        ... )
        >>> print(dataset.summary())
    """
    
    X: np.ndarray
    y: np.ndarray
    groups: Optional[np.ndarray] = None
    feature_names: Optional[List[str]] = None
    class_names: Optional[List[str]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    modality: str = "unknown"
    
    def __post_init__(self):
        """Validate and set defaults after initialization."""
        # Convert to numpy arrays
        self.X = np.asarray(self.X)
        self.y = np.asarray(self.y)
        
        if self.groups is not None:
            self.groups = np.asarray(self.groups)
        
        # Validate shapes
        if self.X.shape[0] != len(self.y):
            raise ValueError(
                f"X and y must have same number of samples. "
                f"Got X: {self.X.shape[0]}, y: {len(self.y)}"
            )
        
        if self.groups is not None and len(self.groups) != len(self.y):
            raise ValueError(
                f"groups must have same length as y. "
                f"Got groups: {len(self.groups)}, y: {len(self.y)}"
            )
        
        # Set default feature names
        if self.feature_names is None:
            self.feature_names = [f"feature_{i}" for i in range(self.n_features)]
        
        # Set default class names from unique labels
        if self.class_names is None:
            unique_labels = np.unique(self.y)
            self.class_names = [str(label) for label in unique_labels]
    
    @property
    def n_samples(self) -> int:
        """Number of samples."""
        return self.X.shape[0]
    
    @property
    def n_features(self) -> int:
        """Number of features."""
        return self.X.shape[1]
    
    @property
    def n_classes(self) -> int:
        """Number of unique classes."""
        return len(np.unique(self.y))
    
    @property
    def n_groups(self) -> int:
        """Number of unique groups (runs, subjects, etc.)."""
        if self.groups is None:
            return 1
        return len(np.unique(self.groups))
    
    @property
    def class_counts(self) -> Dict[str, int]:
        """Count of samples per class."""
        unique, counts = np.unique(self.y, return_counts=True)
        return {self.class_names[i] if i < len(self.class_names) else str(u): int(c) 
                for i, (u, c) in enumerate(zip(unique, counts))}
    
    @property
    def is_balanced(self) -> bool:
        """Check if classes are balanced (within 10%)."""
        counts = list(self.class_counts.values())
        if not counts:
            return True
        return (max(counts) - min(counts)) / max(counts) < 0.1
    
    def get_subset(
        self,
        indices: Optional[np.ndarray] = None,
        classes: Optional[List] = None,
        groups: Optional[List] = None
    ) -> "DecodingDataset":
        """
        Get subset of dataset.
        
        Args:
            indices: Sample indices to include
            classes: Class labels to include
            groups: Group IDs to include
            
        Returns:
            New DecodingDataset with subset
        """
        mask = np.ones(self.n_samples, dtype=bool)
        
        if indices is not None:
            idx_mask = np.zeros(self.n_samples, dtype=bool)
            idx_mask[indices] = True
            mask &= idx_mask
        
        if classes is not None:
            mask &= np.isin(self.y, classes)
        
        if groups is not None and self.groups is not None:
            mask &= np.isin(self.groups, groups)
        
        return DecodingDataset(
            X=self.X[mask],
            y=self.y[mask],
            groups=self.groups[mask] if self.groups is not None else None,
            feature_names=self.feature_names,
            class_names=self.class_names,
            metadata=self.metadata.copy(),
            modality=self.modality
        )
    
    def split_by_group(self, test_group: Any) -> tuple:
        """
        Split dataset by group for cross-validation.
        
        Args:
            test_group: Group ID to use as test set
            
        Returns:
            Tuple of (train_dataset, test_dataset)
        """
        if self.groups is None:
            raise ValueError("Dataset has no groups defined")
        
        test_mask = self.groups == test_group
        train_mask = ~test_mask
        
        train_ds = DecodingDataset(
            X=self.X[train_mask],
            y=self.y[train_mask],
            groups=self.groups[train_mask],
            feature_names=self.feature_names,
            class_names=self.class_names,
            metadata=self.metadata,
            modality=self.modality
        )
        
        test_ds = DecodingDataset(
            X=self.X[test_mask],
            y=self.y[test_mask],
            groups=self.groups[test_mask],
            feature_names=self.feature_names,
            class_names=self.class_names,
            metadata=self.metadata,
            modality=self.modality
        )
        
        return train_ds, test_ds
    
    def to_sklearn(self) -> tuple:
        """
        Get data in scikit-learn format.
        
        Returns:
            Tuple of (X, y, groups)
        """
        return self.X, self.y, self.groups
    
    def summary(self) -> str:
        """Generate text summary of dataset."""
        lines = [
            "DecodingDataset Summary",
            "=" * 40,
            f"Modality: {self.modality}",
            f"Samples: {self.n_samples}",
            f"Features: {self.n_features}",
            f"Classes: {self.n_classes}",
            f"Groups: {self.n_groups}",
            f"Balanced: {'Yes' if self.is_balanced else 'No'}",
            "",
            "Class distribution:",
        ]
        
        for name, count in self.class_counts.items():
            pct = count / self.n_samples * 100
            lines.append(f"  {name}: {count} ({pct:.1f}%)")
        
        if self.metadata:
            lines.append("")
            lines.append("Metadata:")
            for key, value in self.metadata.items():
                lines.append(f"  {key}: {value}")
        
        return "\n".join(lines)
    
    def save(self, path: str):
        """
        Save dataset to npz file.
        
        Args:
            path: Output path (.npz)
        """
        path = Path(path)
        
        save_dict = {
            "X": self.X,
            "y": self.y,
            "modality": self.modality
        }
        
        if self.groups is not None:
            save_dict["groups"] = self.groups
        
        np.savez(path, **save_dict)
        
        # Save metadata separately as JSON
        meta_path = path.with_suffix(".json")
        meta = {
            "feature_names": self.feature_names,
            "class_names": self.class_names,
            "metadata": self.metadata,
            "modality": self.modality
        }
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> "DecodingDataset":
        """
        Load dataset from npz file.
        
        Args:
            path: Path to .npz file
            
        Returns:
            DecodingDataset instance
        """
        path = Path(path)
        data = np.load(path)
        
        # Load metadata
        meta_path = path.with_suffix(".json")
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
        else:
            meta = {}
        
        return cls(
            X=data["X"],
            y=data["y"],
            groups=data.get("groups"),
            feature_names=meta.get("feature_names"),
            class_names=meta.get("class_names"),
            metadata=meta.get("metadata", {}),
            modality=meta.get("modality", str(data.get("modality", "unknown")))
        )
    
    def __repr__(self) -> str:
        return (
            f"DecodingDataset(n_samples={self.n_samples}, "
            f"n_features={self.n_features}, n_classes={self.n_classes}, "
            f"modality='{self.modality}')"
        )
    
    def __len__(self) -> int:
        return self.n_samples
