"""
DecodingConfig - Configuration settings for neural decoding.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any
import os
import json
from pathlib import Path


@dataclass
class ClassifierConfig:
    """Configuration for classifiers."""
    
    # SVM defaults
    svm_kernel: str = "linear"
    svm_C: float = 1.0
    
    # Random Forest defaults
    rf_n_estimators: int = 100
    rf_max_depth: Optional[int] = None
    
    # Logistic Regression defaults
    lr_C: float = 1.0
    lr_max_iter: int = 1000


@dataclass
class CrossValidationConfig:
    """Configuration for cross-validation."""
    
    n_folds: int = 5
    stratified: bool = True
    shuffle: bool = False
    random_state: Optional[int] = 42


@dataclass
class PermutationConfig:
    """Configuration for permutation testing."""
    
    n_permutations: int = 1000
    random_state: Optional[int] = 42


@dataclass
class SearchlightConfig:
    """Configuration for searchlight analysis."""
    
    radius: float = 5.0  # mm
    n_jobs: int = -1  # All CPUs
    verbose: int = 1


@dataclass
class DecodingConfig:
    """
    Global configuration for neural decoding analyses.
    
    Example:
        >>> config = DecodingConfig.from_env()
        >>> config.classifier.svm_C = 0.1
        >>> config.save("my_config.json")
    """
    
    classifier: ClassifierConfig = field(default_factory=ClassifierConfig)
    cv: CrossValidationConfig = field(default_factory=CrossValidationConfig)
    permutation: PermutationConfig = field(default_factory=PermutationConfig)
    searchlight: SearchlightConfig = field(default_factory=SearchlightConfig)
    
    # Paths
    data_dir: str = "data"
    results_dir: str = "results"
    figures_dir: str = "figures"
    
    # General
    random_state: Optional[int] = 42
    n_jobs: int = -1
    verbose: bool = True
    
    @classmethod
    def from_env(cls) -> "DecodingConfig":
        """Load configuration from environment variables."""
        return cls(
            data_dir=os.getenv("DECODING_DATA_DIR", "data"),
            results_dir=os.getenv("DECODING_RESULTS_DIR", "results"),
            figures_dir=os.getenv("DECODING_FIGURES_DIR", "figures"),
            n_jobs=int(os.getenv("DECODING_N_JOBS", "-1")),
            random_state=int(os.getenv("DECODING_RANDOM_STATE", "42"))
        )
    
    @classmethod
    def from_file(cls, path: str) -> "DecodingConfig":
        """Load configuration from JSON file."""
        with open(path) as f:
            data = json.load(f)
        
        config = cls()
        
        if "classifier" in data:
            config.classifier = ClassifierConfig(**data["classifier"])
        if "cv" in data:
            config.cv = CrossValidationConfig(**data["cv"])
        if "permutation" in data:
            config.permutation = PermutationConfig(**data["permutation"])
        if "searchlight" in data:
            config.searchlight = SearchlightConfig(**data["searchlight"])
        
        for key in ["data_dir", "results_dir", "figures_dir", 
                    "random_state", "n_jobs", "verbose"]:
            if key in data:
                setattr(config, key, data[key])
        
        return config
    
    def save(self, path: str):
        """Save configuration to JSON file."""
        from dataclasses import asdict
        
        data = {
            "classifier": asdict(self.classifier),
            "cv": asdict(self.cv),
            "permutation": asdict(self.permutation),
            "searchlight": asdict(self.searchlight),
            "data_dir": self.data_dir,
            "results_dir": self.results_dir,
            "figures_dir": self.figures_dir,
            "random_state": self.random_state,
            "n_jobs": self.n_jobs,
            "verbose": self.verbose
        }
        
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
    
    def ensure_dirs(self):
        """Create output directories."""
        for d in [self.data_dir, self.results_dir, self.figures_dir]:
            Path(d).mkdir(parents=True, exist_ok=True)


# Global config instance
_config: Optional[DecodingConfig] = None


def get_config() -> DecodingConfig:
    """Get global configuration instance."""
    global _config
    if _config is None:
        _config = DecodingConfig.from_env()
    return _config


def set_config(config: DecodingConfig):
    """Set global configuration instance."""
    global _config
    _config = config
