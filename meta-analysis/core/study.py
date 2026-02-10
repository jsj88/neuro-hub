"""
Core data models for meta-analysis studies.

This module defines the fundamental data structures used throughout
the meta-analysis toolkit.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from enum import Enum
import json


class CoordinateSpace(Enum):
    """Brain coordinate space standards."""
    MNI = "mni152"
    TALAIRACH = "talairach"
    UNKNOWN = "unknown"


@dataclass
class Coordinate:
    """
    Single brain activation coordinate.

    Attributes:
        x: X coordinate in mm
        y: Y coordinate in mm
        z: Z coordinate in mm
        space: Coordinate space (MNI or Talairach)
        region: Brain region name (if identified)
        cluster_size: Number of voxels in cluster
        statistic_value: Statistical value (z, t, F)
        statistic_type: Type of statistic
    """
    x: float
    y: float
    z: float
    space: CoordinateSpace = CoordinateSpace.MNI
    region: Optional[str] = None
    cluster_size: Optional[int] = None
    statistic_value: Optional[float] = None
    statistic_type: Optional[str] = None  # "z", "t", "F"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "x": self.x,
            "y": self.y,
            "z": self.z,
            "space": self.space.value,
            "region": self.region,
            "cluster_size": self.cluster_size,
            "statistic_value": self.statistic_value,
            "statistic_type": self.statistic_type
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Coordinate":
        """Create from dictionary."""
        return cls(
            x=data["x"],
            y=data["y"],
            z=data["z"],
            space=CoordinateSpace(data.get("space", "mni152")),
            region=data.get("region"),
            cluster_size=data.get("cluster_size"),
            statistic_value=data.get("statistic_value"),
            statistic_type=data.get("statistic_type")
        )

    def is_valid_mni(self) -> bool:
        """Check if coordinate is within plausible MNI bounds."""
        # Approximate MNI152 brain bounds
        return (
            -90 <= self.x <= 90 and
            -126 <= self.y <= 90 and
            -72 <= self.z <= 108
        )


@dataclass
class EffectSize:
    """
    Effect size with uncertainty estimates.

    Attributes:
        value: Point estimate of effect size
        variance: Sampling variance
        se: Standard error
        ci_lower: Lower bound of 95% CI
        ci_upper: Upper bound of 95% CI
        effect_type: Type of effect size (d, g, r, or, rr)
        outcome_name: Name of the outcome measure
    """
    value: float
    variance: Optional[float] = None
    se: Optional[float] = None
    ci_lower: Optional[float] = None
    ci_upper: Optional[float] = None
    effect_type: str = "d"  # "d", "g", "r", "or", "rr"
    outcome_name: Optional[str] = None

    def __post_init__(self):
        """Compute missing values if possible."""
        import math

        # Compute SE from variance
        if self.se is None and self.variance is not None:
            self.se = math.sqrt(self.variance)

        # Compute variance from SE
        if self.variance is None and self.se is not None:
            self.variance = self.se ** 2

        # Compute CI from SE
        if self.ci_lower is None and self.se is not None:
            self.ci_lower = self.value - 1.96 * self.se
            self.ci_upper = self.value + 1.96 * self.se

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "value": self.value,
            "variance": self.variance,
            "se": self.se,
            "ci_lower": self.ci_lower,
            "ci_upper": self.ci_upper,
            "effect_type": self.effect_type,
            "outcome_name": self.outcome_name
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EffectSize":
        """Create from dictionary."""
        return cls(
            value=data["value"],
            variance=data.get("variance"),
            se=data.get("se"),
            ci_lower=data.get("ci_lower"),
            ci_upper=data.get("ci_upper"),
            effect_type=data.get("effect_type", "d"),
            outcome_name=data.get("outcome_name")
        )

    @classmethod
    def from_means(
        cls,
        mean1: float,
        mean2: float,
        sd1: float,
        sd2: float,
        n1: int,
        n2: int,
        outcome_name: Optional[str] = None
    ) -> "EffectSize":
        """
        Compute Cohen's d from group means and standard deviations.

        Args:
            mean1: Mean of group 1 (experimental)
            mean2: Mean of group 2 (control)
            sd1: Standard deviation of group 1
            sd2: Standard deviation of group 2
            n1: Sample size of group 1
            n2: Sample size of group 2
            outcome_name: Name of outcome measure

        Returns:
            EffectSize with Cohen's d and variance
        """
        import math

        # Pooled standard deviation
        pooled_sd = math.sqrt(
            ((n1 - 1) * sd1**2 + (n2 - 1) * sd2**2) / (n1 + n2 - 2)
        )

        # Cohen's d
        d = (mean1 - mean2) / pooled_sd

        # Variance of d (Hedges & Olkin, 1985)
        var_d = (n1 + n2) / (n1 * n2) + (d**2) / (2 * (n1 + n2))

        return cls(
            value=d,
            variance=var_d,
            effect_type="d",
            outcome_name=outcome_name
        )

    @classmethod
    def from_t_statistic(
        cls,
        t: float,
        n1: int,
        n2: int,
        outcome_name: Optional[str] = None
    ) -> "EffectSize":
        """
        Compute Cohen's d from t-statistic.

        Args:
            t: t-statistic value
            n1: Sample size of group 1
            n2: Sample size of group 2
            outcome_name: Name of outcome measure

        Returns:
            EffectSize with Cohen's d
        """
        import math

        # Convert t to d
        d = t * math.sqrt((n1 + n2) / (n1 * n2))

        # Variance of d
        var_d = (n1 + n2) / (n1 * n2) + (d**2) / (2 * (n1 + n2))

        return cls(
            value=d,
            variance=var_d,
            effect_type="d",
            outcome_name=outcome_name
        )


@dataclass
class Study:
    """
    Container for all extracted data from a single study.

    This is the central data structure that holds both neuroimaging
    coordinates and behavioral effect sizes from a paper.
    """
    # Identification
    study_id: str
    title: str
    authors: List[str]
    year: int
    doi: Optional[str] = None
    pmid: Optional[str] = None

    # Sample characteristics
    n_total: Optional[int] = None
    n_experimental: Optional[int] = None
    n_control: Optional[int] = None
    mean_age: Optional[float] = None
    age_sd: Optional[float] = None
    percent_female: Optional[float] = None
    population: Optional[str] = None  # "healthy", "clinical", etc.

    # Neuroimaging data
    coordinates: List[Coordinate] = field(default_factory=list)
    contrast_name: Optional[str] = None
    task_name: Optional[str] = None
    imaging_modality: Optional[str] = None  # "fMRI", "PET", etc.

    # Effect size data
    effect_sizes: List[EffectSize] = field(default_factory=list)

    # Extraction metadata
    extraction_confidence: float = 0.0
    extraction_notes: str = ""
    manual_verified: bool = False

    # Additional metadata
    keywords: List[str] = field(default_factory=list)
    abstract: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "study_id": self.study_id,
            "title": self.title,
            "authors": self.authors,
            "year": self.year,
            "doi": self.doi,
            "pmid": self.pmid,
            "n_total": self.n_total,
            "n_experimental": self.n_experimental,
            "n_control": self.n_control,
            "mean_age": self.mean_age,
            "age_sd": self.age_sd,
            "percent_female": self.percent_female,
            "population": self.population,
            "coordinates": [c.to_dict() for c in self.coordinates],
            "contrast_name": self.contrast_name,
            "task_name": self.task_name,
            "imaging_modality": self.imaging_modality,
            "effect_sizes": [e.to_dict() for e in self.effect_sizes],
            "extraction_confidence": self.extraction_confidence,
            "extraction_notes": self.extraction_notes,
            "manual_verified": self.manual_verified,
            "keywords": self.keywords,
            "abstract": self.abstract
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Study":
        """Create Study from dictionary."""
        coordinates = [
            Coordinate.from_dict(c) for c in data.get("coordinates", [])
        ]
        effect_sizes = [
            EffectSize.from_dict(e) for e in data.get("effect_sizes", [])
        ]

        return cls(
            study_id=data["study_id"],
            title=data["title"],
            authors=data["authors"],
            year=data["year"],
            doi=data.get("doi"),
            pmid=data.get("pmid"),
            n_total=data.get("n_total"),
            n_experimental=data.get("n_experimental"),
            n_control=data.get("n_control"),
            mean_age=data.get("mean_age"),
            age_sd=data.get("age_sd"),
            percent_female=data.get("percent_female"),
            population=data.get("population"),
            coordinates=coordinates,
            contrast_name=data.get("contrast_name"),
            task_name=data.get("task_name"),
            imaging_modality=data.get("imaging_modality"),
            effect_sizes=effect_sizes,
            extraction_confidence=data.get("extraction_confidence", 0.0),
            extraction_notes=data.get("extraction_notes", ""),
            manual_verified=data.get("manual_verified", False),
            keywords=data.get("keywords", []),
            abstract=data.get("abstract")
        )

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> "Study":
        """Create Study from JSON string."""
        return cls.from_dict(json.loads(json_str))

    @property
    def has_coordinates(self) -> bool:
        """Check if study has neuroimaging coordinates."""
        return len(self.coordinates) > 0

    @property
    def has_effect_sizes(self) -> bool:
        """Check if study has effect size data."""
        return len(self.effect_sizes) > 0

    @property
    def citation(self) -> str:
        """Generate APA-style citation."""
        if len(self.authors) == 1:
            author_str = self.authors[0]
        elif len(self.authors) == 2:
            author_str = f"{self.authors[0]} & {self.authors[1]}"
        else:
            author_str = f"{self.authors[0]} et al."

        return f"{author_str} ({self.year})"
