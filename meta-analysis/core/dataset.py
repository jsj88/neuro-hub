"""
Meta-analysis dataset container.

This module provides the MetaAnalysisDataset class which holds
multiple studies and converts to formats required by NiMARE and PyMARE.
"""

from typing import List, Dict, Any, Optional
from pathlib import Path
import json
import pandas as pd

from .study import Study, Coordinate, EffectSize, CoordinateSpace


class MetaAnalysisDataset:
    """
    Unified dataset container for meta-analysis.

    This class holds multiple Study objects and provides conversion
    methods for both coordinate-based (NiMARE) and effect-size (PyMARE)
    meta-analysis.

    Attributes:
        name: Dataset name
        description: Dataset description
        studies: List of Study objects
    """

    def __init__(self, name: str, description: str = ""):
        """
        Initialize a new meta-analysis dataset.

        Args:
            name: Name for this dataset
            description: Optional description
        """
        self.name = name
        self.description = description
        self.studies: List[Study] = []
        self._nimare_dataset = None  # Cached NiMARE dataset

    def add_study(self, study: Study) -> None:
        """
        Add a study to the dataset.

        Args:
            study: Study object to add
        """
        self.studies.append(study)
        self._nimare_dataset = None  # Invalidate cache

    def add_studies(self, studies: List[Study]) -> None:
        """
        Add multiple studies to the dataset.

        Args:
            studies: List of Study objects to add
        """
        self.studies.extend(studies)
        self._nimare_dataset = None

    def remove_study(self, study_id: str) -> bool:
        """
        Remove a study by ID.

        Args:
            study_id: ID of study to remove

        Returns:
            True if study was found and removed
        """
        for i, study in enumerate(self.studies):
            if study.study_id == study_id:
                del self.studies[i]
                self._nimare_dataset = None
                return True
        return False

    def get_study(self, study_id: str) -> Optional[Study]:
        """Get a study by ID."""
        for study in self.studies:
            if study.study_id == study_id:
                return study
        return None

    # =========================================================================
    # Coordinate-Based Meta-Analysis (CBMA) Methods
    # =========================================================================

    def to_coordinates_df(self) -> pd.DataFrame:
        """
        Export coordinates in NiMARE-compatible DataFrame format.

        Returns:
            DataFrame with columns: study_id, contrast_id, x, y, z, space, n
        """
        records = []
        for study in self.studies:
            for i, coord in enumerate(study.coordinates):
                records.append({
                    "study_id": study.study_id,
                    "contrast_id": f"{study.study_id}_1",
                    "x": coord.x,
                    "y": coord.y,
                    "z": coord.z,
                    "space": coord.space.value,
                    "n": study.n_total or 20,  # Default sample size
                    "region": coord.region,
                    "cluster_size": coord.cluster_size,
                    "statistic_value": coord.statistic_value,
                    "statistic_type": coord.statistic_type
                })
        return pd.DataFrame(records)

    def to_nimare_dict(self) -> Dict[str, Any]:
        """
        Convert to NiMARE-compatible dictionary format.

        This creates the nested dictionary structure that NiMARE
        expects for creating a Dataset object.

        Returns:
            Dictionary in NiMARE format
        """
        nimare_dict = {}

        for study in self.studies:
            if not study.has_coordinates:
                continue

            study_dict = {
                "contrasts": {
                    "1": {
                        "coords": {
                            "x": [c.x for c in study.coordinates],
                            "y": [c.y for c in study.coordinates],
                            "z": [c.z for c in study.coordinates],
                            "space": study.coordinates[0].space.value if study.coordinates else "mni152"
                        },
                        "metadata": {
                            "sample_sizes": [study.n_total or 20]
                        }
                    }
                },
                "metadata": {
                    "authors": ", ".join(study.authors),
                    "title": study.title,
                    "year": study.year
                }
            }

            if study.contrast_name:
                study_dict["contrasts"]["1"]["metadata"]["contrast"] = study.contrast_name

            nimare_dict[study.study_id] = study_dict

        return nimare_dict

    def to_nimare_dataset(self):
        """
        Convert to NiMARE Dataset object.

        Returns:
            nimare.dataset.Dataset object for use with ALE, MKDA, etc.

        Raises:
            ImportError: If NiMARE is not installed
        """
        if self._nimare_dataset is not None:
            return self._nimare_dataset

        try:
            from nimare.dataset import Dataset
            from nimare.utils import tal2mni
        except ImportError:
            raise ImportError(
                "NiMARE is required for coordinate-based meta-analysis. "
                "Install with: pip install nimare"
            )

        # Build coordinate DataFrame
        coords_df = self.to_coordinates_df()

        # Convert Talairach to MNI if needed
        tal_mask = coords_df["space"] == "talairach"
        if tal_mask.any():
            tal_coords = coords_df.loc[tal_mask, ["x", "y", "z"]].values
            mni_coords = tal2mni(tal_coords)
            coords_df.loc[tal_mask, ["x", "y", "z"]] = mni_coords
            coords_df.loc[tal_mask, "space"] = "mni152"

        # Create NiMARE dataset from dictionary
        self._nimare_dataset = Dataset(self.to_nimare_dict())

        return self._nimare_dataset

    # =========================================================================
    # Effect Size Meta-Analysis (ESMA) Methods
    # =========================================================================

    def to_effect_sizes_df(self) -> pd.DataFrame:
        """
        Export effect sizes for PyMARE/PythonMeta.

        Returns:
            DataFrame with columns: study_id, year, n, effect_size, variance, etc.
        """
        records = []
        for study in self.studies:
            for es in study.effect_sizes:
                records.append({
                    "study_id": study.study_id,
                    "citation": study.citation,
                    "year": study.year,
                    "n": study.n_total,
                    "n1": study.n_experimental,
                    "n2": study.n_control,
                    "effect_size": es.value,
                    "variance": es.variance,
                    "se": es.se,
                    "ci_lower": es.ci_lower,
                    "ci_upper": es.ci_upper,
                    "effect_type": es.effect_type,
                    "outcome_name": es.outcome_name
                })
        return pd.DataFrame(records)

    def to_pymare_dataset(self, outcome_filter: Optional[str] = None):
        """
        Convert to PyMARE Dataset object.

        Args:
            outcome_filter: Only include effect sizes with this outcome name

        Returns:
            pymare.Dataset object for use with estimators

        Raises:
            ImportError: If PyMARE is not installed
        """
        try:
            from pymare import Dataset as PyMAREDataset
        except ImportError:
            raise ImportError(
                "PyMARE is required for effect size meta-analysis. "
                "Install with: pip install pymare"
            )

        df = self.to_effect_sizes_df()

        if outcome_filter:
            df = df[df["outcome_name"] == outcome_filter]

        if df.empty:
            raise ValueError("No effect sizes found in dataset")

        y = df["effect_size"].values
        v = df["variance"].values

        # Handle missing variance
        if pd.isna(v).any():
            raise ValueError(
                "Some effect sizes have missing variance. "
                "Use EffectSize.from_means() or provide variance directly."
            )

        return PyMAREDataset(y=y, v=v, X=None, n=df["n"].values)

    # =========================================================================
    # Summary Statistics
    # =========================================================================

    @property
    def n_studies(self) -> int:
        """Total number of studies."""
        return len(self.studies)

    @property
    def n_coordinates(self) -> int:
        """Total number of brain coordinates."""
        return sum(len(s.coordinates) for s in self.studies)

    @property
    def n_effect_sizes(self) -> int:
        """Total number of effect sizes."""
        return sum(len(s.effect_sizes) for s in self.studies)

    @property
    def studies_with_coordinates(self) -> List[Study]:
        """Studies that have coordinate data."""
        return [s for s in self.studies if s.has_coordinates]

    @property
    def studies_with_effect_sizes(self) -> List[Study]:
        """Studies that have effect size data."""
        return [s for s in self.studies if s.has_effect_sizes]

    @property
    def total_sample_size(self) -> int:
        """Sum of sample sizes across all studies."""
        return sum(s.n_total or 0 for s in self.studies)

    @property
    def year_range(self) -> tuple:
        """Range of publication years."""
        years = [s.year for s in self.studies]
        if not years:
            return (None, None)
        return (min(years), max(years))

    def summary(self) -> str:
        """Generate a text summary of the dataset."""
        lines = [
            f"Meta-Analysis Dataset: {self.name}",
            f"{'=' * 50}",
            f"Description: {self.description}",
            f"",
            f"Studies: {self.n_studies}",
            f"  - With coordinates: {len(self.studies_with_coordinates)}",
            f"  - With effect sizes: {len(self.studies_with_effect_sizes)}",
            f"",
            f"Total coordinates: {self.n_coordinates}",
            f"Total effect sizes: {self.n_effect_sizes}",
            f"Total sample size: {self.total_sample_size}",
            f"Year range: {self.year_range[0]} - {self.year_range[1]}"
        ]
        return "\n".join(lines)

    # =========================================================================
    # Serialization
    # =========================================================================

    def to_dict(self) -> Dict[str, Any]:
        """Convert dataset to dictionary for serialization."""
        return {
            "name": self.name,
            "description": self.description,
            "studies": [s.to_dict() for s in self.studies]
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MetaAnalysisDataset":
        """Create dataset from dictionary."""
        dataset = cls(
            name=data["name"],
            description=data.get("description", "")
        )
        for study_data in data.get("studies", []):
            dataset.add_study(Study.from_dict(study_data))
        return dataset

    def save(self, path: str) -> None:
        """
        Save dataset to JSON file.

        Args:
            path: Path to save file
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "MetaAnalysisDataset":
        """
        Load dataset from JSON file.

        Args:
            path: Path to JSON file

        Returns:
            Loaded MetaAnalysisDataset
        """
        with open(path) as f:
            data = json.load(f)
        return cls.from_dict(data)

    def to_csv(self, output_dir: str) -> Dict[str, str]:
        """
        Export dataset to CSV files.

        Args:
            output_dir: Directory to save CSV files

        Returns:
            Dictionary mapping data type to file path
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        files = {}

        # Studies metadata
        studies_df = pd.DataFrame([{
            "study_id": s.study_id,
            "title": s.title,
            "authors": "; ".join(s.authors),
            "year": s.year,
            "doi": s.doi,
            "pmid": s.pmid,
            "n_total": s.n_total,
            "n_coordinates": len(s.coordinates),
            "n_effect_sizes": len(s.effect_sizes)
        } for s in self.studies])
        studies_path = output_dir / "studies.csv"
        studies_df.to_csv(studies_path, index=False)
        files["studies"] = str(studies_path)

        # Coordinates
        coords_df = self.to_coordinates_df()
        if not coords_df.empty:
            coords_path = output_dir / "coordinates.csv"
            coords_df.to_csv(coords_path, index=False)
            files["coordinates"] = str(coords_path)

        # Effect sizes
        es_df = self.to_effect_sizes_df()
        if not es_df.empty:
            es_path = output_dir / "effect_sizes.csv"
            es_df.to_csv(es_path, index=False)
            files["effect_sizes"] = str(es_path)

        return files
