"""
AI-powered brain coordinate extraction from neuroimaging papers.

This module uses LLMs to extract activation coordinates from
fMRI/PET papers for coordinate-based meta-analysis.
"""

from typing import List, Dict, Any, Optional
from .base_extractor import BaseExtractor, LLMProvider
from ...core.study import Coordinate, CoordinateSpace


class CoordinateExtractor(BaseExtractor):
    """
    Extract brain activation coordinates from neuroimaging papers.

    Uses LLM to identify and extract x, y, z coordinates along with
    associated metadata like brain region, cluster size, and statistics.
    """

    EXTRACTION_PROMPT = """You are an expert neuroscientist extracting brain activation coordinates from a neuroimaging paper.

TASK: Extract ALL brain activation coordinates reported in this paper.

For EACH coordinate, identify:
1. x, y, z values in millimeters
2. Coordinate space (MNI152, MNI, Talairach, or unknown)
3. Brain region name (if mentioned)
4. Statistical value (z-score, t-value, F-value)
5. Statistical type ("z", "t", "F", or other)
6. Cluster size in voxels (if reported)

IMPORTANT:
- Look in tables, results sections, and supplementary materials
- Coordinates are typically reported as triplets like (-24, -8, 52) or "x=-24, y=-8, z=52"
- MNI coordinates typically range: x(-90 to 90), y(-126 to 90), z(-72 to 108)
- If space is not explicitly stated but values are in typical MNI range, assume MNI
- Extract ALL coordinates, even from different contrasts/conditions

Paper text:
{text}

Respond with a JSON object containing:
{{
    "coordinates": [
        {{
            "x": <number>,
            "y": <number>,
            "z": <number>,
            "space": "mni152" | "talairach" | "unknown",
            "region": "<brain region or null>",
            "statistic_value": <number or null>,
            "statistic_type": "z" | "t" | "F" | null,
            "cluster_size": <number or null>
        }}
    ],
    "contrast_name": "<name of contrast/condition or null>",
    "notes": "<any important notes about the extraction>"
}}

If no coordinates are found, return: {{"coordinates": [], "notes": "No coordinates found"}}"""

    def __init__(
        self,
        llm_provider: Optional[LLMProvider] = None,
        validate_bounds: bool = True
    ):
        """
        Initialize the coordinate extractor.

        Args:
            llm_provider: LLM provider for extraction
            validate_bounds: Whether to validate MNI coordinate bounds
        """
        super().__init__(llm_provider)
        self.validate_bounds = validate_bounds

    def extract(
        self,
        text: str,
        contrast_name: Optional[str] = None,
        **kwargs
    ) -> List[Coordinate]:
        """
        Extract brain coordinates from paper text.

        Args:
            text: Full text or results section of paper
            contrast_name: Optional contrast name to focus on
            **kwargs: Additional parameters

        Returns:
            List of Coordinate objects
        """
        # Call LLM for extraction
        result = self.llm.extract(
            text=text,
            prompt_template=self.EXTRACTION_PROMPT,
            temperature=0.0
        )

        # Parse coordinates from response
        coordinates = []
        raw_coords = result.get("coordinates", [])

        for raw in raw_coords:
            try:
                coord = Coordinate(
                    x=float(raw["x"]),
                    y=float(raw["y"]),
                    z=float(raw["z"]),
                    space=self._parse_space(raw.get("space", "unknown")),
                    region=raw.get("region"),
                    cluster_size=raw.get("cluster_size"),
                    statistic_value=raw.get("statistic_value"),
                    statistic_type=raw.get("statistic_type")
                )

                # Optionally validate bounds
                if self.validate_bounds:
                    if coord.space == CoordinateSpace.MNI and coord.is_valid_mni():
                        coordinates.append(coord)
                    elif coord.space != CoordinateSpace.MNI:
                        coordinates.append(coord)
                    # Skip invalid MNI coordinates
                else:
                    coordinates.append(coord)

            except (KeyError, ValueError, TypeError) as e:
                # Skip malformed coordinates
                continue

        return coordinates

    def _parse_space(self, space_str: str) -> CoordinateSpace:
        """Parse coordinate space from string."""
        space_str = space_str.lower().strip()

        if "mni" in space_str:
            return CoordinateSpace.MNI
        elif "tal" in space_str:
            return CoordinateSpace.TALAIRACH
        else:
            return CoordinateSpace.UNKNOWN

    def validate(self, coordinates: List[Coordinate]) -> tuple:
        """
        Validate extracted coordinates.

        Args:
            coordinates: List of extracted Coordinate objects

        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []

        if not coordinates:
            errors.append("No coordinates extracted")
            return False, errors

        for i, coord in enumerate(coordinates):
            # Check for MNI bounds
            if coord.space == CoordinateSpace.MNI:
                if not coord.is_valid_mni():
                    errors.append(
                        f"Coordinate {i+1} ({coord.x}, {coord.y}, {coord.z}) "
                        f"outside MNI bounds"
                    )

            # Check for suspicious values (likely errors)
            if abs(coord.x) > 100 or abs(coord.y) > 150 or abs(coord.z) > 120:
                errors.append(
                    f"Coordinate {i+1} ({coord.x}, {coord.y}, {coord.z}) "
                    f"has suspicious values"
                )

        return len(errors) == 0, errors

    def extract_from_table(
        self,
        table_text: str,
        has_header: bool = True
    ) -> List[Coordinate]:
        """
        Extract coordinates specifically from a table.

        Args:
            table_text: Text representation of coordinate table
            has_header: Whether table has a header row

        Returns:
            List of Coordinate objects
        """
        table_prompt = """Extract brain coordinates from this table.

Table:
{text}

Return JSON with coordinates array. Each coordinate should have x, y, z values.
Look for columns like "x", "y", "z", "MNI coordinates", "Peak coordinates", etc.
"""
        result = self.llm.extract(
            text=table_text,
            prompt_template=table_prompt,
            temperature=0.0
        )

        coordinates = []
        for raw in result.get("coordinates", []):
            try:
                coord = Coordinate(
                    x=float(raw["x"]),
                    y=float(raw["y"]),
                    z=float(raw["z"]),
                    space=self._parse_space(raw.get("space", "mni152")),
                    region=raw.get("region"),
                    cluster_size=raw.get("cluster_size"),
                    statistic_value=raw.get("statistic_value"),
                    statistic_type=raw.get("statistic_type")
                )
                coordinates.append(coord)
            except (KeyError, ValueError, TypeError):
                continue

        return coordinates
