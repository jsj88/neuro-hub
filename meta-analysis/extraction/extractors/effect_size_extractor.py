"""
AI-powered effect size extraction from behavioral/clinical papers.

This module uses LLMs to extract effect sizes, statistics, and
sample information for traditional meta-analysis.
"""

from typing import List, Dict, Any, Optional
from .base_extractor import BaseExtractor, LLMProvider
from ...core.study import EffectSize


class EffectSizeExtractor(BaseExtractor):
    """
    Extract effect sizes and statistics from papers.

    Uses LLM to identify and extract effect sizes (Cohen's d, Hedges' g,
    correlation r, odds ratios) or the statistics needed to compute them.
    """

    EXTRACTION_PROMPT = """You are an expert statistician extracting effect sizes and statistics from a research paper for meta-analysis.

TASK: Extract ALL effect sizes and relevant statistics from this paper.

Look for:
1. Direct effect sizes: Cohen's d, Hedges' g, correlation r, odds ratios (OR), risk ratios (RR)
2. Statistics to compute effect sizes: means, standard deviations, sample sizes, t-values, F-values, p-values
3. Confidence intervals

For EACH outcome measure or comparison, extract:
- Effect size value and type (if directly reported)
- OR the statistics needed to compute it:
  * Group means (M1, M2)
  * Standard deviations (SD1, SD2)
  * Sample sizes (n1, n2)
  * t-statistic or F-statistic
  * Confidence interval bounds
- Outcome measure name
- Comparison description (e.g., "treatment vs control")

Paper text:
{text}

Respond with a JSON object:
{{
    "effect_sizes": [
        {{
            "outcome_name": "<name of outcome measure>",
            "comparison": "<what is being compared>",
            "effect_size": <direct effect size value or null>,
            "effect_type": "d" | "g" | "r" | "or" | "rr" | null,
            "variance": <variance if reported, or null>,
            "se": <standard error if reported, or null>,
            "ci_lower": <95% CI lower bound or null>,
            "ci_upper": <95% CI upper bound or null>,
            "mean1": <experimental group mean or null>,
            "mean2": <control group mean or null>,
            "sd1": <experimental group SD or null>,
            "sd2": <control group SD or null>,
            "n1": <experimental group n or null>,
            "n2": <control group n or null>,
            "t_value": <t-statistic or null>,
            "f_value": <F-statistic or null>,
            "p_value": <p-value or null>
        }}
    ],
    "total_n": <total sample size if reported>,
    "notes": "<any important notes>"
}}

If no effect sizes or statistics found: {{"effect_sizes": [], "notes": "No effect sizes found"}}"""

    def __init__(self, llm_provider: Optional[LLMProvider] = None):
        """Initialize the effect size extractor."""
        super().__init__(llm_provider)

    def extract(
        self,
        text: str,
        outcome_filter: Optional[str] = None,
        **kwargs
    ) -> List[EffectSize]:
        """
        Extract effect sizes from paper text.

        Args:
            text: Full text or results section of paper
            outcome_filter: Optional filter for specific outcome
            **kwargs: Additional parameters

        Returns:
            List of EffectSize objects
        """
        # Call LLM for extraction
        result = self.llm.extract(
            text=text,
            prompt_template=self.EXTRACTION_PROMPT,
            temperature=0.0
        )

        # Parse effect sizes from response
        effect_sizes = []
        raw_effects = result.get("effect_sizes", [])

        for raw in raw_effects:
            try:
                es = self._parse_effect_size(raw)
                if es is not None:
                    # Apply outcome filter if specified
                    if outcome_filter:
                        if es.outcome_name and outcome_filter.lower() in es.outcome_name.lower():
                            effect_sizes.append(es)
                    else:
                        effect_sizes.append(es)
            except (KeyError, ValueError, TypeError) as e:
                continue

        return effect_sizes

    def _parse_effect_size(self, raw: Dict[str, Any]) -> Optional[EffectSize]:
        """
        Parse effect size from raw extraction.

        Handles direct effect sizes or computes from statistics.
        """
        outcome_name = raw.get("outcome_name")
        if raw.get("comparison"):
            outcome_name = f"{outcome_name}: {raw['comparison']}" if outcome_name else raw["comparison"]

        # Case 1: Direct effect size reported
        if raw.get("effect_size") is not None:
            return EffectSize(
                value=float(raw["effect_size"]),
                variance=raw.get("variance"),
                se=raw.get("se"),
                ci_lower=raw.get("ci_lower"),
                ci_upper=raw.get("ci_upper"),
                effect_type=raw.get("effect_type", "d"),
                outcome_name=outcome_name
            )

        # Case 2: Compute from means and SDs
        if all(raw.get(k) is not None for k in ["mean1", "mean2", "sd1", "sd2", "n1", "n2"]):
            return EffectSize.from_means(
                mean1=float(raw["mean1"]),
                mean2=float(raw["mean2"]),
                sd1=float(raw["sd1"]),
                sd2=float(raw["sd2"]),
                n1=int(raw["n1"]),
                n2=int(raw["n2"]),
                outcome_name=outcome_name
            )

        # Case 3: Compute from t-statistic
        if raw.get("t_value") is not None and raw.get("n1") and raw.get("n2"):
            return EffectSize.from_t_statistic(
                t=float(raw["t_value"]),
                n1=int(raw["n1"]),
                n2=int(raw["n2"]),
                outcome_name=outcome_name
            )

        # Case 4: Compute from F-statistic (F = t^2 for two groups)
        if raw.get("f_value") is not None and raw.get("n1") and raw.get("n2"):
            import math
            t = math.sqrt(float(raw["f_value"]))
            return EffectSize.from_t_statistic(
                t=t,
                n1=int(raw["n1"]),
                n2=int(raw["n2"]),
                outcome_name=outcome_name
            )

        # Could not extract or compute effect size
        return None

    def validate(self, effect_sizes: List[EffectSize]) -> tuple:
        """
        Validate extracted effect sizes.

        Args:
            effect_sizes: List of extracted EffectSize objects

        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []

        if not effect_sizes:
            errors.append("No effect sizes extracted")
            return False, errors

        for i, es in enumerate(effect_sizes):
            # Check for missing variance
            if es.variance is None and es.se is None:
                errors.append(
                    f"Effect size {i+1} ({es.outcome_name}) missing variance/SE"
                )

            # Check for implausible effect sizes
            if es.effect_type == "d" and abs(es.value) > 5:
                errors.append(
                    f"Effect size {i+1} (d={es.value}) unusually large"
                )

            if es.effect_type == "r" and abs(es.value) > 1:
                errors.append(
                    f"Correlation {i+1} (r={es.value}) outside valid range"
                )

        # Warnings don't make extraction invalid
        is_valid = not any("missing variance" in e for e in errors)

        return is_valid, errors

    def extract_from_stats_table(
        self,
        table_text: str,
        n1: int,
        n2: int
    ) -> List[EffectSize]:
        """
        Extract effect sizes from a statistics table.

        Args:
            table_text: Text representation of results table
            n1: Sample size of group 1
            n2: Sample size of group 2

        Returns:
            List of EffectSize objects
        """
        table_prompt = """Extract statistics from this results table.

Table:
{text}

For each row, extract: outcome name, mean1, sd1, mean2, sd2, or any effect sizes.
Return as JSON with effect_sizes array.
"""
        result = self.llm.extract(
            text=table_text,
            prompt_template=table_prompt,
            temperature=0.0
        )

        effect_sizes = []
        for raw in result.get("effect_sizes", []):
            # Add sample sizes if not in table
            if raw.get("n1") is None:
                raw["n1"] = n1
            if raw.get("n2") is None:
                raw["n2"] = n2

            es = self._parse_effect_size(raw)
            if es:
                effect_sizes.append(es)

        return effect_sizes
