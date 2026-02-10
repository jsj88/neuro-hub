"""
Prompt templates for AI extraction.

These prompts are used by the extractors to guide LLM responses.
"""

COORDINATE_EXTRACTION_PROMPT = '''You are an expert neuroscientist extracting brain coordinates from scientific papers.

Extract ALL activation coordinates from the following text. For each coordinate, identify:
- X, Y, Z coordinates (in mm)
- Coordinate space (MNI or Talairach) - assume MNI if not specified
- Brain region name
- Associated statistic (z-score, t-value, etc.)
- Cluster size if reported

Text to extract from:
{text}

Return as JSON array:
[
    {{
        "x": float,
        "y": float, 
        "z": float,
        "space": "MNI" | "Talairach",
        "region": "region name or null",
        "statistic_value": float or null,
        "statistic_type": "z" | "t" | "F" | null,
        "cluster_size": int or null
    }}
]

If no coordinates found, return empty array: []'''


EFFECT_SIZE_EXTRACTION_PROMPT = '''You are an expert statistician extracting effect sizes from scientific papers.

Extract effect sizes and statistics needed to compute them. Look for:
- Cohen's d, Hedges' g, correlation r, odds ratios
- Means, SDs, and sample sizes for each group
- t-values, F-values, or p-values with sample sizes
- Confidence intervals

Text to extract from:
{text}

Return as JSON array:
[
    {{
        "effect_type": "d" | "g" | "r" | "OR" | "computed",
        "effect_size": float or null,
        "variance": float or null,
        "se": float or null,
        "ci_lower": float or null,
        "ci_upper": float or null,
        "outcome_name": "what was measured",
        "group1_mean": float or null,
        "group1_sd": float or null,
        "group1_n": int or null,
        "group2_mean": float or null,
        "group2_sd": float or null,
        "group2_n": int or null,
        "t_value": float or null,
        "f_value": float or null,
        "p_value": float or null
    }}
]

If no effect sizes found, return empty array: []'''


SCREENING_PROMPT = '''You are an expert systematic reviewer screening abstracts for a meta-analysis.

{criteria_context}

Paper to screen:
Title: {title}
Authors: {authors}
Year: {year}
Abstract: {abstract}

Based ONLY on the abstract, determine:
1. INCLUDE, EXCLUDE, or UNCERTAIN
2. Confidence (0.0-1.0)
3. Which inclusion criteria are met
4. Exclusion reasons if any

Return JSON:
{{
    "decision": "INCLUDE" | "EXCLUDE" | "UNCERTAIN",
    "confidence": 0.0-1.0,
    "reasoning": "brief explanation",
    "criteria_met": ["list"],
    "exclusion_reasons": ["list"]
}}'''


STUDY_METADATA_PROMPT = '''Extract study metadata from this methods section:

{text}

Return JSON:
{{
    "n_total": total sample size (int),
    "n_treatment": treatment group size (int or null),
    "n_control": control group size (int or null),
    "mean_age": mean age in years (float or null),
    "age_sd": age standard deviation (float or null),
    "percent_female": percentage female (float or null),
    "task_name": name of experimental task (string or null),
    "imaging_modality": "fMRI" | "PET" | "EEG" | etc (string or null),
    "scanner_tesla": field strength (float or null),
    "software": analysis software used (string or null)
}}'''
