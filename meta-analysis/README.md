# AI-Powered Meta-Analysis Toolkit

A Python toolkit for conducting neuroscience meta-analyses with AI-assisted data extraction.

## Features

- **AI Paper Screening**: LLM-based abstract screening with dual-review simulation
- **AI Data Extraction**: Extract brain coordinates and effect sizes from papers using Claude/GPT
- **Coordinate-Based Meta-Analysis**: ALE and MKDA via NiMARE
- **Effect Size Meta-Analysis**: Random effects models via PyMARE
- **Visualization**: Brain maps, forest plots, funnel plots

## Quick Start

### Installation

```bash
cd ~/neuro-hub/meta-analysis
pip install -r requirements.txt
```

### Set up API keys

```bash
export ANTHROPIC_API_KEY="your-key-here"
```

### Basic Usage

```python
from core import Study, Coordinate, EffectSize, MetaAnalysisDataset

# Create a study with brain coordinates
study = Study(
    study_id="smith2020",
    title="T-maze navigation in humans",
    authors=["Smith, J.", "Jones, M."],
    year=2020,
    n_total=30,
    coordinates=[
        Coordinate(x=-24, y=-8, z=52),  # Left motor cortex
        Coordinate(x=28, y=-10, z=48),  # Right motor cortex
    ]
)

# Create a dataset
dataset = MetaAnalysisDataset(name="T-Maze Studies")
dataset.add_study(study)

# Run ALE meta-analysis
from analysis.coordinate_based import ALEAnalysis
ale = ALEAnalysis(dataset)
results = ale.run()
```

## Directory Structure

```
meta-analysis/
├── core/                    # Data models (Study, Coordinate, EffectSize)
├── extraction/              # AI-powered data extraction
├── analysis/                # Meta-analysis engines
│   ├── coordinate_based/    # NiMARE (ALE, MKDA)
│   └── effect_size/         # PyMARE (random effects)
├── visualization/           # Brain maps, forest plots
├── ingestion/               # Paper search and screening
└── notebooks/               # Jupyter workflow templates
```

## Notebooks

1. `01_search_and_screen.ipynb` - Find and screen papers
2. `02_data_extraction.ipynb` - Extract coordinates/effect sizes
3. `03_coordinate_meta_analysis.ipynb` - Run ALE analysis
4. `04_effect_size_meta_analysis.ipynb` - Run effect size analysis

## Supported Analysis Types

### Coordinate-Based Meta-Analysis (CBMA)
- **ALE** (Activation Likelihood Estimation)
- **MKDA** (Multilevel Kernel Density Analysis)
- Supports MNI and Talairach coordinates

### Effect Size Meta-Analysis
- Cohen's d, Hedges' g
- Correlation coefficients (r)
- Odds ratios, risk ratios
- DerSimonian-Laird random effects model

## Dependencies

- `nimare` - Neuroimaging meta-analysis
- `pymare` - Effect size meta-analysis
- `nilearn` - Brain visualization
- `anthropic` - Claude API for AI extraction
- `pandas`, `numpy`, `scipy` - Data processing
