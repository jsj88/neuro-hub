# Neuroscience Research Hub

A repository for collecting neuroscience analysis methods, paper notes, and code from the literature.

## Structure

```
neuro-hub/
├── methods/           # Analysis pipelines organized by technique
│   ├── behavior/      # Behavioral paradigms (T-maze, open field, fear conditioning)
│   ├── imaging/       # Calcium imaging, fMRI, optical methods
│   └── ephys/         # Electrophysiology, spike sorting, LFP analysis
├── meta-analysis/     # AI-powered meta-analysis toolkit (NEW)
│   ├── core/          # Data models (Study, Coordinate, EffectSize)
│   ├── extraction/    # AI-powered data extraction
│   ├── analysis/      # NiMARE (ALE) and PyMARE wrappers
│   └── notebooks/     # Jupyter workflow templates
├── papers/            # Paper notes and extracted methods
├── scripts/           # Reusable analysis code
└── resources/         # Reference materials and curated lists
```

## Usage with Claude Code

Use the `/neuro` command to start a research session:
- Searches Google Scholar via Rutgers library access
- Extracts methods and code from papers
- Organizes findings in this repository
- Logs progress to Notion

## Methods Directory

Each subdirectory contains:
- `README.md` - Overview of techniques in this category
- Individual method files with extracted procedures
- Links to source papers and code repositories

## Papers Directory

Paper notes follow this format:
- Citation and DOI
- Key findings summary
- Methods extracted
- Code/data availability
- Relevance to current research

## Adding Content

1. Use `/neuro` to search and access papers
2. Extract relevant methods to appropriate `methods/` subdirectory
3. Save paper notes to `papers/`
4. Clone or save code to `scripts/`

## Meta-Analysis Toolkit

AI-powered meta-analysis for neuroscience research.

### Features
- **AI Paper Screening**: LLM screens abstracts using inclusion/exclusion criteria
- **AI Data Extraction**: Extract brain coordinates and effect sizes from papers
- **Coordinate-Based Meta-Analysis**: ALE/MKDA via NiMARE
- **Effect Size Meta-Analysis**: Random effects models via PyMARE
- **Visualization**: Brain maps, forest plots, funnel plots

### Quick Start

```bash
cd meta-analysis
pip install -r requirements.txt

# In Python
from core import Study, Coordinate, MetaAnalysisDataset
from analysis.coordinate_based import ALEAnalysis
```

See `meta-analysis/README.md` for full documentation.
