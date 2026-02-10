# AI-Powered Meta-Analysis Toolkit

A Python toolkit for conducting neuroimaging and behavioral meta-analyses with AI-assisted data extraction.

## Features

- **Multi-Engine Paper Search**: PubMed, Google Scholar, Semantic Scholar, Scopus
- **AI-Powered Screening**: Automated abstract screening with inclusion/exclusion criteria
- **AI Data Extraction**: Extract brain coordinates and effect sizes from paper text
- **Coordinate-Based Meta-Analysis**: ALE analysis via NiMARE
- **Effect Size Meta-Analysis**: Random effects models via PyMARE
- **Publication-Ready Outputs**: Brain maps, forest plots, PRISMA data

---

## Quick Start

```bash
# Clone repository
git clone https://github.com/jsj88/neuro-hub.git
cd neuro-hub/meta-analysis

# Install dependencies
pip install nimare pymare nilearn anthropic pandas numpy scipy matplotlib
pip install biopython scholarly requests

# Set API key for AI extraction
export ANTHROPIC_API_KEY="your-key"

# Launch notebooks
cd notebooks
jupyter notebook
```

---

## Tutorial: Complete Meta-Analysis Workflow

### Prerequisites

#### 1. Install Dependencies
```bash
pip install -r requirements.txt

# Core dependencies
pip install nimare pymare nilearn anthropic pandas numpy scipy matplotlib

# Search engine dependencies
pip install biopython              # PubMed
pip install scholarly              # Google Scholar (free)
pip install google-search-results  # Google Scholar via SerpAPI (paid)
pip install requests               # Semantic Scholar, Scopus
```

#### 2. Set Up API Keys
```bash
# Required for AI extraction
export ANTHROPIC_API_KEY="your-claude-api-key"

# Optional: for search engines
export PUBMED_EMAIL="your-email@example.com"
export SERPAPI_KEY="your-serpapi-key"  # For Google Scholar
```

---

### Step 1: Search & Screen Papers

Open `notebooks/01_search_and_screen.ipynb`

```python
# Configure credentials
credentials = SearchCredentials(
    pubmed_email="your-email@example.com",
    proxy_username="jss388",   # Your institutional ID
)

# Define research question
RESEARCH_QUESTION = "What brain regions are activated during spatial decision-making?"

# Run search
papers = search_manager.search(
    query='("spatial navigation" OR "T-maze") AND fMRI',
    engines=["PubMed", "Semantic Scholar"],
    max_results_per_engine=100
)
```

---

### Step 2: Extract Data from Papers

Open `notebooks/02_data_extraction.ipynb`

```python
from core import Study, Coordinate, MetaAnalysisDataset

dataset = MetaAnalysisDataset(name="My Meta-Analysis")

study = Study(
    study_id="smith2020",
    title="Neural correlates of navigation",
    authors=["Smith, J."],
    year=2020,
    n_total=30,
    coordinates=[
        Coordinate(x=-24, y=-8, z=52, region="Left premotor"),
        Coordinate(x=28, y=-18, z=-12, region="Right hippocampus"),
    ]
)
dataset.add_study(study)
dataset.save("data/my_dataset.json")
```

---

### Step 3A: Coordinate-Based Meta-Analysis (ALE)

Open `notebooks/03_coordinate_meta_analysis.ipynb`

```python
from analysis.coordinate_based import ALEAnalysis

ale = ALEAnalysis(dataset)
results = ale.run(n_iters=10000, correction_method="fwe")
ale.plot_results(display_mode="glass")
ale.save_nifti("results/ale_zmap.nii.gz")
```

---

### Step 3B: Effect Size Meta-Analysis

Open `notebooks/04_effect_size_meta_analysis.ipynb`

```python
from analysis.effect_size import EffectSizeMetaAnalysis

ma = EffectSizeMetaAnalysis(dataset)
results = ma.run(method="DL")

print(f"Combined effect: {results['combined_effect']:.3f}")
print(f"I-squared = {results['i_squared']:.1f}%")

ma.forest_plot(title="Forest Plot")
```

---

## Directory Structure

```
meta-analysis/
├── README.md
├── requirements.txt
├── core/                     # Data models
│   ├── study.py
│   └── dataset.py
├── extraction/               # AI extractors
│   └── extractors/
├── analysis/                 # Meta-analysis engines
│   ├── coordinate_based/     # ALE via NiMARE
│   └── effect_size/          # PyMARE wrapper
├── ingestion/                # Paper ingestion
│   └── screeners/            # AI screening module
├── notebooks/                # Jupyter workflows
├── data/                     # Your datasets
└── results/                  # Output files
```

---

## Search Engines

| Engine | Auth Required | Notes |
|--------|---------------|-------|
| PubMed | Email only | Free, via Biopython |
| Google Scholar | SerpAPI key OR none | SerpAPI (paid) or scholarly (free) |
| Semantic Scholar | Optional API key | Free, high quality |
| Scopus | Institutional key | Elsevier access |

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| ImportError: nimare | pip install nimare |
| ImportError: pymare | pip install pymare |
| API key errors | Set ANTHROPIC_API_KEY |
| ALE fails | Need 2+ studies with coordinates |

---

## Citation

```
Neuro-Hub Meta-Analysis Toolkit
https://github.com/jsj88/neuro-hub
```

Tools used:
- NiMARE: https://nimare.readthedocs.io/
- PyMARE: https://pymare.readthedocs.io/
