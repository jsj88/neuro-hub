# T-Maze EEG-fMRI Classification Analysis Toolkit

Consolidated analysis framework for T-maze reward learning paradigm classification.
Supports EEG temporal decoding, fMRI ROI classification, and multimodal EEG-fMRI fusion.

**Based on 5 years of T-maze analysis notebooks.**

## T-Maze Paradigm

The T-maze task has 4 conditions:
- **MazeReward**: Navigation + reward feedback
- **MazeNoReward**: Navigation + no reward
- **NoMazeReward**: No navigation + reward feedback
- **NoMazeNoReward**: No navigation + no reward

### Binary Classification Contrasts:
- **Reward vs No-Reward**: {MazeReward, NoMazeReward} vs {MazeNoReward, NoMazeNoReward}
- **Maze vs No-Maze**: {MazeReward, MazeNoReward} vs {NoMazeReward, NoMazeNoReward}

## Installation

```bash
cd ~/neuro-hub/tmaze-analysis
pip install -r requirements.txt
```

## Directory Structure

```
tmaze-analysis/
├── core/
│   └── containers.py      # TMazeEEGData, TMAzefMRIData, TMazeSubject
├── io/
│   └── loaders.py         # TMazeEEGLoader, TMazefMRILoader
├── classification/
│   ├── classifiers.py     # TMazeClassifier, classify_roi, classify_all_rois
│   ├── temporal.py        # temporal_decoding, temporal_generalization
│   └── multimodal.py      # early_fusion, late_fusion
├── rsa/
│   └── rsa.py             # compute_rdm, compare_rdms, model_rdm_tmaze
├── visualization/
│   └── plots.py           # plot_temporal_decoding, plot_roi_accuracies
├── config/
│   └── settings.py        # TMazeConfig, default parameters
└── notebooks/             # Analysis notebooks
```

## Quick Start

### EEG Temporal Decoding (REWP)

```python
from tmaze_analysis.io import TMazeEEGLoader
from tmaze_analysis.classification import temporal_decoding, find_significant_times
from tmaze_analysis.visualization import plot_temporal_decoding

# Load EEG epochs
loader = TMazeEEGLoader(
    condition_mapping={'MazeReward': 1, 'MazeNoReward': 0,
                       'NoMazeReward': 1, 'NoMazeNoReward': 0}
)
eeg_data = loader.load("sub-01-epo.fif")

# Run temporal decoding
result = temporal_decoding(eeg_data, classifier_type='svm', cv=5)

# Find significant time points
sig_mask, clusters = find_significant_times(result, alpha=0.05)

print(f"Peak decoding: {result.peak_score:.2f} at {result.peak_time*1000:.0f}ms")
print(f"Significant clusters: {clusters}")

# Plot
plot_temporal_decoding(result)
```

### fMRI ROI Classification

```python
from tmaze_analysis.io import TMazefMRILoader
from tmaze_analysis.classification import classify_all_rois, get_top_rois
from tmaze_analysis.visualization import plot_roi_accuracies

# Load fMRI data with HCP atlas
loader = TMazefMRILoader(standardize=True)
fmri_data = loader.load_afni_betas("sub-01/glm/", subject_id="sub-01")

# Classify each ROI
roi_results = classify_all_rois(fmri_data, classifier_type='lda', cv=5)

# Get top performing ROIs
top_rois = get_top_rois(roi_results, n_top=20, min_accuracy=0.55)
for roi_name, result in top_rois:
    print(f"{roi_name}: {result.accuracy:.1%}")

# Plot
plot_roi_accuracies(roi_results, n_top=20)
```

### Multimodal EEG-fMRI Fusion

```python
from tmaze_analysis.io import TMazeSubjectLoader
from tmaze_analysis.classification import multimodal_fusion
from tmaze_analysis.visualization import plot_multimodal_comparison

# Load multimodal subject
loader = TMazeSubjectLoader()
subject = loader.load(
    subject_id="sub-01",
    eeg_path="sub-01-epo.fif",
    fmri_path="sub-01/glm/"
)

# Run fusion analysis
result = multimodal_fusion(
    subject,
    fusion_type='early',
    eeg_feature_type='rewp_mean',
    classifier_type='svm'
)

print(f"EEG only: {result.eeg_only_accuracy:.1%}")
print(f"fMRI only: {result.fmri_only_accuracy:.1%}")
print(f"Fused: {result.accuracy:.1%}")
print(f"Improvement: {result.fusion_improvement:+.1%}")

# Plot
plot_multimodal_comparison(result)
```

### RSA Analysis

```python
from tmaze_analysis.rsa import roi_rsa, rsa_multiple_models, model_rdm_tmaze
from tmaze_analysis.visualization import plot_rdm

# Compare neural patterns to reward model
r, p, neural_rdm = roi_rsa(fmri_data, model_type='reward')
print(f"Correlation with reward model: r={r:.3f}, p={p:.4f}")

# Compare to multiple models
results = rsa_multiple_models(fmri_data, model_types=['reward', 'maze', 'interaction'])
for model, (r, p) in results.items():
    print(f"{model}: r={r:.3f}, p={p:.4f}")

# Plot RDM
plot_rdm(neural_rdm, labels=['MR', 'MNR', 'NMR', 'NMNR'])
```

## Key Modules

### `classification.classifiers`
- `TMazeClassifier`: Main classifier wrapper (LDA, SVM, Logistic, RF)
- `classify_roi()`: Single ROI/feature classification
- `classify_all_rois()`: Per-ROI classification across brain
- `permutation_test()`: Statistical significance testing

### `classification.temporal`
- `temporal_decoding()`: Time-resolved classification (MNE SlidingEstimator)
- `temporal_generalization()`: Train/test generalization matrix
- `rewp_temporal_analysis()`: REWP-focused analysis (FCz, 0-500ms)
- `find_significant_times()`: Cluster-based significance

### `classification.multimodal`
- `early_fusion()`: Feature concatenation
- `late_fusion()`: Probability averaging
- `compare_fusion_methods()`: Compare fusion approaches

### `rsa`
- `compute_rdm()`: Representational Dissimilarity Matrix
- `model_rdm_tmaze()`: Theoretical models (reward, maze, interaction)
- `compare_rdms()`: Spearman correlation between RDMs

## Configuration

```python
from tmaze_analysis.config import get_default_config, TMazeConfig

config = get_default_config()

# Modify settings
config.eeg.rewp_tmin = 0.200
config.eeg.rewp_tmax = 0.400
config.classification.cv_folds = 10
config.classification.n_permutations = 5000
```

## Notebooks

Example analysis notebooks will be added:
1. `01_eeg_temporal_decoding.ipynb` - EEG time-resolved classification
2. `02_fmri_roi_classification.ipynb` - fMRI per-region decoding
3. `03_multimodal_fusion.ipynb` - EEG-fMRI integration
4. `04_rsa_analysis.ipynb` - Representational similarity
5. `05_group_analysis.ipynb` - Multi-subject statistics

## Citation

If using this toolkit, please cite the original T-maze study and analysis methods.

## Author

Jaleesa Stringfellow
