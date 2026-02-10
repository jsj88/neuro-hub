# Neural Decoding Toolkit

A Python toolkit for classification-based neural decoding of pre-processed human neuroimaging data (fMRI, EEG) with behavioral integration and multimodal fusion.

## Quick Start

```python
from neural_decoding.io import FMRILoader
from neural_decoding.models import SVMDecoder
from neural_decoding.validation import LeaveOneRunOut

# Load pre-processed data
loader = FMRILoader()
dataset = loader.load(
    data_path="bold.nii.gz",
    mask_path="mask.nii.gz",
    events_path="events.csv",
    label_column="condition"
)

# Classify
decoder = SVMDecoder()
results = decoder.cross_validate(dataset, LeaveOneRunOut())
print(f"Accuracy: {results.accuracy:.1%}")
```

---

## Table of Contents

1. [Installation](#installation)
2. [Supported Data Types](#supported-data-types)
3. [Classification Methods](#classification-methods)
4. [Single-Modality Decoding](#single-modality-decoding)
5. [Multimodal Data Fusion](#multimodal-data-fusion)
6. [Cross-Validation Strategies](#cross-validation-strategies)
7. [Statistical Testing](#statistical-testing)
8. [Visualization](#visualization)
9. [Complete Workflows](#complete-workflows)

---

## Installation

```bash
cd ~/neuro-hub/neural-decoding
pip install -r requirements.txt
```

### Dependencies
- **numpy, pandas, scipy** - Data processing
- **scikit-learn** - Machine learning
- **nilearn, nibabel** - fMRI data handling
- **mne** - EEG/MEG data handling
- **matplotlib, seaborn** - Visualization

---

## Supported Data Types

### 1. fMRI Data
- **Input**: Pre-processed NIfTI files (.nii, .nii.gz)
- **Mask**: Brain mask NIfTI
- **Labels**: Events CSV with trial conditions

```python
from neural_decoding.io import FMRILoader

loader = FMRILoader()
dataset = loader.load(
    data_path="sub-01_task_bold.nii.gz",
    mask_path="brain_mask.nii.gz",
    events_path="events.csv",
    label_column="condition",
    run_column="run"  # For cross-validation
)
```

### 2. EEG/MEG Data
- **Input**: MNE Epochs file (.fif)
- **Labels**: Event IDs from epochs

```python
from neural_decoding.io import EEGLoader

loader = EEGLoader()
dataset = loader.load(
    epochs_path="sub-01-epo.fif",
    time_window=(0.1, 0.5),  # 100-500ms post-stimulus
    flatten=True  # Channels x time -> 1D vector
)
```

### 3. Behavioral Data
- **Input**: CSV file
- **Features**: Any numeric columns

```python
from neural_decoding.io import BehaviorLoader

loader = BehaviorLoader()
dataset = loader.load(
    csv_path="behavior.csv",
    feature_columns=["RT", "accuracy", "confidence"],
    label_column="condition"
)
```

---

## Classification Methods

### Summary Table

| Classifier | Best For | Pros | Cons |
|------------|----------|------|------|
| **Linear SVM** | High-dimensional (fMRI) | Fast, interpretable weights | Linear only |
| **RBF SVM** | Non-linear patterns | Captures complex patterns | Slower, less interpretable |
| **Logistic Regression** | Probabilistic output | Calibrated probabilities | Linear only |
| **Random Forest** | Mixed features | Handles non-linear, feature importance | Can overfit |
| **LDA** | Few samples | Works with small n | Assumes Gaussian |
| **Ensemble** | Best accuracy | Combines strengths | Slower |

---

### 1. Support Vector Machine (SVM)

**Best for**: High-dimensional neuroimaging data (fMRI voxels)

```python
from neural_decoding.models import SVMDecoder

# Linear SVM (recommended for neuroimaging)
decoder = SVMDecoder(kernel="linear", C=1.0)

# RBF SVM (for non-linear patterns)
decoder = SVMDecoder(kernel="rbf", C=1.0, gamma="scale")

# Train and predict
decoder.fit(X_train, y_train)
predictions = decoder.predict(X_test)
accuracy = decoder.score(X_test, y_test)
```

**When to use Linear SVM**:
- fMRI whole-brain decoding (10,000+ voxels)
- When you need interpretable feature weights
- Standard MVPA analyses

**When to use RBF SVM**:
- Lower-dimensional features (ROIs, components)
- Complex, non-linear class boundaries
- Behavioral data with interactions

---

### 2. Logistic Regression

**Best for**: Probabilistic classification, regularized models

```python
from neural_decoding.models import LogisticDecoder

# L2 regularization (Ridge)
decoder = LogisticDecoder(penalty="l2", C=1.0)

# L1 regularization (Lasso - sparse features)
decoder = LogisticDecoder(penalty="l1", C=1.0, solver="saga")

# Elastic Net
decoder = LogisticDecoder(penalty="elasticnet", l1_ratio=0.5, solver="saga")
```

**Advantages**:
- Outputs calibrated probabilities
- L1 penalty provides feature selection
- Good baseline classifier

---

### 3. Random Forest

**Best for**: Feature importance, non-linear patterns

```python
from neural_decoding.models import RandomForestDecoder

decoder = RandomForestDecoder(
    n_estimators=100,
    max_depth=None,  # Full trees
    min_samples_leaf=5,
    n_jobs=-1  # Parallel
)

# Get feature importances after fitting
decoder.fit(X_train, y_train)
importances = decoder.feature_importances_
```

**Advantages**:
- Built-in feature importance
- Handles non-linear relationships
- Robust to outliers

---

### 4. Linear Discriminant Analysis (LDA)

**Best for**: Small sample sizes, dimensionality reduction

```python
from neural_decoding.models import LDADecoder

decoder = LDADecoder(
    solver="svd",  # Singular value decomposition
    shrinkage="auto"  # Regularization for small n
)
```

**When to use**:
- n_samples < n_features (needs regularization)
- When you also want dimensionality reduction
- Gaussian-distributed features

---

### 5. Ensemble Methods

**Best for**: Maximum accuracy, combining diverse models

```python
from neural_decoding.models import EnsembleDecoder, SVMDecoder, RandomForestDecoder, LogisticDecoder

# Voting ensemble
decoder = EnsembleDecoder(
    classifiers=[
        SVMDecoder(kernel="linear"),
        RandomForestDecoder(n_estimators=100),
        LogisticDecoder()
    ],
    voting="soft"  # Average probabilities
)

# Stacking ensemble (uses meta-learner)
decoder = StackingDecoder(
    classifiers=[SVMDecoder(), RandomForestDecoder()],
    meta_classifier=LogisticDecoder()
)
```

---

## Single-Modality Decoding

### fMRI Decoding Workflow

```python
from neural_decoding.io import FMRILoader
from neural_decoding.models import SVMDecoder
from neural_decoding.validation import LeaveOneRunOut, PermutationTest

# 1. Load data
loader = FMRILoader()
dataset = loader.load(
    data_path="sub-01_task_bold.nii.gz",
    mask_path="brain_mask.nii.gz",
    events_path="events.csv",
    label_column="condition",
    run_column="run"
)

print(dataset.summary())

# 2. Initialize decoder
decoder = SVMDecoder(kernel="linear")

# 3. Cross-validate
cv = LeaveOneRunOut()
results = decoder.cross_validate(dataset, cv)

print(f"Accuracy: {results.accuracy:.1%}")
print(f"CV scores: {results.cv_scores}")

# 4. Permutation test
perm_test = PermutationTest(n_permutations=1000)
results.permutation_pvalue = perm_test.test(decoder, dataset, cv)

print(f"P-value: {results.permutation_pvalue:.4f}")

# 5. Visualize
results.plot_confusion_matrix()
results.plot_cv_scores()
```

### EEG Temporal Decoding

```python
from neural_decoding.io import EEGLoader
from neural_decoding.models import TemporalDecoder

# Load epochs
loader = EEGLoader()
dataset = loader.load(
    epochs_path="sub-01-epo.fif",
    time_window=(-0.2, 1.0)
)

# Time-resolved decoding
decoder = TemporalDecoder(
    base_decoder=SVMDecoder(kernel="linear"),
    time_window=0.05,  # 50ms windows
    step=0.01  # 10ms steps
)

decoder.fit(dataset)

# Plot decoding over time
decoder.plot(chance_level=0.5)
```

### Searchlight Analysis (fMRI)

```python
from neural_decoding.models import SearchlightDecoder

searchlight = SearchlightDecoder(
    mask_path="brain_mask.nii.gz",
    radius=5.0,  # 5mm sphere
    decoder=SVMDecoder(kernel="linear"),
    cv=LeaveOneRunOut(),
    n_jobs=-1  # Use all CPUs
)

searchlight.fit(dataset)

# Save accuracy map
searchlight.save_nifti("searchlight_accuracy.nii.gz")

# Visualize
from neural_decoding.visualization import plot_brain_map
plot_brain_map("searchlight_accuracy.nii.gz", threshold=0.6)
```

---

## Multimodal Data Fusion

### Fusion Strategies Overview

| Strategy | Method | When to Use |
|----------|--------|-------------|
| **Early Fusion** | Concatenate features | Complementary modalities, same trials |
| **Late Fusion** | Combine predictions | Different sample rates, independent info |
| **Intermediate Fusion** | Shared representation | Deep learning, learned embeddings |

---

### Early Fusion (Feature Concatenation)

Combine features from multiple modalities before classification.

```python
from neural_decoding.io import FMRILoader, EEGLoader, BehaviorLoader, MultimodalLoader
from neural_decoding.models import SVMDecoder

# Load each modality separately
fmri_loader = FMRILoader()
fmri_data = fmri_loader.load(
    data_path="bold.nii.gz",
    mask_path="mask.nii.gz",
    events_path="events.csv",
    label_column="condition"
)

eeg_loader = EEGLoader()
eeg_data = eeg_loader.load(
    epochs_path="epochs.fif",
    time_window=(0.1, 0.5)
)

behavior_loader = BehaviorLoader()
behavior_data = behavior_loader.load(
    csv_path="behavior.csv",
    feature_columns=["RT", "confidence"],
    label_column="condition"
)

# Fuse modalities (early fusion)
multimodal = MultimodalLoader()
fused_dataset = multimodal.early_fusion(
    datasets=[fmri_data, eeg_data, behavior_data],
    normalize=True  # Z-score each modality
)

print(f"Fused features: {fused_dataset.n_features}")
# fMRI voxels + EEG channels*time + behavior features

# Train on fused data
decoder = SVMDecoder(kernel="linear")
results = decoder.cross_validate(fused_dataset, LeaveOneRunOut())
```

**Advantages of Early Fusion**:
- Simple implementation
- Classifier learns cross-modal patterns
- Works well when modalities are complementary

**Considerations**:
- Need same number of samples across modalities
- May need to balance feature scales (normalize)
- High dimensionality may require regularization

---

### Late Fusion (Decision-Level)

Train separate classifiers per modality, combine predictions.

```python
from neural_decoding.models import LateFusionDecoder

# Create modality-specific decoders
fmri_decoder = SVMDecoder(kernel="linear")
eeg_decoder = SVMDecoder(kernel="rbf")
behavior_decoder = LogisticDecoder()

# Late fusion: combine predictions
late_fusion = LateFusionDecoder(
    modality_decoders={
        "fmri": fmri_decoder,
        "eeg": eeg_decoder,
        "behavior": behavior_decoder
    },
    fusion_method="voting",  # or "stacking", "weighted"
    weights=None  # Auto-weight by accuracy
)

# Fit on separate datasets
late_fusion.fit({
    "fmri": fmri_data,
    "eeg": eeg_data,
    "behavior": behavior_data
})

# Predict (requires all modalities)
predictions = late_fusion.predict({
    "fmri": X_test_fmri,
    "eeg": X_test_eeg,
    "behavior": X_test_behavior
})
```

**Fusion Methods**:
- `voting`: Majority vote or average probabilities
- `stacking`: Train meta-classifier on modality outputs
- `weighted`: Weight by modality accuracy

**Advantages of Late Fusion**:
- Modalities can have different samples
- Handles missing modalities gracefully
- Each modality optimized independently

---

### Weighted Fusion Based on Reliability

```python
# Train individual models, weight by performance
accuracies = {}
for name, dataset in [("fmri", fmri_data), ("eeg", eeg_data)]:
    decoder = SVMDecoder()
    results = decoder.cross_validate(dataset, cv)
    accuracies[name] = results.accuracy

# Normalize weights
total = sum(accuracies.values())
weights = {k: v/total for k, v in accuracies.items()}

print(f"Modality weights: {weights}")
# {'fmri': 0.65, 'eeg': 0.35}

# Use weights in late fusion
late_fusion = LateFusionDecoder(
    modality_decoders={"fmri": fmri_decoder, "eeg": eeg_decoder},
    fusion_method="weighted",
    weights=weights
)
```

---

### Hierarchical Fusion

```python
# First level: fuse related modalities
visual_data = multimodal.early_fusion([fmri_visual, eeg_visual])
motor_data = multimodal.early_fusion([fmri_motor, emg_data])

# Second level: late fusion of domains
domain_fusion = LateFusionDecoder(
    modality_decoders={
        "visual": SVMDecoder(),
        "motor": SVMDecoder()
    },
    fusion_method="stacking"
)
```

---

## Cross-Validation Strategies

| Strategy | Use Case | Code |
|----------|----------|------|
| **Leave-One-Run-Out** | fMRI with multiple runs | `LeaveOneRunOut()` |
| **Leave-One-Subject-Out** | Group analysis | `LeaveOneSubjectOut()` |
| **Stratified K-Fold** | General purpose | `StratifiedKFold(n_splits=5)` |
| **Leave-One-Out** | Small datasets | `LeaveOneOut()` |

```python
from neural_decoding.validation import (
    LeaveOneRunOut,
    LeaveOneSubjectOut,
    StratifiedKFold
)

# fMRI: leave-one-run-out (must have run_column in events)
cv = LeaveOneRunOut()

# Group analysis: leave-one-subject-out
cv = LeaveOneSubjectOut()

# General: stratified k-fold
cv = StratifiedKFold(n_splits=5, shuffle=True)

# Cross-validate
results = decoder.cross_validate(dataset, cv)
```

---

## Statistical Testing

### Permutation Test

```python
from neural_decoding.validation import PermutationTest

# Run permutation test
perm_test = PermutationTest(n_permutations=1000, random_state=42)
p_value = perm_test.test(decoder, dataset, cv)

print(f"P-value: {p_value:.4f}")
print(f"Significant: {p_value < 0.05}")
```

### Comparing Classifiers

```python
from neural_decoding.validation import compare_classifiers

results = compare_classifiers(
    dataset=dataset,
    classifiers={
        "SVM": SVMDecoder(),
        "RF": RandomForestDecoder(),
        "LR": LogisticDecoder()
    },
    cv=LeaveOneRunOut(),
    n_permutations=1000
)

# Print comparison table
print(results.to_dataframe())
```

---

## Visualization

```python
from neural_decoding.visualization import (
    plot_confusion_matrix,
    plot_cv_scores,
    plot_brain_map,
    plot_temporal_decoding,
    plot_feature_importance
)

# Confusion matrix
plot_confusion_matrix(results, normalize=True)

# CV fold scores
plot_cv_scores(results)

# Brain map (searchlight)
plot_brain_map("searchlight.nii.gz", threshold=0.6)

# Temporal decoding
plot_temporal_decoding(time_scores, times, chance=0.5)

# Feature importance
plot_feature_importance(results, top_n=20)
```

---

## Complete Workflows

### Workflow 1: Basic fMRI Classification

```python
# Full pipeline from data to results
from neural_decoding.io import FMRILoader
from neural_decoding.models import SVMDecoder
from neural_decoding.validation import LeaveOneRunOut, PermutationTest

# 1. Load
loader = FMRILoader()
dataset = loader.load("bold.nii.gz", "mask.nii.gz", "events.csv", "condition", "run")

# 2. Decode
decoder = SVMDecoder(kernel="linear")
cv = LeaveOneRunOut()
results = decoder.cross_validate(dataset, cv)

# 3. Test significance
perm = PermutationTest(n_permutations=1000)
results.permutation_pvalue = perm.test(decoder, dataset, cv)

# 4. Report
print(results.summary())
results.plot_confusion_matrix(output_path="confusion.png")
results.save("results.json")
```

### Workflow 2: Multimodal Fusion

```python
from neural_decoding.io import FMRILoader, EEGLoader, MultimodalLoader
from neural_decoding.models import SVMDecoder
from neural_decoding.validation import LeaveOneRunOut

# Load modalities
fmri = FMRILoader().load("bold.nii.gz", "mask.nii.gz", "events.csv", "condition")
eeg = EEGLoader().load("epochs.fif", time_window=(0.1, 0.5))

# Early fusion
fused = MultimodalLoader().early_fusion([fmri, eeg], normalize=True)

# Decode
results = SVMDecoder().cross_validate(fused, LeaveOneRunOut())
print(f"Multimodal accuracy: {results.accuracy:.1%}")
```

### Workflow 3: Searchlight + Group Analysis

```python
from neural_decoding.models import SearchlightDecoder
from neural_decoding.validation import LeaveOneSubjectOut

# Per-subject searchlights
for subject in subjects:
    dataset = load_subject(subject)
    searchlight = SearchlightDecoder(mask_path="mask.nii.gz", radius=5.0)
    searchlight.fit(dataset)
    searchlight.save_nifti(f"searchlight_{subject}.nii.gz")

# Group-level statistics (using nilearn)
from nilearn.image import mean_img
from nilearn.glm.second_level import SecondLevelModel

# Average across subjects
group_map = mean_img([f"searchlight_{s}.nii.gz" for s in subjects])
```

---

## Directory Structure

```
neural-decoding/
├── core/
│   ├── dataset.py      # DecodingDataset
│   ├── results.py      # DecodingResults
│   └── config.py       # Configuration
├── io/
│   └── loaders.py      # FMRILoader, EEGLoader, etc.
├── models/
│   ├── classifiers.py  # SVM, RF, LR, Ensemble
│   ├── searchlight.py  # Searchlight analysis
│   └── temporal.py     # Time-resolved decoding
├── validation/
│   ├── cross_validation.py
│   ├── metrics.py
│   └── permutation.py
├── visualization/
│   └── plots.py
└── notebooks/
    ├── 01_load_data.ipynb
    ├── 02_classification.ipynb
    └── 03_multimodal.ipynb
```

---

## Citation

If using this toolkit, please cite:

```
Neuro-Hub Neural Decoding Toolkit
https://github.com/jsj88/neuro-hub/neural-decoding
```

And the underlying tools:
- scikit-learn: https://scikit-learn.org/
- nilearn: https://nilearn.github.io/
- MNE-Python: https://mne.tools/
