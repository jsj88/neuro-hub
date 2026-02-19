"""
Deep learning models for T-maze classification.

Includes:
- EEG models (EEGNet, LSTM, Transformer)
- fMRI models (ROI MLP, 3D CNN)
- Multimodal fusion models
- Training infrastructure
- Interpretability tools
"""

from .base import BaseDeepDecoder, DeepDecodingResult

from .eeg_models import (
    EEGNet,
    LSTMDecoder,
    EEGTransformer,
    ShallowConvNet
)

from .fmri_models import (
    ROIMLP,
    BrainNetCNN,
    GraphAttentionNet
)

from .multimodal_models import (
    CrossAttentionFusion,
    EarlyFusion,
    LateFusion,
    HierarchicalFusion
)

from .training import (
    DeepTrainer,
    EarlyStopping,
    LRScheduler,
    cross_validate_deep,
    train_test_split_subjects
)

from .interpretation import (
    GradCAM,
    IntegratedGradients,
    attention_weights,
    feature_importance_deep,
    layer_activations
)

__all__ = [
    # base
    'BaseDeepDecoder',
    'DeepDecodingResult',
    # eeg_models
    'EEGNet',
    'LSTMDecoder',
    'EEGTransformer',
    'ShallowConvNet',
    # fmri_models
    'ROIMLP',
    'BrainNetCNN',
    'GraphAttentionNet',
    # multimodal_models
    'CrossAttentionFusion',
    'EarlyFusion',
    'LateFusion',
    'HierarchicalFusion',
    # training
    'DeepTrainer',
    'EarlyStopping',
    'LRScheduler',
    'cross_validate_deep',
    'train_test_split_subjects',
    # interpretation
    'GradCAM',
    'IntegratedGradients',
    'attention_weights',
    'feature_importance_deep',
    'layer_activations'
]
