"""
Multimodal fusion models for combined EEG-fMRI classification.
"""

from typing import Optional, Tuple, List
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from .base import BaseDeepDecoder


class EarlyFusion(BaseDeepDecoder):
    """
    Early fusion: Concatenate EEG and fMRI features before classification.

    Parameters
    ----------
    n_classes : int
        Number of output classes
    hidden_sizes : List[int]
        Hidden layer sizes after concatenation
    dropout : float
        Dropout rate
    """

    def __init__(
        self,
        n_classes: int = 2,
        hidden_sizes: List[int] = [256, 128],
        dropout: float = 0.5,
        **kwargs
    ):
        super().__init__(n_classes=n_classes, dropout=dropout, **kwargs)
        self.hidden_sizes = hidden_sizes

    def build_model(self, input_shape: Tuple) -> nn.Module:
        """Build early fusion model."""
        # Input shape: (eeg_features + fmri_features,)
        n_features = input_shape[0] if isinstance(input_shape, tuple) else input_shape

        return _EarlyFusionModule(
            n_features=n_features,
            n_classes=self.n_classes,
            hidden_sizes=self.hidden_sizes,
            dropout=self.dropout
        )


class _EarlyFusionModule(nn.Module):
    """Internal early fusion module."""

    def __init__(
        self,
        n_features: int,
        n_classes: int,
        hidden_sizes: List[int],
        dropout: float
    ):
        super().__init__()

        layers = []
        prev_size = n_features

        for size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, size),
                nn.BatchNorm1d(size),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_size = size

        self.features = nn.Sequential(*layers)
        self.classifier = nn.Linear(prev_size, n_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


class LateFusion(BaseDeepDecoder):
    """
    Late fusion: Separate encoders for EEG and fMRI, fused at decision level.

    Parameters
    ----------
    n_classes : int
        Number of output classes
    eeg_hidden : List[int]
        EEG encoder hidden sizes
    fmri_hidden : List[int]
        fMRI encoder hidden sizes
    fusion_hidden : List[int]
        Fusion layer hidden sizes
    dropout : float
        Dropout rate
    """

    def __init__(
        self,
        n_classes: int = 2,
        eeg_hidden: List[int] = [128, 64],
        fmri_hidden: List[int] = [256, 128],
        fusion_hidden: List[int] = [64],
        dropout: float = 0.5,
        n_eeg_features: int = 64,
        n_fmri_features: int = 426,
        **kwargs
    ):
        super().__init__(n_classes=n_classes, dropout=dropout, **kwargs)

        self.eeg_hidden = eeg_hidden
        self.fmri_hidden = fmri_hidden
        self.fusion_hidden = fusion_hidden
        self.n_eeg_features = n_eeg_features
        self.n_fmri_features = n_fmri_features

    def build_model(self, input_shape: Tuple) -> nn.Module:
        """Build late fusion model."""
        return _LateFusionModule(
            n_eeg=self.n_eeg_features,
            n_fmri=self.n_fmri_features,
            n_classes=self.n_classes,
            eeg_hidden=self.eeg_hidden,
            fmri_hidden=self.fmri_hidden,
            fusion_hidden=self.fusion_hidden,
            dropout=self.dropout
        )


class _LateFusionModule(nn.Module):
    """Internal late fusion module."""

    def __init__(
        self,
        n_eeg: int,
        n_fmri: int,
        n_classes: int,
        eeg_hidden: List[int],
        fmri_hidden: List[int],
        fusion_hidden: List[int],
        dropout: float
    ):
        super().__init__()

        self.n_eeg = n_eeg

        # EEG encoder
        eeg_layers = []
        prev = n_eeg
        for size in eeg_hidden:
            eeg_layers.extend([
                nn.Linear(prev, size),
                nn.BatchNorm1d(size),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev = size
        self.eeg_encoder = nn.Sequential(*eeg_layers)
        self.eeg_out_size = eeg_hidden[-1]

        # fMRI encoder
        fmri_layers = []
        prev = n_fmri
        for size in fmri_hidden:
            fmri_layers.extend([
                nn.Linear(prev, size),
                nn.BatchNorm1d(size),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev = size
        self.fmri_encoder = nn.Sequential(*fmri_layers)
        self.fmri_out_size = fmri_hidden[-1]

        # Fusion layers
        fusion_layers = []
        prev = self.eeg_out_size + self.fmri_out_size
        for size in fusion_hidden:
            fusion_layers.extend([
                nn.Linear(prev, size),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev = size
        self.fusion = nn.Sequential(*fusion_layers)

        # Classifier
        self.classifier = nn.Linear(prev, n_classes)

    def forward(self, x):
        # Split input into EEG and fMRI
        eeg = x[:, :self.n_eeg]
        fmri = x[:, self.n_eeg:]

        # Encode separately
        eeg_features = self.eeg_encoder(eeg)
        fmri_features = self.fmri_encoder(fmri)

        # Concatenate and fuse
        combined = torch.cat([eeg_features, fmri_features], dim=1)
        fused = self.fusion(combined)

        # Classify
        out = self.classifier(fused)
        return out


class CrossAttentionFusion(BaseDeepDecoder):
    """
    Cross-attention fusion: EEG and fMRI attend to each other.

    Parameters
    ----------
    n_classes : int
        Number of output classes
    d_model : int
        Model dimension
    n_heads : int
        Number of attention heads
    n_layers : int
        Number of cross-attention layers
    dropout : float
        Dropout rate
    """

    def __init__(
        self,
        n_classes: int = 2,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1,
        n_eeg_features: int = 64,
        n_fmri_features: int = 426,
        **kwargs
    ):
        super().__init__(n_classes=n_classes, dropout=dropout, **kwargs)

        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.n_eeg_features = n_eeg_features
        self.n_fmri_features = n_fmri_features

    def build_model(self, input_shape: Tuple) -> nn.Module:
        return _CrossAttentionModule(
            n_eeg=self.n_eeg_features,
            n_fmri=self.n_fmri_features,
            n_classes=self.n_classes,
            d_model=self.d_model,
            n_heads=self.n_heads,
            n_layers=self.n_layers,
            dropout=self.dropout
        )


class _CrossAttentionModule(nn.Module):
    """Internal cross-attention module."""

    def __init__(
        self,
        n_eeg: int,
        n_fmri: int,
        n_classes: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        dropout: float
    ):
        super().__init__()

        self.n_eeg = n_eeg
        self.d_model = d_model

        # Project to common dimension
        self.eeg_proj = nn.Linear(n_eeg, d_model)
        self.fmri_proj = nn.Linear(n_fmri, d_model)

        # Cross-attention layers
        self.cross_attn_layers = nn.ModuleList()
        for _ in range(n_layers):
            self.cross_attn_layers.append(
                _CrossAttentionLayer(d_model, n_heads, dropout)
            )

        # Fusion and classification
        self.fusion = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.classifier = nn.Linear(d_model, n_classes)

    def forward(self, x):
        # Split
        eeg = x[:, :self.n_eeg]
        fmri = x[:, self.n_eeg:]

        # Project
        eeg_emb = self.eeg_proj(eeg).unsqueeze(1)  # (batch, 1, d_model)
        fmri_emb = self.fmri_proj(fmri).unsqueeze(1)

        # Cross-attention
        for layer in self.cross_attn_layers:
            eeg_emb, fmri_emb = layer(eeg_emb, fmri_emb)

        # Fuse
        eeg_out = eeg_emb.squeeze(1)
        fmri_out = fmri_emb.squeeze(1)
        combined = torch.cat([eeg_out, fmri_out], dim=1)
        fused = self.fusion(combined)

        # Classify
        out = self.classifier(fused)
        return out


class _CrossAttentionLayer(nn.Module):
    """Cross-attention layer."""

    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()

        # EEG attends to fMRI
        self.eeg_cross_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.eeg_norm = nn.LayerNorm(d_model)

        # fMRI attends to EEG
        self.fmri_cross_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.fmri_norm = nn.LayerNorm(d_model)

        # FFN
        self.eeg_ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout)
        )
        self.fmri_ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, eeg, fmri):
        # EEG attends to fMRI
        eeg_attn, _ = self.eeg_cross_attn(eeg, fmri, fmri)
        eeg = self.eeg_norm(eeg + eeg_attn)
        eeg = eeg + self.eeg_ffn(eeg)

        # fMRI attends to EEG
        fmri_attn, _ = self.fmri_cross_attn(fmri, eeg, eeg)
        fmri = self.fmri_norm(fmri + fmri_attn)
        fmri = fmri + self.fmri_ffn(fmri)

        return eeg, fmri


class HierarchicalFusion(BaseDeepDecoder):
    """
    Hierarchical fusion: Multi-level integration of modalities.

    Level 1: Early feature fusion
    Level 2: Intermediate representation fusion
    Level 3: Late decision fusion

    Parameters
    ----------
    n_classes : int
        Number of output classes
    dropout : float
        Dropout rate
    """

    def __init__(
        self,
        n_classes: int = 2,
        dropout: float = 0.5,
        n_eeg_features: int = 64,
        n_fmri_features: int = 426,
        **kwargs
    ):
        super().__init__(n_classes=n_classes, dropout=dropout, **kwargs)
        self.n_eeg_features = n_eeg_features
        self.n_fmri_features = n_fmri_features

    def build_model(self, input_shape: Tuple) -> nn.Module:
        return _HierarchicalFusionModule(
            n_eeg=self.n_eeg_features,
            n_fmri=self.n_fmri_features,
            n_classes=self.n_classes,
            dropout=self.dropout
        )


class _HierarchicalFusionModule(nn.Module):
    """Internal hierarchical fusion module."""

    def __init__(
        self,
        n_eeg: int,
        n_fmri: int,
        n_classes: int,
        dropout: float
    ):
        super().__init__()

        self.n_eeg = n_eeg
        hidden = 64

        # Level 1: Early fusion branch
        self.early_fusion = nn.Sequential(
            nn.Linear(n_eeg + n_fmri, hidden * 2),
            nn.BatchNorm1d(hidden * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden * 2, hidden),
            nn.ReLU()
        )

        # Level 2: Modality-specific encoders
        self.eeg_encoder = nn.Sequential(
            nn.Linear(n_eeg, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU()
        )

        self.fmri_encoder = nn.Sequential(
            nn.Linear(n_fmri, hidden * 2),
            nn.BatchNorm1d(hidden * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden * 2, hidden // 2),
            nn.ReLU()
        )

        # Level 3: Final fusion
        # Input: early (hidden) + eeg (hidden/2) + fmri (hidden/2)
        self.final_fusion = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_classes)
        )

    def forward(self, x):
        # Split
        eeg = x[:, :self.n_eeg]
        fmri = x[:, self.n_eeg:]

        # Level 1: Early fusion
        early_combined = torch.cat([eeg, fmri], dim=1)
        early_features = self.early_fusion(early_combined)

        # Level 2: Modality-specific
        eeg_features = self.eeg_encoder(eeg)
        fmri_features = self.fmri_encoder(fmri)

        # Level 3: Hierarchical combination
        all_features = torch.cat([
            early_features,
            eeg_features,
            fmri_features
        ], dim=1)

        out = self.final_fusion(all_features)
        return out
