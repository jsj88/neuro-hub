"""
Deep learning models for fMRI classification.

Includes MLP for ROI data and graph neural networks for
connectivity-based classification.
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


class ROIMLP(BaseDeepDecoder):
    """
    Multi-layer perceptron for ROI-based fMRI classification.

    Parameters
    ----------
    n_classes : int
        Number of output classes
    hidden_sizes : List[int]
        Sizes of hidden layers
    dropout : float
        Dropout rate
    batch_norm : bool
        Use batch normalization
    activation : str
        'relu', 'elu', or 'gelu'
    """

    def __init__(
        self,
        n_classes: int = 2,
        hidden_sizes: List[int] = [256, 128, 64],
        dropout: float = 0.5,
        batch_norm: bool = True,
        activation: str = 'relu',
        **kwargs
    ):
        super().__init__(n_classes=n_classes, dropout=dropout, **kwargs)

        self.hidden_sizes = hidden_sizes
        self.batch_norm = batch_norm
        self.activation = activation

    def build_model(self, input_shape: Tuple) -> nn.Module:
        """Build MLP architecture."""
        # Input shape: (n_rois,) or (n_trials, n_rois)
        if isinstance(input_shape, int):
            n_features = input_shape
        else:
            n_features = input_shape[-1] if len(input_shape) > 0 else input_shape[0]

        return _ROIMLPModule(
            n_features=n_features,
            n_classes=self.n_classes,
            hidden_sizes=self.hidden_sizes,
            dropout=self.dropout,
            batch_norm=self.batch_norm,
            activation=self.activation
        )


class _ROIMLPModule(nn.Module):
    """Internal MLP module."""

    def __init__(
        self,
        n_features: int,
        n_classes: int,
        hidden_sizes: List[int],
        dropout: float,
        batch_norm: bool,
        activation: str
    ):
        super().__init__()

        layers = []
        prev_size = n_features

        # Activation function
        if activation == 'relu':
            act_fn = nn.ReLU
        elif activation == 'elu':
            act_fn = nn.ELU
        elif activation == 'gelu':
            act_fn = nn.GELU
        else:
            act_fn = nn.ReLU

        # Hidden layers
        for size in hidden_sizes:
            layers.append(nn.Linear(prev_size, size))
            if batch_norm:
                layers.append(nn.BatchNorm1d(size))
            layers.append(act_fn())
            layers.append(nn.Dropout(dropout))
            prev_size = size

        self.features = nn.Sequential(*layers)
        self.classifier = nn.Linear(prev_size, n_classes)

    def forward(self, x):
        # Flatten if needed
        if x.dim() > 2:
            x = x.flatten(1)
        x = self.features(x)
        x = self.classifier(x)
        return x


class BrainNetCNN(BaseDeepDecoder):
    """
    BrainNetCNN for connectivity-based classification.

    Based on: Kawahara et al. (2017) "BrainNetCNN: Convolutional neural
    networks for brain networks; towards predicting neurodevelopment"

    Parameters
    ----------
    n_classes : int
        Number of output classes
    n_filters : List[int]
        Number of filters in each E2E layer
    dropout : float
        Dropout rate
    """

    def __init__(
        self,
        n_classes: int = 2,
        n_filters: List[int] = [32, 64, 128],
        dropout: float = 0.5,
        **kwargs
    ):
        super().__init__(n_classes=n_classes, dropout=dropout, **kwargs)
        self.n_filters = n_filters

    def build_model(self, input_shape: Tuple) -> nn.Module:
        """Build BrainNetCNN architecture."""
        # Input shape: (n_rois, n_rois) connectivity matrix
        if len(input_shape) == 2:
            n_rois = input_shape[0]
        else:
            n_rois = input_shape[-1]

        return _BrainNetCNNModule(
            n_rois=n_rois,
            n_classes=self.n_classes,
            n_filters=self.n_filters,
            dropout=self.dropout
        )


class _BrainNetCNNModule(nn.Module):
    """Internal BrainNetCNN module."""

    def __init__(
        self,
        n_rois: int,
        n_classes: int,
        n_filters: List[int],
        dropout: float
    ):
        super().__init__()

        self.n_rois = n_rois

        # Edge-to-Edge (E2E) layers
        # Convolve over rows and columns of connectivity matrix
        self.e2e_layers = nn.ModuleList()
        self.e2e_bn = nn.ModuleList()

        in_channels = 1
        for n_filt in n_filters:
            # Row convolution (along each ROI)
            self.e2e_layers.append(
                nn.Conv2d(in_channels, n_filt, (1, n_rois), bias=False)
            )
            self.e2e_bn.append(nn.BatchNorm2d(n_filt))
            in_channels = n_filt

        # Edge-to-Node (E2N) layer
        self.e2n = nn.Conv2d(n_filters[-1], n_filters[-1], (n_rois, 1), bias=False)
        self.e2n_bn = nn.BatchNorm2d(n_filters[-1])

        # Node-to-Graph (N2G) layer
        self.n2g = nn.Conv1d(n_filters[-1], n_filters[-1], n_rois, bias=False)
        self.n2g_bn = nn.BatchNorm1d(n_filters[-1])

        # Classifier
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(n_filters[-1], n_classes)

    def forward(self, x):
        # x: (batch, n_rois, n_rois) or (batch, 1, n_rois, n_rois)
        if x.dim() == 3:
            x = x.unsqueeze(1)

        # E2E layers
        for e2e, bn in zip(self.e2e_layers, self.e2e_bn):
            x = e2e(x)
            x = bn(x)
            x = F.leaky_relu(x)

        # E2N layer
        x = self.e2n(x)
        x = self.e2n_bn(x)
        x = F.leaky_relu(x)

        # Reshape for N2G: (batch, filters, 1, n_rois) -> (batch, filters, n_rois)
        x = x.squeeze(2)

        # N2G layer
        x = self.n2g(x)
        x = self.n2g_bn(x)
        x = F.leaky_relu(x)

        # Global features
        x = x.squeeze(-1)
        x = self.dropout(x)
        x = self.classifier(x)

        return x


class GraphAttentionNet(BaseDeepDecoder):
    """
    Graph Attention Network for brain connectivity.

    Uses attention mechanism to weight ROI connections.

    Parameters
    ----------
    n_classes : int
        Number of output classes
    hidden_dim : int
        Hidden layer dimension
    n_heads : int
        Number of attention heads
    n_layers : int
        Number of GAT layers
    dropout : float
        Dropout rate
    """

    def __init__(
        self,
        n_classes: int = 2,
        hidden_dim: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.5,
        **kwargs
    ):
        super().__init__(n_classes=n_classes, dropout=dropout, **kwargs)

        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.n_layers = n_layers

    def build_model(self, input_shape: Tuple) -> nn.Module:
        """Build Graph Attention Network."""
        # Input: (n_rois, n_rois) connectivity + (n_rois,) or (n_rois, n_features) node features
        if len(input_shape) == 2:
            n_rois, n_features = input_shape
        else:
            n_rois = input_shape[0]
            n_features = 1  # Use connectivity as features

        return _GraphAttentionModule(
            n_nodes=n_rois,
            in_features=n_features,
            hidden_dim=self.hidden_dim,
            n_classes=self.n_classes,
            n_heads=self.n_heads,
            n_layers=self.n_layers,
            dropout=self.dropout
        )


class _GraphAttentionModule(nn.Module):
    """Internal GAT module."""

    def __init__(
        self,
        n_nodes: int,
        in_features: int,
        hidden_dim: int,
        n_classes: int,
        n_heads: int,
        n_layers: int,
        dropout: float
    ):
        super().__init__()

        self.n_nodes = n_nodes

        # GAT layers
        self.gat_layers = nn.ModuleList()

        # First layer
        self.gat_layers.append(
            _GATLayer(in_features, hidden_dim, n_heads, dropout)
        )

        # Subsequent layers
        for _ in range(n_layers - 1):
            self.gat_layers.append(
                _GATLayer(hidden_dim * n_heads, hidden_dim, n_heads, dropout)
            )

        # Readout
        self.readout = nn.Sequential(
            nn.Linear(hidden_dim * n_heads, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_classes)
        )

    def forward(self, x):
        # x can be:
        # - (batch, n_rois, n_rois): connectivity matrix (use as adjacency + diagonal as features)
        # - (batch, n_rois, n_features): node features + adjacency from __init__

        if x.dim() == 3 and x.size(1) == x.size(2):
            # Connectivity matrix
            adj = x
            # Use node degree as feature
            node_features = x.sum(dim=2, keepdim=True)  # (batch, n_rois, 1)
        else:
            # Assume features with implicit full connectivity
            node_features = x
            adj = torch.ones(x.size(0), self.n_nodes, self.n_nodes, device=x.device)

        h = node_features

        # GAT layers
        for gat in self.gat_layers:
            h = gat(h, adj)

        # Global mean pooling
        h = h.mean(dim=1)  # (batch, hidden * heads)

        # Classifier
        out = self.readout(h)
        return out


class _GATLayer(nn.Module):
    """Single Graph Attention layer."""

    def __init__(self, in_features: int, out_features: int, n_heads: int, dropout: float):
        super().__init__()

        self.n_heads = n_heads
        self.out_features = out_features

        # Linear transformations for each head
        self.W = nn.Linear(in_features, out_features * n_heads, bias=False)

        # Attention parameters
        self.a = nn.Parameter(torch.zeros(n_heads, 2 * out_features))
        nn.init.xavier_uniform_(self.a)

        self.leaky_relu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h, adj):
        """
        h: (batch, n_nodes, in_features)
        adj: (batch, n_nodes, n_nodes)
        """
        batch_size, n_nodes, _ = h.shape

        # Transform
        Wh = self.W(h)  # (batch, n_nodes, out_features * n_heads)
        Wh = Wh.view(batch_size, n_nodes, self.n_heads, self.out_features)

        # Attention coefficients
        # For each pair (i, j): a^T [Wh_i || Wh_j]
        Wh_i = Wh.unsqueeze(2)  # (batch, n_nodes, 1, heads, out)
        Wh_j = Wh.unsqueeze(1)  # (batch, 1, n_nodes, heads, out)

        # Concatenate
        Wh_concat = torch.cat([
            Wh_i.expand(-1, -1, n_nodes, -1, -1),
            Wh_j.expand(-1, n_nodes, -1, -1, -1)
        ], dim=-1)  # (batch, n_nodes, n_nodes, heads, 2*out)

        # Attention scores
        e = (Wh_concat * self.a).sum(dim=-1)  # (batch, n_nodes, n_nodes, heads)
        e = self.leaky_relu(e)

        # Mask with adjacency
        mask = adj.unsqueeze(-1).expand_as(e)
        e = e.masked_fill(mask == 0, float('-inf'))

        # Softmax
        alpha = F.softmax(e, dim=2)
        alpha = self.dropout(alpha)

        # Aggregate
        # Wh: (batch, n_nodes, heads, out) -> weighted sum over neighbors
        h_new = torch.einsum('bijk,bjkl->bikl', alpha, Wh)  # (batch, n_nodes, heads, out)

        # Flatten heads
        h_new = h_new.reshape(batch_size, n_nodes, -1)

        return F.elu(h_new)
