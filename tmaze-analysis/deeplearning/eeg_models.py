"""
Deep learning models for EEG classification.

Includes EEGNet, LSTM, and Transformer architectures optimized
for temporal EEG decoding.
"""

from typing import Optional, Tuple
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from .base import BaseDeepDecoder


class EEGNet(BaseDeepDecoder):
    """
    EEGNet architecture for EEG classification.

    Based on: Lawhern et al. (2018) "EEGNet: A Compact Convolutional
    Neural Network for EEG-based Brain-Computer Interfaces"

    Parameters
    ----------
    n_classes : int
        Number of output classes
    n_channels : int
        Number of EEG channels
    n_times : int
        Number of time samples
    f1 : int
        Number of temporal filters
    d : int
        Depth multiplier for depthwise convolution
    f2 : int
        Number of pointwise filters (f1 * d by default)
    kernel_length : int
        Length of temporal kernel
    dropout : float
        Dropout rate
    """

    def __init__(
        self,
        n_classes: int = 2,
        n_channels: int = 64,
        n_times: int = 128,
        f1: int = 8,
        d: int = 2,
        f2: Optional[int] = None,
        kernel_length: int = 64,
        dropout: float = 0.5,
        **kwargs
    ):
        super().__init__(n_classes=n_classes, dropout=dropout, **kwargs)

        self.n_channels = n_channels
        self.n_times = n_times
        self.f1 = f1
        self.d = d
        self.f2 = f2 if f2 is not None else f1 * d
        self.kernel_length = kernel_length

    def build_model(self, input_shape: Tuple) -> nn.Module:
        """Build EEGNet architecture."""
        # Input shape: (n_channels, n_times)
        if len(input_shape) == 2:
            n_channels, n_times = input_shape
        else:
            # Assume (1, n_channels, n_times) format
            n_channels, n_times = input_shape[-2], input_shape[-1]

        return _EEGNetModule(
            n_channels=n_channels,
            n_times=n_times,
            n_classes=self.n_classes,
            f1=self.f1,
            d=self.d,
            f2=self.f2,
            kernel_length=self.kernel_length,
            dropout=self.dropout
        )


class _EEGNetModule(nn.Module):
    """Internal EEGNet PyTorch module."""

    def __init__(
        self,
        n_channels: int,
        n_times: int,
        n_classes: int,
        f1: int,
        d: int,
        f2: int,
        kernel_length: int,
        dropout: float
    ):
        super().__init__()

        # Block 1: Temporal convolution
        self.conv1 = nn.Conv2d(1, f1, (1, kernel_length), padding='same', bias=False)
        self.bn1 = nn.BatchNorm2d(f1)

        # Block 1: Depthwise convolution (spatial)
        self.conv2 = nn.Conv2d(f1, f1 * d, (n_channels, 1), groups=f1, bias=False)
        self.bn2 = nn.BatchNorm2d(f1 * d)
        self.pool1 = nn.AvgPool2d((1, 4))
        self.drop1 = nn.Dropout(dropout)

        # Block 2: Separable convolution
        self.conv3 = nn.Conv2d(f1 * d, f1 * d, (1, 16), padding='same',
                               groups=f1 * d, bias=False)
        self.conv4 = nn.Conv2d(f1 * d, f2, (1, 1), bias=False)
        self.bn3 = nn.BatchNorm2d(f2)
        self.pool2 = nn.AvgPool2d((1, 8))
        self.drop2 = nn.Dropout(dropout)

        # Calculate flattened size
        with torch.no_grad():
            x = torch.zeros(1, 1, n_channels, n_times)
            x = self._forward_features(x)
            self.flat_size = x.numel()

        # Classifier
        self.classifier = nn.Linear(self.flat_size, n_classes)

    def _forward_features(self, x):
        # Block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.elu(x)
        x = self.pool1(x)
        x = self.drop1(x)

        # Block 2
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.bn3(x)
        x = F.elu(x)
        x = self.pool2(x)
        x = self.drop2(x)

        return x

    def forward(self, x):
        # Add channel dimension if needed
        if x.dim() == 3:
            x = x.unsqueeze(1)

        x = self._forward_features(x)
        x = x.flatten(1)
        x = self.classifier(x)
        return x


class LSTMDecoder(BaseDeepDecoder):
    """
    LSTM-based decoder for EEG temporal classification.

    Parameters
    ----------
    n_classes : int
        Number of output classes
    hidden_size : int
        LSTM hidden state size
    n_layers : int
        Number of LSTM layers
    bidirectional : bool
        Use bidirectional LSTM
    dropout : float
        Dropout rate
    """

    def __init__(
        self,
        n_classes: int = 2,
        hidden_size: int = 64,
        n_layers: int = 2,
        bidirectional: bool = True,
        dropout: float = 0.5,
        **kwargs
    ):
        super().__init__(n_classes=n_classes, dropout=dropout, **kwargs)

        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.bidirectional = bidirectional

    def build_model(self, input_shape: Tuple) -> nn.Module:
        """Build LSTM architecture."""
        # Input shape: (n_channels, n_times) or (n_times, n_features)
        if len(input_shape) == 2:
            n_features = input_shape[0]  # Channels as features
        else:
            n_features = input_shape[-1]

        return _LSTMModule(
            input_size=n_features,
            hidden_size=self.hidden_size,
            n_layers=self.n_layers,
            n_classes=self.n_classes,
            bidirectional=self.bidirectional,
            dropout=self.dropout
        )


class _LSTMModule(nn.Module):
    """Internal LSTM PyTorch module."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        n_layers: int,
        n_classes: int,
        bidirectional: bool,
        dropout: float
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if n_layers > 1 else 0
        )

        lstm_output_size = hidden_size * 2 if bidirectional else hidden_size

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(lstm_output_size, n_classes)

    def forward(self, x):
        # x: (batch, channels, times) -> (batch, times, channels)
        if x.dim() == 3:
            x = x.permute(0, 2, 1)

        lstm_out, (hidden, _) = self.lstm(x)

        # Use last hidden state
        if self.lstm.bidirectional:
            hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)
        else:
            hidden = hidden[-1]

        out = self.dropout(hidden)
        out = self.fc(out)
        return out


class EEGTransformer(BaseDeepDecoder):
    """
    Transformer-based decoder for EEG classification.

    Uses self-attention to capture temporal dependencies.

    Parameters
    ----------
    n_classes : int
        Number of output classes
    d_model : int
        Transformer embedding dimension
    n_heads : int
        Number of attention heads
    n_layers : int
        Number of transformer layers
    dim_feedforward : int
        Feedforward network dimension
    dropout : float
        Dropout rate
    """

    def __init__(
        self,
        n_classes: int = 2,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
        **kwargs
    ):
        super().__init__(n_classes=n_classes, dropout=dropout, **kwargs)

        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.dim_feedforward = dim_feedforward

    def build_model(self, input_shape: Tuple) -> nn.Module:
        """Build Transformer architecture."""
        if len(input_shape) == 2:
            n_channels, n_times = input_shape
        else:
            n_channels, n_times = input_shape[-2], input_shape[-1]

        return _EEGTransformerModule(
            n_channels=n_channels,
            n_times=n_times,
            n_classes=self.n_classes,
            d_model=self.d_model,
            n_heads=self.n_heads,
            n_layers=self.n_layers,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout
        )


class _EEGTransformerModule(nn.Module):
    """Internal Transformer PyTorch module."""

    def __init__(
        self,
        n_channels: int,
        n_times: int,
        n_classes: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        dim_feedforward: int,
        dropout: float
    ):
        super().__init__()

        self.d_model = d_model

        # Input projection
        self.input_proj = nn.Linear(n_channels, d_model)

        # Positional encoding
        self.pos_encoding = nn.Parameter(torch.randn(1, n_times, d_model) * 0.1)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Class token
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

        # Classifier
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(d_model, n_classes)

    def forward(self, x):
        # x: (batch, channels, times) -> (batch, times, channels)
        if x.dim() == 3:
            x = x.permute(0, 2, 1)

        batch_size = x.size(0)

        # Project to d_model
        x = self.input_proj(x)

        # Add positional encoding
        x = x + self.pos_encoding[:, :x.size(1), :]

        # Prepend class token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)

        # Transformer
        x = self.transformer(x)

        # Use class token output
        cls_output = x[:, 0]
        out = self.dropout(cls_output)
        out = self.classifier(out)

        return out


class ShallowConvNet(BaseDeepDecoder):
    """
    Shallow ConvNet for EEG classification.

    Based on: Schirrmeister et al. (2017) "Deep learning with
    convolutional neural networks for EEG decoding and visualization"

    Parameters
    ----------
    n_classes : int
        Number of output classes
    n_filters : int
        Number of temporal filters
    filter_time_length : int
        Length of temporal filters
    pool_time_length : int
        Length of pooling kernel
    pool_time_stride : int
        Stride of pooling
    dropout : float
        Dropout rate
    """

    def __init__(
        self,
        n_classes: int = 2,
        n_filters: int = 40,
        filter_time_length: int = 25,
        pool_time_length: int = 75,
        pool_time_stride: int = 15,
        dropout: float = 0.5,
        **kwargs
    ):
        super().__init__(n_classes=n_classes, dropout=dropout, **kwargs)

        self.n_filters = n_filters
        self.filter_time_length = filter_time_length
        self.pool_time_length = pool_time_length
        self.pool_time_stride = pool_time_stride

    def build_model(self, input_shape: Tuple) -> nn.Module:
        if len(input_shape) == 2:
            n_channels, n_times = input_shape
        else:
            n_channels, n_times = input_shape[-2], input_shape[-1]

        return _ShallowConvNetModule(
            n_channels=n_channels,
            n_times=n_times,
            n_classes=self.n_classes,
            n_filters=self.n_filters,
            filter_time_length=self.filter_time_length,
            pool_time_length=self.pool_time_length,
            pool_time_stride=self.pool_time_stride,
            dropout=self.dropout
        )


class _ShallowConvNetModule(nn.Module):
    """Internal Shallow ConvNet module."""

    def __init__(
        self,
        n_channels: int,
        n_times: int,
        n_classes: int,
        n_filters: int,
        filter_time_length: int,
        pool_time_length: int,
        pool_time_stride: int,
        dropout: float
    ):
        super().__init__()

        # Temporal convolution
        self.conv_time = nn.Conv2d(1, n_filters, (1, filter_time_length), bias=False)

        # Spatial convolution
        self.conv_spat = nn.Conv2d(n_filters, n_filters, (n_channels, 1), bias=False)
        self.bn = nn.BatchNorm2d(n_filters)

        # Pooling
        self.pool = nn.AvgPool2d((1, pool_time_length), stride=(1, pool_time_stride))
        self.drop = nn.Dropout(dropout)

        # Calculate output size
        with torch.no_grad():
            x = torch.zeros(1, 1, n_channels, n_times)
            x = self.conv_time(x)
            x = self.conv_spat(x)
            x = self.pool(x)
            self.flat_size = x.numel()

        self.classifier = nn.Linear(self.flat_size, n_classes)

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)

        x = self.conv_time(x)
        x = self.conv_spat(x)
        x = self.bn(x)
        x = x.pow(2)  # Square nonlinearity
        x = self.pool(x)
        x = torch.log(torch.clamp(x, min=1e-6))  # Log nonlinearity
        x = self.drop(x)
        x = x.flatten(1)
        x = self.classifier(x)

        return x
