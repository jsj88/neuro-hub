"""
Interpretation tools for deep learning models.

Provides GradCAM, integrated gradients, and attention visualization
for understanding model decisions.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class GradCAM:
    """
    Gradient-weighted Class Activation Mapping.

    Highlights regions of input most relevant for classification.

    Parameters
    ----------
    model : nn.Module
        Trained PyTorch model
    target_layer : str
        Name of layer to compute CAM for
    """

    def __init__(self, model: nn.Module, target_layer: str):
        if not HAS_TORCH:
            raise ImportError("PyTorch required")

        self.model = model
        self.model.eval()

        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # Register hooks
        self._register_hooks()

    def _register_hooks(self):
        """Register forward and backward hooks."""
        for name, module in self.model.named_modules():
            if name == self.target_layer:
                module.register_forward_hook(self._forward_hook)
                module.register_full_backward_hook(self._backward_hook)
                return

        raise ValueError(f"Layer '{self.target_layer}' not found in model")

    def _forward_hook(self, module, input, output):
        """Store activations during forward pass."""
        self.activations = output.detach()

    def _backward_hook(self, module, grad_input, grad_output):
        """Store gradients during backward pass."""
        self.gradients = grad_output[0].detach()

    def __call__(
        self,
        x: np.ndarray,
        target_class: Optional[int] = None
    ) -> np.ndarray:
        """
        Compute GradCAM for input.

        Parameters
        ----------
        x : np.ndarray
            Input data (batch_size, ...)
        target_class : int, optional
            Class to compute CAM for (default: predicted class)

        Returns
        -------
        np.ndarray
            GradCAM heatmap
        """
        x_tensor = torch.FloatTensor(x)
        if x_tensor.dim() == 2:
            x_tensor = x_tensor.unsqueeze(0)

        x_tensor.requires_grad_(True)

        # Forward pass
        output = self.model(x_tensor)

        if target_class is None:
            target_class = output.argmax(dim=1).item()

        # Backward pass
        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0, target_class] = 1
        output.backward(gradient=one_hot, retain_graph=True)

        # Compute CAM
        # Global average pooling of gradients
        weights = self.gradients.mean(dim=(2, 3) if self.gradients.dim() == 4
                                       else tuple(range(2, self.gradients.dim())),
                                       keepdim=True)

        # Weighted combination of activations
        cam = (weights * self.activations).sum(dim=1, keepdim=True)

        # ReLU and normalize
        cam = F.relu(cam)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)

        return cam.squeeze().cpu().numpy()


class IntegratedGradients:
    """
    Integrated Gradients attribution method.

    Computes feature importance by integrating gradients along
    path from baseline to input.

    Parameters
    ----------
    model : nn.Module
        Trained PyTorch model
    n_steps : int
        Number of interpolation steps
    """

    def __init__(self, model: nn.Module, n_steps: int = 50):
        if not HAS_TORCH:
            raise ImportError("PyTorch required")

        self.model = model
        self.model.eval()
        self.n_steps = n_steps

    def __call__(
        self,
        x: np.ndarray,
        baseline: Optional[np.ndarray] = None,
        target_class: Optional[int] = None
    ) -> np.ndarray:
        """
        Compute integrated gradients.

        Parameters
        ----------
        x : np.ndarray
            Input data
        baseline : np.ndarray, optional
            Baseline for integration (default: zeros)
        target_class : int, optional
            Target class (default: predicted)

        Returns
        -------
        np.ndarray
            Attribution map (same shape as input)
        """
        x_tensor = torch.FloatTensor(x)
        if x_tensor.dim() == 1:
            x_tensor = x_tensor.unsqueeze(0)

        if baseline is None:
            baseline = torch.zeros_like(x_tensor)
        else:
            baseline = torch.FloatTensor(baseline)

        # Get target class
        if target_class is None:
            with torch.no_grad():
                output = self.model(x_tensor)
                target_class = output.argmax(dim=1).item()

        # Generate interpolated inputs
        alphas = torch.linspace(0, 1, self.n_steps + 1).view(-1, *([1] * (x_tensor.dim() - 1)))
        interpolated = baseline + alphas * (x_tensor - baseline)

        # Compute gradients at each step
        interpolated.requires_grad_(True)

        # Batch forward pass
        outputs = self.model(interpolated)

        # Backward for target class
        target_outputs = outputs[:, target_class]
        grads = torch.autograd.grad(
            outputs=target_outputs.sum(),
            inputs=interpolated,
            create_graph=False
        )[0]

        # Integrate (average gradients * input difference)
        avg_grads = grads.mean(dim=0)
        integrated_grads = (x_tensor - baseline) * avg_grads

        return integrated_grads.squeeze().detach().numpy()


def attention_weights(
    model: nn.Module,
    x: np.ndarray,
    layer_name: Optional[str] = None
) -> Dict[str, np.ndarray]:
    """
    Extract attention weights from transformer models.

    Parameters
    ----------
    model : nn.Module
        Model with attention layers
    x : np.ndarray
        Input data
    layer_name : str, optional
        Specific layer to extract (default: all)

    Returns
    -------
    Dict[str, np.ndarray]
        Attention weights per layer
    """
    if not HAS_TORCH:
        raise ImportError("PyTorch required")

    model.eval()
    attention_maps = {}

    # Register hooks to capture attention
    handles = []

    def make_hook(name):
        def hook(module, input, output):
            if isinstance(output, tuple) and len(output) == 2:
                # MultiheadAttention returns (output, weights)
                attention_maps[name] = output[1].detach().cpu().numpy()
        return hook

    for name, module in model.named_modules():
        if 'attention' in name.lower() or 'attn' in name.lower():
            if layer_name is None or layer_name in name:
                handles.append(module.register_forward_hook(make_hook(name)))

    # Forward pass
    x_tensor = torch.FloatTensor(x)
    if x_tensor.dim() == 1:
        x_tensor = x_tensor.unsqueeze(0)

    with torch.no_grad():
        _ = model(x_tensor)

    # Remove hooks
    for handle in handles:
        handle.remove()

    return attention_maps


def feature_importance_deep(
    model: nn.Module,
    X: np.ndarray,
    y: np.ndarray,
    method: str = 'gradient',
    n_samples: int = 100
) -> np.ndarray:
    """
    Compute feature importance using gradient-based methods.

    Parameters
    ----------
    model : nn.Module
        Trained model
    X : np.ndarray
        Input data
    y : np.ndarray
        True labels
    method : str
        'gradient', 'integrated', or 'occlusion'
    n_samples : int
        Number of samples to use

    Returns
    -------
    np.ndarray
        Feature importance scores
    """
    if not HAS_TORCH:
        raise ImportError("PyTorch required")

    model.eval()

    # Sample data
    if len(X) > n_samples:
        idx = np.random.choice(len(X), n_samples, replace=False)
        X_sample = X[idx]
        y_sample = y[idx]
    else:
        X_sample = X
        y_sample = y

    if method == 'gradient':
        # Simple gradient importance
        X_tensor = torch.FloatTensor(X_sample)
        X_tensor.requires_grad_(True)

        outputs = model(X_tensor)

        # Get gradients for correct class predictions
        importances = np.zeros(X_sample.shape[1:])

        for i in range(len(X_sample)):
            model.zero_grad()
            loss = outputs[i, y_sample[i]]
            loss.backward(retain_graph=True)

            grad = X_tensor.grad[i].abs().cpu().numpy()
            importances += grad

        importances /= len(X_sample)

    elif method == 'integrated':
        ig = IntegratedGradients(model)
        importances = np.zeros(X_sample.shape[1:])

        for i in range(len(X_sample)):
            attr = ig(X_sample[i:i+1], target_class=int(y_sample[i]))
            importances += np.abs(attr)

        importances /= len(X_sample)

    elif method == 'occlusion':
        # Occlusion sensitivity
        importances = _occlusion_sensitivity(model, X_sample, y_sample)

    else:
        raise ValueError(f"Unknown method: {method}")

    return importances


def _occlusion_sensitivity(
    model: nn.Module,
    X: np.ndarray,
    y: np.ndarray,
    occlusion_value: float = 0.0
) -> np.ndarray:
    """Compute importance via occlusion."""
    model.eval()
    X_tensor = torch.FloatTensor(X)

    # Get baseline predictions
    with torch.no_grad():
        baseline_outputs = model(X_tensor)
        baseline_probs = F.softmax(baseline_outputs, dim=1)

    n_features = np.prod(X.shape[1:])
    importances = np.zeros(X.shape[1:])
    flat_shape = X.shape[1:]

    for i in range(n_features):
        # Occlude feature i
        X_occluded = X.copy()
        idx = np.unravel_index(i, flat_shape)
        X_occluded[(slice(None),) + idx] = occlusion_value

        X_occ_tensor = torch.FloatTensor(X_occluded)

        with torch.no_grad():
            outputs = model(X_occ_tensor)
            probs = F.softmax(outputs, dim=1)

        # Importance = drop in correct class probability
        for j in range(len(X)):
            drop = baseline_probs[j, y[j]] - probs[j, y[j]]
            importances[idx] += drop.item()

    importances /= len(X)
    return importances


def layer_activations(
    model: nn.Module,
    x: np.ndarray,
    layer_names: Optional[List[str]] = None
) -> Dict[str, np.ndarray]:
    """
    Extract activations from specified layers.

    Parameters
    ----------
    model : nn.Module
        Model to analyze
    x : np.ndarray
        Input data
    layer_names : List[str], optional
        Layers to extract (default: all)

    Returns
    -------
    Dict[str, np.ndarray]
        Activations per layer
    """
    if not HAS_TORCH:
        raise ImportError("PyTorch required")

    model.eval()
    activations = {}
    handles = []

    def make_hook(name):
        def hook(module, input, output):
            if isinstance(output, torch.Tensor):
                activations[name] = output.detach().cpu().numpy()
        return hook

    for name, module in model.named_modules():
        if layer_names is None or name in layer_names:
            if len(list(module.children())) == 0:  # Leaf modules only
                handles.append(module.register_forward_hook(make_hook(name)))

    # Forward pass
    x_tensor = torch.FloatTensor(x)
    if x_tensor.dim() == 1:
        x_tensor = x_tensor.unsqueeze(0)

    with torch.no_grad():
        _ = model(x_tensor)

    for handle in handles:
        handle.remove()

    return activations


def visualize_filters(
    model: nn.Module,
    layer_name: str,
    n_filters: int = 16
) -> np.ndarray:
    """
    Visualize convolutional filters.

    Parameters
    ----------
    model : nn.Module
        Model with convolutional layers
    layer_name : str
        Name of conv layer
    n_filters : int
        Number of filters to visualize

    Returns
    -------
    np.ndarray
        Filter weights
    """
    if not HAS_TORCH:
        raise ImportError("PyTorch required")

    for name, module in model.named_modules():
        if name == layer_name and isinstance(module, nn.Conv2d):
            weights = module.weight.detach().cpu().numpy()
            # Return first n_filters
            return weights[:min(n_filters, len(weights))]

    raise ValueError(f"Conv layer '{layer_name}' not found")
