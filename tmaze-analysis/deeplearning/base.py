"""
Base classes for deep learning decoders.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


@dataclass
class DeepDecodingResult:
    """Container for deep learning classification results."""
    accuracy: float
    loss: float
    predictions: np.ndarray
    probabilities: np.ndarray
    true_labels: np.ndarray

    # Training history
    train_losses: Optional[List[float]] = None
    val_losses: Optional[List[float]] = None
    train_accuracies: Optional[List[float]] = None
    val_accuracies: Optional[List[float]] = None

    # Cross-validation
    cv_accuracies: Optional[np.ndarray] = None
    cv_fold: Optional[int] = None

    # Model info
    model_name: str = ""
    n_parameters: int = 0
    training_time: float = 0.0

    metadata: Dict = field(default_factory=dict)

    def __repr__(self):
        return (f"DeepDecodingResult({self.model_name}: "
                f"accuracy={self.accuracy:.1%}, loss={self.loss:.4f})")

    @property
    def accuracy_std(self) -> Optional[float]:
        if self.cv_accuracies is not None:
            return np.std(self.cv_accuracies)
        return None

    def to_dict(self) -> Dict:
        """Convert to dictionary for saving."""
        return {
            'accuracy': self.accuracy,
            'loss': self.loss,
            'predictions': self.predictions.tolist(),
            'probabilities': self.probabilities.tolist(),
            'true_labels': self.true_labels.tolist(),
            'cv_accuracies': self.cv_accuracies.tolist() if self.cv_accuracies is not None else None,
            'model_name': self.model_name,
            'n_parameters': self.n_parameters,
            'training_time': self.training_time,
            'metadata': self.metadata
        }


class BaseDeepDecoder(ABC):
    """
    Abstract base class for deep learning decoders.

    Provides common interface for EEG, fMRI, and multimodal models.
    """

    def __init__(
        self,
        n_classes: int = 2,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        dropout: float = 0.5,
        device: Optional[str] = None
    ):
        if not HAS_TORCH:
            raise ImportError("PyTorch required for deep learning models")

        self.n_classes = n_classes
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.dropout = dropout

        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        self.model = None
        self.optimizer = None
        self.criterion = None
        self.is_fitted = False

    @abstractmethod
    def build_model(self, input_shape: Tuple) -> nn.Module:
        """Build the neural network architecture."""
        pass

    def compile(
        self,
        input_shape: Tuple,
        optimizer: str = 'adam',
        loss: str = 'cross_entropy'
    ):
        """
        Compile the model with optimizer and loss function.

        Parameters
        ----------
        input_shape : Tuple
            Shape of input data (excluding batch dimension)
        optimizer : str
            'adam', 'sgd', or 'adamw'
        loss : str
            'cross_entropy' or 'focal'
        """
        self.model = self.build_model(input_shape).to(self.device)

        # Optimizer
        if optimizer == 'adam':
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay
            )
        elif optimizer == 'adamw':
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay
            )
        elif optimizer == 'sgd':
            self.optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
                momentum=0.9
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer}")

        # Loss function
        if loss == 'cross_entropy':
            self.criterion = nn.CrossEntropyLoss()
        elif loss == 'focal':
            self.criterion = FocalLoss(gamma=2.0)
        else:
            raise ValueError(f"Unknown loss: {loss}")

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        epochs: int = 100,
        batch_size: int = 32,
        verbose: bool = True
    ) -> Dict[str, List[float]]:
        """
        Train the model.

        Parameters
        ----------
        X : np.ndarray
            Training data
        y : np.ndarray
            Training labels
        X_val, y_val : np.ndarray, optional
            Validation data
        epochs : int
            Number of training epochs
        batch_size : int
            Batch size
        verbose : bool
            Print progress

        Returns
        -------
        Dict
            Training history
        """
        if self.model is None:
            self.compile(X.shape[1:])

        # Convert to tensors
        X_train = torch.FloatTensor(X).to(self.device)
        y_train = torch.LongTensor(y).to(self.device)

        if X_val is not None:
            X_val = torch.FloatTensor(X_val).to(self.device)
            y_val = torch.LongTensor(y_val).to(self.device)

        # Training loop
        history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }

        dataset = torch.utils.data.TensorDataset(X_train, y_train)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True
        )

        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0

            for batch_X, batch_y in loader:
                self.optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item() * batch_X.size(0)
                _, predicted = outputs.max(1)
                train_correct += predicted.eq(batch_y).sum().item()
                train_total += batch_y.size(0)

            train_loss /= train_total
            train_acc = train_correct / train_total
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)

            # Validation phase
            if X_val is not None:
                self.model.eval()
                with torch.no_grad():
                    val_outputs = self.model(X_val)
                    val_loss = self.criterion(val_outputs, y_val).item()
                    _, val_predicted = val_outputs.max(1)
                    val_acc = val_predicted.eq(y_val).float().mean().item()

                history['val_loss'].append(val_loss)
                history['val_acc'].append(val_acc)

                if verbose and (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch+1}/{epochs}: "
                          f"train_loss={train_loss:.4f}, train_acc={train_acc:.3f}, "
                          f"val_loss={val_loss:.4f}, val_acc={val_acc:.3f}")
            elif verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}: "
                      f"train_loss={train_loss:.4f}, train_acc={train_acc:.3f}")

        self.is_fitted = True
        return history

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        if not self.is_fitted:
            raise ValueError("Model not fitted")

        self.model.eval()
        X_tensor = torch.FloatTensor(X).to(self.device)

        with torch.no_grad():
            outputs = self.model(X_tensor)
            _, predicted = outputs.max(1)

        return predicted.cpu().numpy()

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        if not self.is_fitted:
            raise ValueError("Model not fitted")

        self.model.eval()
        X_tensor = torch.FloatTensor(X).to(self.device)

        with torch.no_grad():
            outputs = self.model(X_tensor)
            proba = torch.softmax(outputs, dim=1)

        return proba.cpu().numpy()

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
        """Evaluate model on test data."""
        if not self.is_fitted:
            raise ValueError("Model not fitted")

        self.model.eval()
        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.LongTensor(y).to(self.device)

        with torch.no_grad():
            outputs = self.model(X_tensor)
            loss = self.criterion(outputs, y_tensor).item()
            _, predicted = outputs.max(1)
            accuracy = predicted.eq(y_tensor).float().mean().item()

        return accuracy, loss

    def n_parameters(self) -> int:
        """Count trainable parameters."""
        if self.model is None:
            return 0
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def save(self, path: str):
        """Save model weights."""
        if self.model is not None:
            torch.save(self.model.state_dict(), path)

    def load(self, path: str, input_shape: Tuple):
        """Load model weights."""
        self.compile(input_shape)
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.is_fitted = True


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance."""

    def __init__(self, gamma: float = 2.0, alpha: Optional[float] = None):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss

        if self.alpha is not None:
            focal_loss = self.alpha * focal_loss

        return focal_loss.mean()
