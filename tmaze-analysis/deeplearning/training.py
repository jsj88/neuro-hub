"""
Training infrastructure for deep learning models.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Callable, Any
import time
import numpy as np

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from .base import BaseDeepDecoder, DeepDecodingResult


class EarlyStopping:
    """
    Early stopping to prevent overfitting.

    Parameters
    ----------
    patience : int
        Number of epochs without improvement before stopping
    min_delta : float
        Minimum change to qualify as improvement
    mode : str
        'min' for loss, 'max' for accuracy
    restore_best : bool
        Restore best weights when stopping
    """

    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.001,
        mode: str = 'min',
        restore_best: bool = True
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.restore_best = restore_best

        self.counter = 0
        self.best_score = None
        self.best_weights = None
        self.should_stop = False

    def __call__(self, score: float, model: nn.Module) -> bool:
        """
        Check if training should stop.

        Parameters
        ----------
        score : float
            Current validation metric
        model : nn.Module
            Model to potentially save weights from

        Returns
        -------
        bool
            True if training should stop
        """
        if self.best_score is None:
            self.best_score = score
            self._save_checkpoint(model)
            return False

        if self.mode == 'min':
            improved = score < self.best_score - self.min_delta
        else:
            improved = score > self.best_score + self.min_delta

        if improved:
            self.best_score = score
            self._save_checkpoint(model)
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
                if self.restore_best and self.best_weights is not None:
                    model.load_state_dict(self.best_weights)

        return self.should_stop

    def _save_checkpoint(self, model: nn.Module):
        """Save model weights."""
        self.best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}


class LRScheduler:
    """
    Learning rate scheduler wrapper.

    Parameters
    ----------
    scheduler_type : str
        'step', 'cosine', 'plateau', or 'warmup_cosine'
    **kwargs
        Scheduler-specific parameters
    """

    def __init__(self, scheduler_type: str = 'cosine', **kwargs):
        self.scheduler_type = scheduler_type
        self.kwargs = kwargs
        self.scheduler = None

    def create(self, optimizer: torch.optim.Optimizer, n_epochs: int):
        """Create the scheduler."""
        if self.scheduler_type == 'step':
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=self.kwargs.get('step_size', 30),
                gamma=self.kwargs.get('gamma', 0.1)
            )
        elif self.scheduler_type == 'cosine':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=n_epochs,
                eta_min=self.kwargs.get('eta_min', 1e-6)
            )
        elif self.scheduler_type == 'plateau':
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode=self.kwargs.get('mode', 'min'),
                factor=self.kwargs.get('factor', 0.5),
                patience=self.kwargs.get('patience', 5)
            )
        elif self.scheduler_type == 'warmup_cosine':
            warmup_epochs = self.kwargs.get('warmup_epochs', 5)
            self.scheduler = _WarmupCosineScheduler(
                optimizer, warmup_epochs, n_epochs
            )
        else:
            self.scheduler = None

    def step(self, val_loss: Optional[float] = None):
        """Step the scheduler."""
        if self.scheduler is None:
            return

        if self.scheduler_type == 'plateau':
            self.scheduler.step(val_loss)
        else:
            self.scheduler.step()


class _WarmupCosineScheduler:
    """Warmup + Cosine annealing scheduler."""

    def __init__(self, optimizer, warmup_epochs: int, total_epochs: int):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        self.current_epoch = 0

    def step(self):
        self.current_epoch += 1

        if self.current_epoch <= self.warmup_epochs:
            # Linear warmup
            scale = self.current_epoch / self.warmup_epochs
        else:
            # Cosine annealing
            progress = (self.current_epoch - self.warmup_epochs) / (
                self.total_epochs - self.warmup_epochs
            )
            scale = 0.5 * (1 + np.cos(np.pi * progress))

        for param_group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            param_group['lr'] = base_lr * scale


class DeepTrainer:
    """
    Training manager for deep learning models.

    Parameters
    ----------
    model : BaseDeepDecoder
        Model to train
    device : str
        'cuda' or 'cpu'
    mixed_precision : bool
        Use automatic mixed precision
    """

    def __init__(
        self,
        model: BaseDeepDecoder,
        device: Optional[str] = None,
        mixed_precision: bool = False
    ):
        if not HAS_TORCH:
            raise ImportError("PyTorch required")

        self.model = model

        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        self.mixed_precision = mixed_precision
        if mixed_precision and torch.cuda.is_available():
            self.scaler = torch.cuda.amp.GradScaler()
        else:
            self.scaler = None

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        epochs: int = 100,
        batch_size: int = 32,
        early_stopping: Optional[EarlyStopping] = None,
        lr_scheduler: Optional[LRScheduler] = None,
        verbose: bool = True
    ) -> DeepDecodingResult:
        """
        Train the model.

        Parameters
        ----------
        X_train, y_train : np.ndarray
            Training data
        X_val, y_val : np.ndarray, optional
            Validation data
        epochs : int
            Number of training epochs
        batch_size : int
            Batch size
        early_stopping : EarlyStopping, optional
            Early stopping callback
        lr_scheduler : LRScheduler, optional
            Learning rate scheduler
        verbose : bool
            Print progress

        Returns
        -------
        DeepDecodingResult
        """
        start_time = time.time()

        # Compile model if needed
        if self.model.model is None:
            self.model.compile(X_train.shape[1:])

        self.model.model.to(self.device)

        # Create data loaders
        train_loader = self._create_loader(X_train, y_train, batch_size, shuffle=True)

        if X_val is not None:
            val_loader = self._create_loader(X_val, y_val, batch_size, shuffle=False)
        else:
            val_loader = None

        # Setup scheduler
        if lr_scheduler is not None:
            lr_scheduler.create(self.model.optimizer, epochs)

        # Training history
        history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'lr': []
        }

        # Training loop
        for epoch in range(epochs):
            # Training phase
            train_loss, train_acc = self._train_epoch(train_loader)
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)

            # Validation phase
            if val_loader is not None:
                val_loss, val_acc = self._validate(val_loader)
                history['val_loss'].append(val_loss)
                history['val_acc'].append(val_acc)
            else:
                val_loss, val_acc = None, None

            # Learning rate
            current_lr = self.model.optimizer.param_groups[0]['lr']
            history['lr'].append(current_lr)

            # Scheduler step
            if lr_scheduler is not None:
                lr_scheduler.step(val_loss)

            # Early stopping
            if early_stopping is not None and val_loss is not None:
                if early_stopping(val_loss, self.model.model):
                    if verbose:
                        print(f"Early stopping at epoch {epoch + 1}")
                    break

            # Logging
            if verbose and (epoch + 1) % 10 == 0:
                msg = f"Epoch {epoch + 1}/{epochs}: train_loss={train_loss:.4f}, train_acc={train_acc:.3f}"
                if val_loss is not None:
                    msg += f", val_loss={val_loss:.4f}, val_acc={val_acc:.3f}"
                print(msg)

        training_time = time.time() - start_time

        # Final evaluation
        self.model.is_fitted = True

        if X_val is not None:
            predictions = self.model.predict(X_val)
            probabilities = self.model.predict_proba(X_val)
            final_acc = np.mean(predictions == y_val)
            final_loss = history['val_loss'][-1]
            true_labels = y_val
        else:
            predictions = self.model.predict(X_train)
            probabilities = self.model.predict_proba(X_train)
            final_acc = np.mean(predictions == y_train)
            final_loss = history['train_loss'][-1]
            true_labels = y_train

        return DeepDecodingResult(
            accuracy=final_acc,
            loss=final_loss,
            predictions=predictions,
            probabilities=probabilities,
            true_labels=true_labels,
            train_losses=history['train_loss'],
            val_losses=history['val_loss'] if history['val_loss'] else None,
            train_accuracies=history['train_acc'],
            val_accuracies=history['val_acc'] if history['val_acc'] else None,
            model_name=self.model.__class__.__name__,
            n_parameters=self.model.n_parameters(),
            training_time=training_time
        )

    def _train_epoch(self, loader: DataLoader) -> Tuple[float, float]:
        """Train for one epoch."""
        self.model.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for batch_x, batch_y in loader:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)

            self.model.optimizer.zero_grad()

            if self.scaler is not None:
                with torch.cuda.amp.autocast():
                    outputs = self.model.model(batch_x)
                    loss = self.model.criterion(outputs, batch_y)

                self.scaler.scale(loss).backward()
                self.scaler.step(self.model.optimizer)
                self.scaler.update()
            else:
                outputs = self.model.model(batch_x)
                loss = self.model.criterion(outputs, batch_y)
                loss.backward()
                self.model.optimizer.step()

            total_loss += loss.item() * batch_x.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(batch_y).sum().item()
            total += batch_y.size(0)

        return total_loss / total, correct / total

    def _validate(self, loader: DataLoader) -> Tuple[float, float]:
        """Validate the model."""
        self.model.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch_x, batch_y in loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                outputs = self.model.model(batch_x)
                loss = self.model.criterion(outputs, batch_y)

                total_loss += loss.item() * batch_x.size(0)
                _, predicted = outputs.max(1)
                correct += predicted.eq(batch_y).sum().item()
                total += batch_y.size(0)

        return total_loss / total, correct / total

    def _create_loader(
        self,
        X: np.ndarray,
        y: np.ndarray,
        batch_size: int,
        shuffle: bool
    ) -> DataLoader:
        """Create a data loader."""
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.LongTensor(y)
        dataset = TensorDataset(X_tensor, y_tensor)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def cross_validate_deep(
    model_class: type,
    X: np.ndarray,
    y: np.ndarray,
    subjects: Optional[np.ndarray] = None,
    n_folds: int = 5,
    epochs: int = 100,
    batch_size: int = 32,
    early_stopping: bool = True,
    verbose: bool = True,
    **model_kwargs
) -> DeepDecodingResult:
    """
    Cross-validate a deep learning model.

    Parameters
    ----------
    model_class : type
        Model class to instantiate
    X : np.ndarray
        Features
    y : np.ndarray
        Labels
    subjects : np.ndarray, optional
        Subject IDs for leave-one-subject-out CV
    n_folds : int
        Number of CV folds
    epochs : int
        Training epochs per fold
    batch_size : int
        Batch size
    early_stopping : bool
        Use early stopping
    verbose : bool
        Print progress
    **model_kwargs
        Arguments for model constructor

    Returns
    -------
    DeepDecodingResult
        Aggregated results across folds
    """
    if subjects is not None:
        # Leave-one-subject-out
        unique_subjects = np.unique(subjects)
        n_folds = len(unique_subjects)

        fold_results = []
        all_predictions = np.zeros(len(y))
        all_probabilities = np.zeros((len(y), len(np.unique(y))))

        for fold, test_subj in enumerate(unique_subjects):
            if verbose:
                print(f"\nFold {fold + 1}/{n_folds} (Subject {test_subj})")

            train_mask = subjects != test_subj
            test_mask = subjects == test_subj

            X_train, X_test = X[train_mask], X[test_mask]
            y_train, y_test = y[train_mask], y[test_mask]

            # Create and train model
            model = model_class(**model_kwargs)
            trainer = DeepTrainer(model)

            es = EarlyStopping(patience=10) if early_stopping else None

            result = trainer.train(
                X_train, y_train,
                X_val=X_test, y_val=y_test,
                epochs=epochs,
                batch_size=batch_size,
                early_stopping=es,
                verbose=verbose
            )

            fold_results.append(result.accuracy)
            all_predictions[test_mask] = result.predictions
            all_probabilities[test_mask] = result.probabilities

    else:
        # Stratified k-fold
        from sklearn.model_selection import StratifiedKFold

        kfold = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

        fold_results = []
        all_predictions = np.zeros(len(y))
        all_probabilities = np.zeros((len(y), len(np.unique(y))))

        for fold, (train_idx, test_idx) in enumerate(kfold.split(X, y)):
            if verbose:
                print(f"\nFold {fold + 1}/{n_folds}")

            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            model = model_class(**model_kwargs)
            trainer = DeepTrainer(model)

            es = EarlyStopping(patience=10) if early_stopping else None

            result = trainer.train(
                X_train, y_train,
                X_val=X_test, y_val=y_test,
                epochs=epochs,
                batch_size=batch_size,
                early_stopping=es,
                verbose=verbose
            )

            fold_results.append(result.accuracy)
            all_predictions[test_idx] = result.predictions
            all_probabilities[test_idx] = result.probabilities

    # Aggregate results
    cv_accuracies = np.array(fold_results)

    return DeepDecodingResult(
        accuracy=np.mean(cv_accuracies),
        loss=0.0,  # Not tracked across folds
        predictions=all_predictions.astype(int),
        probabilities=all_probabilities,
        true_labels=y,
        cv_accuracies=cv_accuracies,
        model_name=model_class.__name__,
        metadata={'n_folds': n_folds, 'epochs': epochs}
    )


def train_test_split_subjects(
    X: np.ndarray,
    y: np.ndarray,
    subjects: np.ndarray,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split data by subject (no subject appears in both train and test).

    Parameters
    ----------
    X : np.ndarray
        Features
    y : np.ndarray
        Labels
    subjects : np.ndarray
        Subject IDs
    test_size : float
        Fraction of subjects for test set
    random_state : int
        Random seed

    Returns
    -------
    X_train, X_test, y_train, y_test : np.ndarray
    """
    np.random.seed(random_state)

    unique_subjects = np.unique(subjects)
    n_test = max(1, int(len(unique_subjects) * test_size))

    test_subjects = np.random.choice(unique_subjects, n_test, replace=False)
    test_mask = np.isin(subjects, test_subjects)

    X_train, X_test = X[~test_mask], X[test_mask]
    y_train, y_test = y[~test_mask], y[test_mask]

    return X_train, X_test, y_train, y_test
