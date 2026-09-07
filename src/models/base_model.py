"""
base_model.py

Abstract interface common to all benchmark models (classical and deep).
Each concrete model (RandomForest, CNN, LSTM, ...) inherits from BaseHARModel
and implements fit/predict/save/load.

Keeping this interface intentionally minimal and framework-independent
(sklearn, torch, etc.) allows `run.py` and `evaluation/evaluator.py` (to be implemented)
to treat all models in the same way, without if/else by model type.
"""
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np


class BaseHARModel(ABC):
    """
    - Input :
        - X : np.ndarray shape (n_samples, n_features) for ML models
              or (n_samples, n_timesteps, n_channels) for DL models
        - y : np.ndarray of shape (n_samples,), labels already encoded as integers
              (label -> int encoding is handled upstream, outside the model,
              to maintain consistency across all benchmark models).
    """

    def __init__(self, config):
        self.config = config or {}
        self.is_fitted: bool = False

    @abstractmethod
    def fit(self, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray = None, y_val: np.ndarray = None) -> "BaseHARModel": # type: ignore
        """Trains the model. Must return self to allow chaining."""
        raise NotImplementedError

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Returns predicted labels, shape (n_samples,)."""
        raise NotImplementedError

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Returns class probabilities, shape (n_samples, n_classes).
        Optional: not all models natively support it (e.g., some SVMs).
        Subclasses that do not support it can leave the default implementation,
        which raises NotImplementedError.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support predict_proba."
        )

    @abstractmethod
    def save(self, path: str | Path) -> None:
        """Saves the trained model to disk."""
        raise NotImplementedError

    @abstractmethod
    def load(self, path: str | Path) -> "BaseHARModel":
        """Loads the trained model from disk. Must return self to allow chaining."""
        raise NotImplementedError

    def _check_is_fitted(self) -> None:
        if not self.is_fitted:
            raise RuntimeError(
                f"{self.__class__.__name__} is not yet fitted. "
                f"Call .fit(X_train, y_train) before .predict()."
            )