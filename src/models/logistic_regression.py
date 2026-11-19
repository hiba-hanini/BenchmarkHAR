from pathlib import Path

import joblib
import numpy as np

from sklearn.linear_model import LogisticRegression as SklearnLogisticRegression
from sklearn.preprocessing import StandardScaler

from src.models.base_model import BaseHARModel
from src.utils.reproducibility import DEFAULT_SEED


class LogisticRegression(BaseHARModel):

    def __init__(self, config):
        super().__init__(config)

        params = self.config.get("params", {})

        default_params = dict(
            class_weight="balanced",
        )

        default_params.update(params)
        default_params.setdefault(
            "random_state", self.config.get("seed", DEFAULT_SEED)
        )

        # Optional standardization
        self.standardize = self.config.get("standardize", False)

        self.scaler = StandardScaler() if self.standardize else None

        self.model = SklearnLogisticRegression(**default_params)

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "LogisticRegression":

        # Fit scaler ONLY on training data
        if self.standardize:
            X_train = self.scaler.fit_transform(X_train)

        self.model.fit(X_train, y_train)

        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:

        self._check_is_fitted()

        # Apply training-fitted scaler
        if self.standardize:
            X = self.scaler.transform(X)

        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:

        self._check_is_fitted()

        if self.standardize:
            X = self.scaler.transform(X)

        return self.model.predict_proba(X)

    def feature_importances(self) -> np.ndarray:

        self._check_is_fitted()

        return np.abs(self.model.coef_).mean(axis=0)

    def save(self, path: str | Path) -> None:

        self._check_is_fitted()

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        artifact = {
            "model": self.model,
            "scaler": self.scaler,
            "standardize": self.standardize,
        }

        joblib.dump(artifact, path)

    def load(self, path: str | Path) -> "LogisticRegression":

        artifact = joblib.load(path)

        # Support both new and legacy model artifacts
        if isinstance(artifact, dict) and "model" in artifact:
            self.model = artifact["model"]
            self.scaler = artifact.get("scaler")
            self.standardize = artifact.get("standardize", False)

        else:
            # Legacy artifact: only the classifier was saved
            self.model = artifact
            self.scaler = None
            self.standardize = False

        self.is_fitted = True
        return self