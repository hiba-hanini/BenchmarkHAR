from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from src.models.base_model import BaseHARModel
from src.utils.reproducibility import DEFAULT_SEED


class RandomForestModel(BaseHARModel):

    def __init__(self, config):
        super().__init__(config)

        params = self.config.get("params", {})

        # if params is empty, we use default parameters
        default_params = dict(
            n_estimators=200,
            max_depth=None,
            max_features="sqrt",
            class_weight="balanced",
            n_jobs=-1,
        )

        # update default_params with params, if params is not empty
        default_params.update(params)
        default_params.setdefault("random_state", self.config.get("seed", DEFAULT_SEED))

        self.model = RandomForestClassifier(**default_params)

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "RandomForestModel":
        self.model.fit(X_train, y_train)
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        self._check_is_fitted()
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self._check_is_fitted()
        return self.model.predict_proba(X)

    def feature_importances(self) -> np.ndarray:
        self._check_is_fitted()
        return self.model.feature_importances_

    def save(self, path: str | Path) -> None:
        self._check_is_fitted()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, path)

    def load(self, path: str | Path) -> "RandomForestModel":
        self.model = joblib.load(path)
        self.is_fitted = True
        return self
