import numpy as np


from src.models.base_model import BaseHARModel
from src.models.deep.architectures import MODEL_BUILDERS
from pathlib import Path
from tensorflow import keras
import tensorflow as tf

import os


class KerasHARModel(BaseHARModel):
    def __init__(self, config):
        super().__init__(config)
        params = self.config.get("params", {})
        self.model_name = config["name"]
        self.epochs = params.get("epochs", 100)
        self.batch_size = params.get("batch_size", 64)
        self.model = None


    def fit(self, X_train, y_train, n_classes, verbose=0, validation_data=None, callbacks=None):
        self.model = MODEL_BUILDERS[self.model_name](
            X_train[0].shape,
            n_classes
        )

        if validation_data is not None:
            X_val, y_val = validation_data

        fit_kwargs = {
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "verbose": verbose,
        }
        if validation_data is not None:
            fit_kwargs["validation_data"] = validation_data
        if callbacks is not None:
            fit_kwargs["callbacks"] = callbacks

        history = self.model.fit(
            X_train,
            y_train,
            **fit_kwargs,
        )

        self.is_fitted = True
        return history

    
    def predict(self, X_test):
        return self.model.predict(X_test, verbose=0).argmax(axis=1)
    
    def save(self, path: str | Path) -> None:
        self._check_is_fitted()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save(path)

    def load(self, path: str | Path) -> "KerasHARModel":
        self.model = keras.models.load_model(path)
        self.is_fitted = True
        return self