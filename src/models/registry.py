
from src.models.logistic_regression import LogisticRegression
from src.models.random_forest import RandomForestModel

class _LazyModelRegistry(dict):
    """Dictionary-compatible registry that defers optional framework imports."""

    def __missing__(self, model_name):
        if model_name == "tinyhar":
            from src.models.deep.tinyHAR_wrapper import TinyHARWrapper
            return TinyHARWrapper
        if model_name in {"cnn", "lstm", "dnn", "rnn", "deepconv_lstm", "astar_har"}:
            from src.models.deep.keras_models import KerasHARModel
            return KerasHARModel
        raise KeyError(model_name)

    def get(self, model_name, default=None):
        try:
            return self[model_name]
        except KeyError:
            return default


MODEL_REGISTRY = _LazyModelRegistry({
    "random_forest": RandomForestModel,
    "logistic_regression": LogisticRegression,
})


def get_model_class(model_name):
    """Resolve deep model classes lazily to keep ML-only installs lightweight."""
    return MODEL_REGISTRY.get(model_name)
