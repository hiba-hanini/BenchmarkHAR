import numpy as np
import pandas as pd

from src.models.deep.architectures import build_dnn, standardize_data
from src.models.run_model_dl import _load_dnn_features, _select_dnn_features


def test_dnn_loads_the_cached_statistical_feature_matrix(tmp_path, monkeypatch):
    features_dir = tmp_path / "data" / "features"
    features_dir.mkdir(parents=True)
    pd.DataFrame({
        "subject_id": ["s1", "s1", "s2"],
        "activity": ["walk", "sit", "walk"],
        "label": ["walk", "sit", "walk"],
        "mean": [1.0, 2.0, 3.0],
        "std": [0.1, 0.2, 0.3],
    }).to_pickle(features_dir / "demo_features_all.pkl")
    monkeypatch.chdir(tmp_path)

    X, y, groups = _load_dnn_features(
        "demo", {"label_to_predict": "activity", "label_cols": ["activity", "label"]},
    )

    assert list(X.columns) == ["mean", "std"]
    np.testing.assert_array_equal(y, ["walk", "sit", "walk"])
    np.testing.assert_array_equal(groups, ["s1", "s1", "s2"])


def test_dnn_filters_using_training_features_and_uses_selected_input_size():
    X_train = pd.DataFrame({
        "a": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        "b": [2.0, 4.0, 6.0, 8.0, 10.0, 12.0],
        "c": [0.0, 1.0, 0.0, 1.0, 1.0, 0.0],
    })
    X_test = pd.DataFrame({
        "a": [10.0, 20.0],
        "b": [-100.0, 1.0],
        "c": [1.0, 0.0],
    })

    train_selected, test_selected, selector = _select_dnn_features(
        X_train, X_test, threshold=0.95,
    )

    assert selector.n_fit_samples == len(X_train)
    assert selector.selected_feature_names == ("a", "c")
    assert train_selected.shape == (len(X_train), 2)
    assert test_selected.shape == (len(X_test), 2)

    train_scaled, test_scaled, scaler = standardize_data("dnn", train_selected, test_selected)
    assert train_scaled.shape == train_selected.shape
    assert test_scaled.shape == test_selected.shape
    np.testing.assert_allclose(scaler.mean_, train_selected.mean(axis=0))

    model = build_dnn((train_selected.shape[1],), n_classes=2)
    assert model.input_shape == (None, len(selector.selected_feature_names))
    model.fit(train_scaled, np.array([0, 0, 1, 1, 0, 1]), epochs=1, batch_size=2, verbose=0)
    assert model.predict(test_scaled, verbose=0).shape == (len(X_test), 2)
