import json
import random
from pathlib import Path

import numpy as np

from src.models.random_forest import RandomForestModel
from src.utils.cv import get_cv_splitter
from src.utils.reproducibility import (
    build_experiment_manifest,
    compute_experiment_id,
    get_result_path,
    manifest_matches,
    set_global_seed,
)


def test_python_and_numpy_rngs_are_reproducible():
    set_global_seed(17)
    first_python = [random.random() for _ in range(4)]
    first_numpy = np.random.random(4)

    set_global_seed(17)
    assert [random.random() for _ in range(4)] == first_python
    np.testing.assert_array_equal(np.random.random(4), first_numpy)


def _splits(seed):
    groups = np.repeat(np.arange(20), 3)
    splitter, name = get_cv_splitter(20, {"n_splits": 3, "test_size": 0.2}, seed=seed)
    assert name == "GroupShuffleSplit"
    return [(train.tolist(), test.tolist()) for train, test in splitter.split(groups, groups, groups)]


def test_group_shuffle_split_uses_experiment_seed():
    assert _splits(42) == _splits(42)
    assert _splits(42) != _splits(123)


def test_sklearn_model_inherits_experiment_seed():
    first = RandomForestModel({"seed": 9, "params": {"n_estimators": 5}})
    second = RandomForestModel({"seed": 9, "params": {"n_estimators": 5}})
    assert first.model.get_params()["random_state"] == second.model.get_params()["random_state"] == 9

    X = np.array([[0], [1], [2], [3], [4], [5]])
    y = np.array([0, 0, 0, 1, 1, 1])
    first.fit(X, y)
    second.fit(X, y)
    np.testing.assert_array_equal(first.predict(X), second.predict(X))


def test_experiment_id_is_stable_for_same_config_and_seed():
    base = {
        "dataset": "pamap2",
        "model": "random_forest",
        "seed": 42,
        "deterministic": True,
        "config_data": {"preprocessing": {"window_size": 128}, "label_to_predict": "activity"},
        "config_model": {"params": {"n_estimators": 25}, "cv": {"n_splits": 5, "test_size": 0.2}},
    }
    assert compute_experiment_id(base) == compute_experiment_id(base)


def test_experiment_id_changes_across_reproducibility_inputs():
    base = {
        "dataset": "pamap2",
        "model": "random_forest",
        "seed": 42,
        "deterministic": True,
        "config_data": {"preprocessing": {"window_size": 128}, "label_to_predict": "activity"},
        "config_model": {"params": {"n_estimators": 25}, "cv": {"n_splits": 5, "test_size": 0.2}},
    }
    changed_seed = json.loads(json.dumps(base))
    changed_seed["seed"] = 43
    changed_model = json.loads(json.dumps(base))
    changed_model["config_model"] = {"params": {"n_estimators": 50}, "cv": {"n_splits": 5, "test_size": 0.2}}
    changed_preprocessing = json.loads(json.dumps(base))
    changed_preprocessing["config_data"] = {"preprocessing": {"window_size": 64}, "label_to_predict": "activity"}

    assert compute_experiment_id(base) != compute_experiment_id(changed_seed)
    assert compute_experiment_id(base) != compute_experiment_id(changed_model)
    assert compute_experiment_id(base) != compute_experiment_id(changed_preprocessing)


def test_manifest_accepts_only_matching_cache():
    manifest = build_experiment_manifest(
        dataset="pamap2",
        model="random_forest",
        seed=42,
        deterministic=True,
        resolved_configuration={"preprocessing": {"window_size": 128}, "model": {"n_estimators": 25}},
        artifact_path="data/results/test.pkl",
        artifact_type="results",
    )
    assert manifest_matches(manifest, manifest)

    different = dict(manifest)
    different["seed"] = 43
    assert not manifest_matches(manifest, different)


def test_result_paths_are_seed_specific_and_same_seed_same_cv_groups_are_identical():
    groups = np.repeat(np.arange(8), 2)
    splitter_a, _ = get_cv_splitter(8, {"n_splits": 3, "test_size": 0.25}, seed=11)
    splitter_b, _ = get_cv_splitter(8, {"n_splits": 3, "test_size": 0.25}, seed=11)
    split_a = [tuple((train.tolist(), test.tolist())) for train, test in splitter_a.split(groups, groups, groups)]
    split_b = [tuple((train.tolist(), test.tolist())) for train, test in splitter_b.split(groups, groups, groups)]
    assert split_a == split_b

    path_a = get_result_path("pamap2", "random_forest", {"seed": 11, "deterministic": True, "dataset": "pamap2", "model": "random_forest"})
    path_b = get_result_path("pamap2", "random_forest", {"seed": 12, "deterministic": True, "dataset": "pamap2", "model": "random_forest"})
    assert path_a != path_b

    for train, test in split_a:
        assert set(train).isdisjoint(set(test))


def test_experiment_id_is_not_time_sensitive():
    base = {
        "dataset": "pamap2",
        "model": "random_forest",
        "seed": 42,
        "deterministic": True,
        "config_data": {"preprocessing": {"window_size": 128}},
        "config_model": {"params": {"n_estimators": 25}},
    }
    first = compute_experiment_id(base)
    base["config_data"]["preprocessing"]["window_size"] = 128
    second = compute_experiment_id(base)
    assert first == second
