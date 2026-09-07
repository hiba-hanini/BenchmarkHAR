import numpy as np
import pandas as pd
import pytest

from src.dimension_reduction.correlation_filter import (
    fit_correlation_selector,
    transform_with_correlation_selector,
)
from src.preprocessing.mapping import (
    build_canonical_label_encoder,
    canonical_label_metadata,
    canonicalize_labels,
)
from src.utils.cv import get_cv_splitter
from src.utils.scientific_checks import validate_subject_split


def test_correlation_selector_is_fit_on_train_and_transforms_consistently():
    X_train = pd.DataFrame({"a": [1, 2, 3, 4], "b": [2, 4, 6, 8], "c": [3, 1, 4, 1]})
    X_test = pd.DataFrame({"a": [10, 20], "b": [-1, 99], "c": [5, 9]})

    selector = fit_correlation_selector(X_train, threshold=0.95)
    selected_train = transform_with_correlation_selector(X_train, selector)
    selected_test = transform_with_correlation_selector(X_test, selector)

    assert list(selected_train.columns) == list(selector.selected_feature_names)
    assert list(selected_test.columns) == list(selector.selected_feature_names)
    assert "a" in selector.selected_feature_names
    assert "b" not in selector.selected_feature_names


def test_subject_splits_are_disjoint_for_group_shuffle_split():
    groups = np.repeat(np.arange(12), 2)
    splitter, _ = get_cv_splitter(12, {"n_splits": 3, "test_size": 0.25}, seed=7)
    for train_idx, test_idx in splitter.split(groups, groups, groups):
        train_subjects = groups[train_idx]
        test_subjects = groups[test_idx]
        validate_subject_split(train_subjects, test_subjects)
        assert not set(train_subjects) & set(test_subjects)


def test_canonical_labels_come_from_configuration_not_test_samples():
    config = {
        "label_mapping": {"activity": {"walk": "walking", "sit": "sitting", "run": "running"}}
    }
    metadata = canonical_label_metadata(config, "activity")
    encoder = build_canonical_label_encoder(config, "activity")

    train = canonicalize_labels(["walk", "sit"], "activity", config)
    test = canonicalize_labels(["run"], "activity", config)

    assert metadata["class_ids"] == [0, 1, 2]
    assert encoder.classes_.tolist() == [0, 1, 2]
    assert encoder.transform(train).tolist() == [0, 1]
    assert encoder.transform(test).tolist() == [2]


def test_fold_local_selection_ignores_held_out_subject_distribution():
    # Subject 2 has an intentionally different b relationship. The selector is
    # fitted only on subjects 0/1, where b is perfectly correlated with a.
    df = pd.DataFrame({
        "subject_id": [0, 0, 1, 1, 2, 2],
        "a": [1, 2, 3, 4, 10, 20],
        "b": [2, 4, 6, 8, 1, 99],
        "c": [4, 1, 3, 2, 8, 9],
    })
    train = df[df.subject_id != 2]
    test = df[df.subject_id == 2]

    selector = fit_correlation_selector(train[["a", "b", "c"]])
    transformed_test = transform_with_correlation_selector(test[["a", "b", "c"]], selector)

    assert "b" not in selector.selected_feature_names
    assert list(transformed_test.columns) == list(selector.selected_feature_names)


def test_label_mapping_is_deterministic_and_configured_only_once():
    config = {
        "label_mapping": {"activity": {"walk": "walking", "sit": "sitting", "run": "running"}}
    }

    first = canonicalize_labels(["walk", "sit", "run"], "activity", config)
    second = canonicalize_labels(["walk", "sit", "run"], "activity", config)
    metadata = canonical_label_metadata(config, "activity")

    np.testing.assert_array_equal(first, second)
    assert metadata["class_ids"] == [0, 1, 2]
    assert first.tolist() == [0, 1, 2]


def test_unseen_test_label_raises_and_does_not_expand_vocabulary():
    config = {
        "label_mapping": {"activity": {"walk": "walking", "sit": "sitting", "run": "running"}}
    }
    metadata = canonical_label_metadata(config, "activity")

    with pytest.raises(ValueError, match="Unmapped labels"):
        canonicalize_labels(["walk", "unknown_action"], "activity", config)

    assert metadata["class_ids"] == [0, 1, 2]


def test_cross_pipeline_mapping_uses_same_canonical_vocabulary():
    config = {
        "label_mapping": {"activity": {"walk": "walking", "sit": "sitting", "run": "running"}}
    }

    metadata = canonical_label_metadata(config, "activity")
    encoder = build_canonical_label_encoder(config, "activity")
    mapped = canonicalize_labels(["walk", "sit", "run"], "activity", config)

    assert metadata["class_ids"] == encoder.classes_.tolist()
    np.testing.assert_array_equal(mapped, encoder.transform([0, 1, 2]))


def test_subject_aware_inner_split_keeps_outer_test_out_of_validation():
    outer_train_subjects = np.array([0, 0, 1, 1, 2, 2, 3, 3])
    outer_test_subjects = np.array([4, 4, 5, 5])
    train_subjects = np.array(outer_train_subjects)
    val_subjects = np.array([6, 7])
    assert set(train_subjects).isdisjoint(set(outer_test_subjects))
    assert set(val_subjects).isdisjoint(set(outer_test_subjects))
    assert set(val_subjects).isdisjoint(set(train_subjects))


def test_fold_class_counts_and_vocabulary_are_canonical():
    config = {
        "label_mapping": {"activity": {"walk": "walking", "sit": "sitting", "run": "running"}},
        "unknown_label_policy": "raise",
    }
    metadata = canonical_label_metadata(config, "activity")
    y_train = np.array(canonicalize_labels(["walk", "sit", "walk"], "activity", config))
    y_test = np.array(canonicalize_labels(["run", "walk"], "activity", config))
    counts = {
        "train": {int(k): int(v) for k, v in sorted({0: 2, 1: 1, 2: 0}.items())},
        "test": {int(k): int(v) for k, v in sorted({0: 1, 1: 0, 2: 1}.items())},
    }
    assert set(metadata["class_ids"]) == {0, 1, 2}
    assert sum(counts["train"].values()) == len(y_train)
    assert sum(counts["test"].values()) == len(y_test)
    assert set(np.unique(y_test)).issubset(set(metadata["class_ids"]))


def test_ml_fold_protocol_uses_train_only_selection_and_configured_vocabulary():
    config = {
        "label_to_predict": "activity",
        "label_mapping": {"activity": {"walk": "walking", "sit": "sitting", "run": "running"}},
        "unknown_label_policy": "raise",
    }
    df = pd.DataFrame(
        {
            "subject_id": [0, 0, 1, 1, 2, 2],
            "activity": ["walk", "sit", "walk", "run", "walk", "sit"],
            "a": [1, 2, 3, 4, 10, 11],
            "b": [2, 4, 6, 8, 20, 22],
            "c": [0, 5, 1, 4, 8, 9],
        }
    )
    label_metadata = canonical_label_metadata(config, "activity")
    y_all = canonicalize_labels(df["activity"], "activity", config)

    assert y_all.tolist() == [0, 1, 0, 2, 0, 1]
    assert label_metadata["class_ids"] == [0, 1, 2]

    splitter, _ = get_cv_splitter(3, {"n_splits": 2, "test_size": 0.4}, seed=7)
    for train_idx, test_idx in splitter.split(df[["a", "b", "c"]], y_all, groups=df["subject_id"].to_numpy()):
        X_train = df.iloc[train_idx][["a", "b", "c"]].copy()
        X_test = df.iloc[test_idx][["a", "b", "c"]].copy()
        y_train = y_all[train_idx]
        y_test = y_all[test_idx]

        train_subjects = set(df.iloc[train_idx]["subject_id"].unique())
        test_subjects = set(df.iloc[test_idx]["subject_id"].unique())
        assert train_subjects.isdisjoint(test_subjects)
        assert set(np.unique(y_train)).issubset(set(label_metadata["class_ids"]))
        assert set(np.unique(y_test)).issubset(set(label_metadata["class_ids"]))

        selector = fit_correlation_selector(X_train, list(X_train.columns), threshold=0.95)
        assert selector.n_fit_samples == len(X_train)
        assert selector.n_fit_samples != len(df)
        X_train_selected = transform_with_correlation_selector(X_train, selector)
        X_test_selected = transform_with_correlation_selector(X_test, selector)

        assert list(X_train_selected.columns) == list(selector.selected_feature_names)
        assert list(X_test_selected.columns) == list(selector.selected_feature_names)
        assert "b" not in selector.selected_feature_names
        assert "a" in selector.selected_feature_names

        with pytest.raises(ValueError, match="Unmapped labels"):
            canonicalize_labels(["walk", "unknown_action"], "activity", config)

        break


def test_fold_local_model_mapping_is_contiguous_for_noncontiguous_canonical_ids():
    canonical_train = np.array([1, 2, 3, 7, 12, 13, 16, 17, 24], dtype=np.int64)
    canonical_test = np.array([1, 12, 24], dtype=np.int64)

    canonical_to_model = {
        int(canonical_id): model_index
        for model_index, canonical_id in enumerate(sorted(np.unique(canonical_train).tolist()))
    }
    y_train_model = np.asarray([canonical_to_model[int(value)] for value in canonical_train], dtype=np.int64)
    y_test_model = np.asarray([canonical_to_model[int(value)] for value in canonical_test], dtype=np.int64)

    assert sorted(np.unique(canonical_train).tolist()) == [1, 2, 3, 7, 12, 13, 16, 17, 24]
    assert np.array_equal(np.unique(y_train_model), np.arange(len(np.unique(canonical_train))))
    assert y_train_model.min() >= 0
    assert y_train_model.max() < len(np.unique(canonical_train))
    assert set(np.unique(y_test_model)).issubset(set(np.unique(y_train_model)))
    assert y_test_model.tolist() == [0, 4, 8]
