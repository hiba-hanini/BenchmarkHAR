import gc
import json
from pathlib import Path
import tensorflow as tf
import numpy as np
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import GroupShuffleSplit
import pandas as pd

from src.models.deep.architectures import prepare_input
from src.models.deep.architectures import standardize_data, apply_standardize
from src.models.registry import get_model_class
from src.utils.cv import get_cv_splitter
from src.utils.load_windows import _load_all_windows
from src.utils.reproducibility import (
    DEFAULT_SEED,
    build_experiment_manifest,
    compute_experiment_id,
    get_model_artifact_dir,
    get_result_path,
    load_manifest,
    manifest_matches,
    save_manifest,
)
from src.preprocessing.mapping import (
    build_canonical_label_encoder,
    canonical_label_metadata,
    canonicalize_labels,
)
from src.utils.scientific_checks import (
    class_counts_for_labels,
    validate_fold_samples,
    validate_label_mapping,
    validate_subject_split,
)

import joblib
import time
import psutil


def _subject_level_inner_validation_split(train_subject_ids, validation_fraction=0.2, seed=DEFAULT_SEED):
    unique_subjects = np.asarray(sorted(np.unique(train_subject_ids)))
    if len(unique_subjects) < 2:
        return unique_subjects.tolist(), []
    validation_fraction = float(validation_fraction)
    if not (0 < validation_fraction < 1):
        validation_fraction = 0.2
    n_subjects = len(unique_subjects)
    n_val_subjects = max(1, int(round(n_subjects * validation_fraction)))
    if n_val_subjects >= n_subjects:
        n_val_subjects = max(1, n_subjects - 1)
    splitter = GroupShuffleSplit(n_splits=1, test_size=n_val_subjects / n_subjects, random_state=seed)
    subject_idx = np.arange(n_subjects)
    train_idx, val_idx = next(splitter.split(np.zeros(n_subjects), groups=unique_subjects))
    inner_train = unique_subjects[train_idx].tolist()
    inner_val = unique_subjects[val_idx].tolist()
    return inner_train, inner_val



def run_model_dl(data_name, config_data, model_name, config_model, max_subjects=None,
                 seed=DEFAULT_SEED, deterministic=True, experiment_id=None, resolved_config=None):
    model_class = get_model_class(model_name)
    if model_class is None:
        raise ValueError(f"Model '{model_name}' not found in the registry.")

    effective_config = resolved_config or {
        "dataset": data_name,
        "model": model_name,
        "seed": int(seed),
        "deterministic": bool(deterministic),
        "data": config_data,
        "model_config": config_model,
    }
    experiment_id = experiment_id or compute_experiment_id(effective_config)
    model_dir = get_model_artifact_dir(data_name, model_name, experiment_id)
    model_dir.mkdir(parents=True, exist_ok=True)
    model_manifest_path = model_dir / "manifest.json"
    expected_model_manifest = build_experiment_manifest(
        dataset=data_name,
        model=model_name,
        seed=seed,
        deterministic=deterministic,
        resolved_configuration=effective_config,
        artifact_path=model_dir,
        artifact_type="model_checkpoint_dir",
        experiment_id=experiment_id,
    )
    if not model_manifest_path.exists():
        save_manifest(expected_model_manifest, model_manifest_path)

    print(f"[LOAD] Windows : {data_name}...")
    X, y, groups = _load_all_windows(config_data, max_subjects=max_subjects)
    print(f"X shape: {X.shape} | y shape: {y.shape} | subjects: {len(np.unique(groups))}")

    X = prepare_input(X, model_name)
    
    label_col = config_data["label_to_predict"]
    label_metadata = canonical_label_metadata(config_data, label_col)
    canonical_y = canonicalize_labels(y, label_col, config_data)
    validate_label_mapping(canonical_y, label_metadata["class_ids"], "all samples")
    le = build_canonical_label_encoder(config_data, label_col)
    y = np.asarray(le.transform(canonical_y), dtype=np.int64)
    groups = np.asarray(groups)

    joblib.dump(le, model_dir / "label_encoder.joblib")


    n_classes = len(np.unique(y))

    unique_subjects = sorted(np.unique(groups))
    n_subjects = len(unique_subjects)

    if n_subjects < 2:
        raise ValueError(f"Need at least 2 subjects for LOSO, got {n_subjects}.")

    cv_config = config_model.get("cv", {})
    outer, cv_name = get_cv_splitter(n_subjects, cv_config, seed=seed)
    if cv_name == "LOSO":
        n_splits = outer.get_n_splits(groups=groups)
    else:
        n_splits = outer.get_n_splits()

    results = []

    process = psutil.Process()

    for i, (train_idx, test_idx) in enumerate(outer.split(X, canonical_y, groups=groups)):
        test_subjects = np.unique(groups[test_idx])
        print(f"\n[{cv_name} {i+1}/{n_splits}] test_subjects={test_subjects}")

        X_train, X_test = X[train_idx], X[test_idx]
        canonical_y_train = canonical_y[train_idx]
        canonical_y_test = canonical_y[test_idx]
        train_subjects = sorted(np.unique(groups[train_idx]).tolist())
        validate_subject_split(train_subjects, test_subjects.tolist())
        validate_fold_samples(X_train, X_test)
        validate_label_mapping(canonical_y_train, label_metadata["class_ids"], "training")
        validate_label_mapping(canonical_y_test, label_metadata["class_ids"], "test")

        train_canonical_classes = sorted(np.unique(canonical_y_train).tolist())
        canonical_to_model = {
            int(canonical_id): model_index
            for model_index, canonical_id in enumerate(train_canonical_classes)
        }
        model_to_canonical = {
            model_index: int(canonical_id)
            for model_index, canonical_id in enumerate(train_canonical_classes)
        }

        validation_fraction = float(config_model.get("params", {}).get("validation_fraction", 0.2))
        selected_validation_subjects = []
        if len(np.unique(groups[train_idx])) > 1:
            inner_train_subjects, inner_val_subjects = _subject_level_inner_validation_split(
                groups[train_idx], validation_fraction=validation_fraction, seed=seed + i
            )
            inner_train_mask = np.isin(groups[train_idx], inner_train_subjects)
            inner_val_mask = np.isin(groups[train_idx], inner_val_subjects)
            X_train_inner = X_train[inner_train_mask]
            canonical_y_train_inner = canonical_y_train[inner_train_mask]
            X_val_inner = X_train[inner_val_mask]
            canonical_y_val_inner = canonical_y_train[inner_val_mask]
            selected_validation_subjects = inner_val_subjects
            if len(X_train_inner) == 0 or len(X_val_inner) == 0:
                X_train_inner = X_train
                canonical_y_train_inner = canonical_y_train
                X_val_inner = None
                canonical_y_val_inner = None
                selected_validation_subjects = []
        else:
            X_train_inner = X_train
            canonical_y_train_inner = canonical_y_train
            X_val_inner = None
            canonical_y_val_inner = None

        y_train_model = np.asarray([canonical_to_model[int(value)] for value in canonical_y_train_inner], dtype=np.int64)
        if X_val_inner is not None and len(X_val_inner) > 0:
            val_missing = sorted(set(np.unique(canonical_y_val_inner).tolist()) - set(train_canonical_classes))
            if val_missing:
                raise ValueError(
                    f"Fold {i}: validation canonical labels {val_missing} are missing from the training-fold vocabulary {train_canonical_classes}."
                )
            y_val_model = np.asarray([canonical_to_model[int(value)] for value in canonical_y_val_inner], dtype=np.int64)
            X_train_inner, X_val_inner, scaler = standardize_data(model_name, X_train_inner, X_val_inner)
            X_test_scaled = apply_standardize(model_name, X_test, scaler)
            validation_data = (X_val_inner, y_val_model)
            validation_strategy = "subject_level_inner_split"
        else:
            test_missing = sorted(set(np.unique(canonical_y_test).tolist()) - set(train_canonical_classes))
            if test_missing:
                raise ValueError(
                    f"Fold {i}: test canonical labels {test_missing} are missing from the training-fold vocabulary {train_canonical_classes}."
                )
            X_train_inner, X_test_scaled, scaler = standardize_data(model_name, X_train, X_test)
            validation_data = None
            validation_strategy = "fixed_configuration"

        test_missing = sorted(set(np.unique(canonical_y_test).tolist()) - set(train_canonical_classes))
        if test_missing:
            raise ValueError(
                f"Fold {i}: test canonical labels {test_missing} are missing from the training-fold vocabulary {train_canonical_classes}."
            )
        y_test_model = np.asarray([canonical_to_model[int(value)] for value in canonical_y_test], dtype=np.int64)

        n_classes = len(train_canonical_classes)
        assert np.array_equal(np.unique(y_train_model), np.arange(len(train_canonical_classes)))
        assert n_classes == len(train_canonical_classes)
        assert y_train_model.min() >= 0
        assert y_train_model.max() < n_classes
        if validation_data is not None:
            y_val_model = validation_data[1]
            assert y_val_model.min() >= 0
            assert y_val_model.max() < n_classes
        assert y_test_model.min() >= 0
        assert y_test_model.max() < n_classes

        if model_name == "tinyhar":
            model_path = model_dir / f"fold_{i}.pt"
        else :
            model_path = model_dir / f"fold_{i}.keras"

        scaler_path = model_dir / f"scaler_fold_{i}.joblib"
        label_mapping_path = model_dir / f"label_mapping_fold_{i}.joblib"
        joblib.dump(scaler, scaler_path)
        joblib.dump({
            "canonical_classes": train_canonical_classes,
            "canonical_to_model": canonical_to_model,
            "model_to_canonical": model_to_canonical,
        }, label_mapping_path)

        model_config = dict(config_model)
        model_config["seed"] = seed
        model = model_class(model_config)

        existing_manifest = load_manifest(model_manifest_path)
        reuse_allowed = model_path.exists() and existing_manifest is not None and manifest_matches(existing_manifest, expected_model_manifest)
        try:
            tf.config.experimental.reset_memory_stats("GPU:0")
        except Exception:
            pass
        training_time_sec = None
        if reuse_allowed:
            if model_name == "tinyhar":
                model._input_shape = X_train_inner.shape[1:]
                model._n_classes = int(n_classes)
            print(f"  [SKIP FIT] Model already exists -> {model_path}, loading...")
            model.load(model_path)
        else:
            if model_path.exists():
                print(f"  [STALE] Model artifact at {model_path} is stale for experiment {experiment_id}; refitting.")
            callbacks = []
            if validation_data is not None and model_name != "tinyhar":
                callbacks = [
                    tf.keras.callbacks.EarlyStopping(
                        monitor="val_loss",
                        mode="min",
                        patience=int(config_model.get("params", {}).get("early_stopping_patience", 5)),
                        restore_best_weights=True,
                    ),
                    tf.keras.callbacks.ModelCheckpoint(
                        str(model_path),
                        monitor="val_loss",
                        mode="min",
                        save_best_only=True,
                        save_weights_only=False,
                    ),
                ]

            start_training = time.perf_counter()
            history = model.fit(X_train_inner, y_train_model, n_classes, verbose=1, validation_data=validation_data, callbacks=callbacks)
            training_time_sec = time.perf_counter() - start_training

            history_path = model_dir / f"fold_{i}_history.json"

            history_payload = {
                "model": model_name,
                "dataset": data_name,
                "experiment_id": experiment_id,
                "fold_index": i,
                "train_subject_ids": train_subjects,
                "test_subject_ids": test_subjects.tolist(),
                "validation_subject_ids": sorted(selected_validation_subjects),
                "history": history.history,
            }

            with open(history_path, "w", encoding="utf-8") as f:
                json.dump(history_payload, f, indent=2)

            print(f"  [HISTORY] Saved training history -> {history_path}")

            if validation_data is not None and model_name != "tinyhar" and model_path.exists():
                model.load(model_path)
            else:
                model.save(model_path)
            save_manifest(expected_model_manifest, model_manifest_path)

        start_inference = time.perf_counter()
        y_pred_model = model.predict(X_test_scaled)
        inference_time_sec = time.perf_counter() - start_inference
        inference_time_per_sample_ms = (
            inference_time_sec / len(X_test_scaled) * 1000
        )

        cpu_memory_mb = process.memory_info().rss / (1024 ** 2)
        gpu_memory_mb = None

        try:
            memory_info = tf.config.experimental.get_memory_info("GPU:0")
            gpu_memory_mb = memory_info["peak"] / (1024 ** 2)
        except Exception:
            pass
        gpu_available = bool(tf.config.list_physical_devices("GPU"))

        y_pred_canonical = np.asarray([model_to_canonical[int(value)] for value in y_pred_model], dtype=np.int64)

        train_class_counts = class_counts_for_labels(canonical_y_train_inner, label_metadata["class_ids"])
        test_class_counts = class_counts_for_labels(canonical_y_test, label_metadata["class_ids"])
        fold_result = {
            "fold_index": i,
            "cv_strategy": cv_name,
            "train_subject_ids": train_subjects,
            "test_subjects": test_subjects.tolist(),
            "test_subject_ids": test_subjects.tolist(),
            "validation_subject_ids": sorted(selected_validation_subjects),
            "canonical_train_classes": train_canonical_classes,
            "canonical_to_model": canonical_to_model,
            "model_to_canonical": model_to_canonical,
            "n_train": len(X_train_inner),
            "n_test": len(X_test_scaled),
            "train_class_counts": train_class_counts,
            "test_class_counts": test_class_counts,
            "validation_strategy": validation_strategy,
            "y_true": canonical_y_test.tolist(),
            "y_pred": y_pred_canonical.tolist(),
            "f1_macro": f1_score(canonical_y_test, y_pred_canonical, average="macro"),
            "accuracy": accuracy_score(canonical_y_test, y_pred_canonical),
            "training_time_sec": training_time_sec,
            "inference_time_sec": inference_time_sec,
            "inference_time_per_sample_ms": inference_time_per_sample_ms,
            "cpu_memory_mb": cpu_memory_mb,
            "gpu_memory_mb": gpu_memory_mb,
            "gpu_available": gpu_available,
        }
        results.append(fold_result)
        print("F1 Macro:", fold_result["f1_macro"], "Accuracy:", fold_result["accuracy"])

        del model
        del X_train_inner, X_test_scaled, X_val_inner, canonical_y_train_inner, canonical_y_val_inner, canonical_y_train, canonical_y_test
        tf.keras.backend.clear_session()
        gc.collect()

    results_df = pd.DataFrame(results)
    print(f"\n[{cv_name}] Results:")
    print(results_df[["f1_macro", "accuracy"]].agg(["mean", "std"]))
    canonical_vocab_payload = json.dumps(label_metadata, sort_keys=True)
    resolved_config_payload = json.dumps(effective_config, sort_keys=True, default=str)
    results_df = results_df.assign(
        cv_strategy=cv_name,
        seed=seed,
        deterministic=deterministic,
        dataset=data_name,
        model=model_name,
        n_subjects=n_subjects,
        canonical_label_vocabulary=[canonical_vocab_payload] * len(results_df),
        experiment_id=experiment_id,
        resolved_configuration=[resolved_config_payload] * len(results_df),
    )

    results_path = get_result_path(data_name, model_name, {
        "dataset": data_name,
        "model": model_name,
        "seed": seed,
        "deterministic": deterministic,
        "resolved_configuration": effective_config,
        "experiment_id": experiment_id,
    })
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_pickle(results_path)
    save_manifest(build_experiment_manifest(
        dataset=data_name,
        model=model_name,
        seed=seed,
        deterministic=deterministic,
        resolved_configuration=effective_config,
        artifact_path=results_path,
        artifact_type="results",
        experiment_id=experiment_id,
    ), results_path.parent / "manifest.json")

    print(f"\n[{cv_name}] done. {results_path}")

    return results_df
