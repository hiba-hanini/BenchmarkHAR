import pandas as pd
import json

from src.models.registry import get_model_class
from sklearn.metrics import f1_score, accuracy_score
from pathlib import Path

from src.utils.cv import get_cv_splitter
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
from src.dimension_reduction.correlation_filter import (
    fit_correlation_selector,
    transform_with_correlation_selector,
)
from src.utils.scientific_checks import (
    validate_fold_samples,
    validate_label_mapping,
    validate_subject_split,
)


import time


def run_model(data_name, config_data, model_name, config_model, max_subjects=None,
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

    df_features_path = f"data/features/{data_name}_features_all.pkl"
    
    df = pd.read_pickle(df_features_path)

    # If requested, restrict to the first `max_subjects` subjects
    if max_subjects is not None:
        unique_subjects_all = sorted(df["subject_id"].unique())
        selected_subjects = unique_subjects_all[:int(max_subjects)]
        df = df[df["subject_id"].isin(selected_subjects)].reset_index(drop=True)

    label_col = config_data["label_to_predict"]
    label_metadata = canonical_label_metadata(config_data, label_col)
    canonical_labels = canonicalize_labels(df[label_col], label_col, config_data)
    validate_label_mapping(canonical_labels, label_metadata["class_ids"], "all samples")
    le = build_canonical_label_encoder(config_data, label_col)
    df[label_col] = le.transform(canonical_labels)

    groups = df["subject_id"].values
    unique_subjects = sorted(df["subject_id"].unique())
    n_subjects = len(unique_subjects)
    if n_subjects < 2:
        raise ValueError(f"Need at least 2 subjects for LOSO, got {n_subjects}.")

    cv_config = config_model.get("cv", {})
    outer, cv_name = get_cv_splitter(n_subjects, cv_config, seed=seed)
    if cv_name == "LOSO":
        n_splits = outer.get_n_splits(groups=groups)
    else:
        n_splits = outer.get_n_splits()

    drop_cols = ["subject_id"] + config_data["label_cols"]
    results = []

    X_all = df.drop(columns=drop_cols)
    y_all = df[label_col]


    for i, (train_idx, test_idx) in enumerate(outer.split(X_all, y_all, groups=groups)):
        X_train, X_test = X_all.iloc[train_idx], X_all.iloc[test_idx]
        y_train, y_test = y_all.iloc[train_idx], y_all.iloc[test_idx]

        train_subjects = sorted(df.iloc[train_idx]["subject_id"].unique().tolist())
        test_subjects = sorted(df.iloc[test_idx]["subject_id"].unique())
        validate_subject_split(train_subjects, test_subjects)
        validate_fold_samples(X_train, X_test)
        validate_label_mapping(y_train, range(len(le.classes_)), "training")
        validate_label_mapping(y_test, range(len(le.classes_)), "test")

        selector = fit_correlation_selector(
            X_train, list(X_all.columns), threshold=config_data.get("correlation_threshold", 0.95),
        )
        if selector.n_fit_samples != len(X_train):
            raise AssertionError("Correlation selector was not fitted on exactly the training samples.")
        X_train = transform_with_correlation_selector(X_train, selector)
        X_test = transform_with_correlation_selector(X_test, selector)
        if list(X_train.columns) != list(X_test.columns):
            raise AssertionError("Train/test feature columns differ after correlation selection.")

        print(f"\n[{cv_name} {i+1}/{n_splits}] test_subjects={test_subjects}")

        model_path = model_dir / f"fold_{i}.joblib"
        selector_path = model_dir / f"selected_features_fold_{i}.json"

        model_config = dict(config_model)
        model_config["seed"] = seed
        model = model_class(model_config)

        cached_selector_matches = (
            selector_path.exists()
            and json.loads(selector_path.read_text()) == list(selector.selected_feature_names)
        )
        existing_manifest = load_manifest(model_manifest_path)
        reuse_allowed = model_path.exists() and cached_selector_matches and existing_manifest is not None and manifest_matches(existing_manifest, expected_model_manifest)
        if reuse_allowed:
            print(f"  [SKIP FIT] Model already exists -> {model_path}, loading...")
            model.load(model_path)
        else:
            if model_path.exists() and not cached_selector_matches:
                print(f"  [STALE] Model artifact at {model_path} is stale for experiment {experiment_id}; refitting.")
            start_training = time.perf_counter()
            model.fit(X_train, y_train)
            training_time_sec = time.perf_counter() - start_training

            model.save(model_path)
            selector_path.write_text(json.dumps(list(selector.selected_feature_names)))
            save_manifest(expected_model_manifest, model_manifest_path)


        start_inference = time.perf_counter()
        y_pred = model.predict(X_test)
        inference_time_sec = time.perf_counter() - start_inference
        inference_time_per_sample_ms = (
            inference_time_sec / len(X_test) * 1000
        )

        y_test_decoded = le.inverse_transform(y_test)
        y_pred_decoded = le.inverse_transform(y_pred)

        fold_result = {
            "fold_index": i,
            "cv_strategy": cv_name,
            "train_subject_ids": train_subjects,
            "test_subjects": test_subjects,
            "test_subject_ids": test_subjects,
            "selected_feature_names": list(selector.selected_feature_names),
            "n_train": len(X_train),
            "n_test": len(X_test),
            "y_true": y_test_decoded.tolist(),
            "y_pred": y_pred_decoded.tolist(),
            "f1_macro": f1_score(y_test, y_pred, average="macro"),
            "accuracy": accuracy_score(y_test, y_pred),
            "training_time_sec": training_time_sec,
            "inference_time_sec": inference_time_sec,
            "inference_time_per_sample_ms": inference_time_per_sample_ms,
        }

        results.append(fold_result)
        print("F1 Macro:", fold_result["f1_macro"] , "Accuracy:", fold_result["accuracy"])

        del model, X_train, X_test, y_train, y_test

    results_df = pd.DataFrame(results)
    print(f"\n[{cv_name}] Results:")
    print(results_df[["f1_macro", "accuracy"]].agg(["mean", "std"]))
    results_df = results_df.assign(
        cv_strategy=cv_name,
        seed=seed,
        deterministic=deterministic,
        dataset=data_name,
        model=model_name,
        n_subjects=n_subjects,
        canonical_label_vocabulary=[label_metadata] * len(results_df),
        experiment_id=experiment_id,
        resolved_configuration=[effective_config] * len(results_df),
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
