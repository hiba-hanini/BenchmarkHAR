import json
from biopm.biopm.inference import extract_features
import pandas as pd
from pathlib import Path
import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, accuracy_score

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
from src.preprocessing.mapping import canonical_label_metadata, canonicalize_labels
from src.utils.scientific_checks import validate_fold_samples, validate_label_mapping, validate_subject_split
import time

model_name = "biopm"

# Helper functions from biopm.evaluation
def _fit_logreg(X: np.ndarray, y: np.ndarray, C: float, seed: int) -> LogisticRegression:
    clf = LogisticRegression(
        C=C, solver="lbfgs", tol=1e-3, max_iter=1000, random_state=seed,
    )
    clf.fit(X, y)
    return clf

def _inner_C_search(X_train: np.ndarray, y_train: np.ndarray,
                    groups_train: np.ndarray,
                    val_fraction: float,
                    C_grid, seed: int) -> float:
    if len(np.unique(groups_train)) < 3:
        return 1.0
    gss = GroupShuffleSplit(n_splits=1, test_size=val_fraction, random_state=seed)
    tr, va = next(gss.split(X_train, y_train, groups=groups_train))
    sc = StandardScaler()
    Xtr = sc.fit_transform(X_train[tr])
    Xva = sc.transform(X_train[va])
    best_C, best_f1 = C_grid[0], -1.0
    for C in C_grid:
        try:
            clf = _fit_logreg(Xtr, y_train[tr], C, seed)
            f1 = f1_score(y_train[va], clf.predict(Xva), average="macro")
            if f1 > best_f1:
                best_f1, best_C = f1, C
        except Exception:
            continue
    return best_C

def run_biopm(data_name, config_data, config_model, max_subjects=None,
              seed=DEFAULT_SEED, deterministic=True, experiment_id=None, resolved_config=None):

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

    # Feature extraction/loading
    features_dir = Path("data/features")
    features_dir.mkdir(parents=True, exist_ok=True)
    masking_rate = config_model["params"]["masking_rate"]
    features_path = features_dir / f"{data_name}_{model_name}_{masking_rate}_features.npz"

    if features_path.exists():
        print(f"--- Loading BioPM features from {features_path} ---")
        data = np.load(features_path)
        features, labels, subjects = data["features"], data["labels"], data["subjects"]
    else:
        print(f"--- Extracting BioPM features for {data_name} ---")
        features, labels, subjects = extract_features(
            data_root=config_data["out_dir_biopm"],
            masking_rate=masking_rate,
            per_axis=True,
            batch_size=config_model["params"]["batch_size"],
        )
        np.savez(features_path, features=features, labels=labels, subjects=subjects)
        print(f"--- Saved BioPM features to {features_path} ---")

    # CV and modeling
    label_col = config_data["label_to_predict"]
    label_metadata = canonical_label_metadata(config_data, label_col)
    y = canonicalize_labels(labels, label_col, config_data)
    validate_label_mapping(y, label_metadata["class_ids"], "all samples")
    groups = subjects

    # If requested, restrict to the first `max_subjects` subjects
    if max_subjects is not None:
        unique_subjects_all = sorted(np.unique(groups))
        selected_subjects = unique_subjects_all[:int(max_subjects)]
        mask = np.isin(groups, selected_subjects)
        features = features[mask]
        y = y[mask]
        groups = groups[mask]
        print(f"--- Subsetting to first {max_subjects} subjects: {selected_subjects} ---")
    n_subjects = len(np.unique(groups))

    LOSO_THRESHOLD = 10
    DEFAULT_C_GRID = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)
    
    cv_config = config_model.get("cv", {})
    outer, cv_name = get_cv_splitter(n_subjects, cv_config, seed=seed)
    if cv_name == "LOSO":
        n_splits = outer.get_n_splits(groups=groups)
    else:
        n_splits = outer.get_n_splits()

    results = []
    
    for i, (train_idx, test_idx) in enumerate(outer.split(features, y, groups=groups)):
        X_train, X_test = features[train_idx], features[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        groups_train = groups[train_idx]
        
        train_subjects = sorted(np.unique(groups_train).tolist())
        test_subjects = sorted(np.unique(groups[test_idx]).tolist())
        validate_subject_split(train_subjects, test_subjects)
        validate_fold_samples(X_train, X_test)
        validate_label_mapping(y_train, label_metadata["class_ids"], "training")
        validate_label_mapping(y_test, label_metadata["class_ids"], "test")
        print(f"\n[{cv_name} {i+1}/{n_splits}] test_subjects={test_subjects}")

        model_path = model_dir / f"fold_{i}.joblib"
        scaler_path = model_dir / f"scaler_fold_{i}.joblib"
        
        best_C = None
        existing_manifest = load_manifest(model_manifest_path)
        reuse_allowed = model_path.exists() and scaler_path.exists() and existing_manifest is not None and manifest_matches(existing_manifest, expected_model_manifest)
        if reuse_allowed:
            print(f"  [SKIP FIT] Model and scaler already exist, loading...")
            training_time_sec = None
            clf = joblib.load(model_path)
            sc = joblib.load(scaler_path)
            X_test_scaled = sc.transform(X_test)
        else:
            if model_path.exists() or scaler_path.exists():
                print(f"  [STALE] Model artifact for experiment {experiment_id} is stale; refitting.")
            
            start_training = time.perf_counter()
            val_fraction = 0.125
            best_C = _inner_C_search(X_train, y_train, groups_train, val_fraction, DEFAULT_C_GRID, seed)
            
            sc = StandardScaler()
            X_train_scaled = sc.fit_transform(X_train)
            X_test_scaled = sc.transform(X_test)
            
            clf = _fit_logreg(X_train_scaled, y_train, best_C, seed)
            training_time_sec = time.perf_counter() - start_training
            
            joblib.dump(clf, model_path)
            joblib.dump(sc, scaler_path)
            save_manifest(expected_model_manifest, model_manifest_path)
            print(f"  [FIT] Saved model to {model_path} and scaler to {scaler_path}")

        y_pred = clf.predict(X_test_scaled)
        
        fold_result = {
            "fold_index": i,
            "cv_strategy": cv_name,
            "train_subject_ids": train_subjects,
            "test_subjects": test_subjects,
            "test_subject_ids": test_subjects,
            "n_train": len(X_train),
            "n_test": len(X_test),
            "y_true": y_test.tolist(),
            "y_pred": y_pred.tolist(),
            "f1_macro": f1_score(y_test, y_pred, average="macro"),
            "accuracy": accuracy_score(y_test, y_pred),
            "selected_C": best_C,
            "training_time_sec": training_time_sec,
        }
        
        results.append(fold_result)
        print("F1 Macro:", fold_result["f1_macro"] , "Accuracy:", fold_result["accuracy"])

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
