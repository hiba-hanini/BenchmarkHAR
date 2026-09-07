import gc
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.features.feature_extraction import extract_features
from src.preprocessing.sensor_groups import build_sensor_groups, get_all_feature_names
from src.utils.reproducibility import (
    build_experiment_manifest,
    compute_experiment_id,
    load_manifest,
    manifest_matches,
    save_manifest,
)


def run_feature_extraction(dataset_name, config, max_subjects=None, experiment_id=None, resolved_config=None):
    effective_config = resolved_config or {"dataset": dataset_name, "data": config}
    experiment_id = experiment_id or compute_experiment_id(effective_config)

    windows_dir = Path(config["out_dir"])
    features_dir = Path(config["out_dir"]).parent.parent / "features" / dataset_name
    merged_path = features_dir.parent / f"{dataset_name}_features_all.pkl"
    manifest_path = merged_path.parent / "manifest.json"
    expected_manifest = build_experiment_manifest(
        dataset=dataset_name,
        model=None,
        seed=effective_config.get("seed", 42),
        deterministic=bool(effective_config.get("deterministic", True)),
        resolved_configuration=effective_config,
        artifact_path=merged_path,
        artifact_type="feature_cache",
        experiment_id=experiment_id,
    )

    if merged_path.exists() and manifest_path.exists():
        existing_manifest = load_manifest(manifest_path)
        if existing_manifest is not None and manifest_matches(existing_manifest, expected_manifest):
            print(f"Feature cache '{merged_path}' is already valid for experiment {experiment_id}; skipping feature extraction.")
            return
        print(f"Feature cache '{merged_path}' is stale or mismatched for experiment {experiment_id}; rebuilding.")
    elif merged_path.exists():
        print(f"Feature cache '{merged_path}' exists without a manifest; rebuilding.")

    features_dir.mkdir(parents=True, exist_ok=True)

    sensor_groups = build_sensor_groups(config)
    tabular_cols = config.get("tabular_cols", [])
    channel_names = get_all_feature_names(
        config, sensor_groups,
        include_raw=True, include_temperature=config.get("include_temperature", False)
    ) + tabular_cols

    label_cols = config["label_cols"]

    npz_files = sorted(windows_dir.glob("*.npz"))
    if max_subjects is not None:
        npz_files = npz_files[:int(max_subjects)]
        print(f"[run_feature_extraction] Limiting to first {max_subjects} subjects: {[f.stem for f in npz_files]}")
    print(f"{len(npz_files)} .npz files found in {windows_dir} for feature extraction.")
    if len(npz_files) == 0:
        # je veux un msg comme ça         print(f"[run_correlation_filter] {dataset_name}_features_all.pkl not found in {features_dir}, please run the feature extraction step first.")
        print(f"[run_feature_extraction] No .npz files found in {windows_dir}, please run the preprocessing step first.")
        return

    for i_file, f in enumerate(npz_files):
        subject_id = f.stem
        out_path = features_dir / f"{subject_id}.pkl"
        if out_path.exists():
            print(f"[{i_file+1}/{len(npz_files)}] {subject_id} already processed, skip")
            continue

        try:
            data = np.load(f, allow_pickle=True)
            X_subject = data["X"]

        except Exception as e:
            print(f"[{i_file+1}/{len(npz_files)}] {subject_id} -> Could not load ({e}), skip")
            continue

        subject_feats = []
        for i, window in enumerate(X_subject):
            feats = extract_features(
                window, channel_names, sensor_groups=sensor_groups,
                tabular_cols=tabular_cols, fs=config["sampling_rate"],
            )
            feats["subject_id"] = subject_id
            for col in label_cols:
                if col in data:
                    feats[col] = str(data[col][i])
            subject_feats.append(feats)


        df_subject = pd.DataFrame(subject_feats)
            
        df_subject.to_pickle(out_path)
        print(f"[{i_file+1}/{len(npz_files)}] {subject_id} -> {len(X_subject)} windows -> features extracted")

        del data, X_subject, subject_feats, df_subject
        gc.collect()

    print("\nFeature extraction done ->", features_dir)

    print("Combining all .pkl files into a single DataFrame...")
    feature_pkls = sorted(features_dir.glob("*.pkl"))
    df_features = pd.concat((pd.read_pickle(p) for p in feature_pkls), ignore_index=True)
    print("df_features shape :", df_features.shape)

    df_features.to_pickle(merged_path)
    save_manifest(expected_manifest, manifest_path)
    print("Saved ->", merged_path)
    print(f"Manifest saved -> {manifest_path}")
