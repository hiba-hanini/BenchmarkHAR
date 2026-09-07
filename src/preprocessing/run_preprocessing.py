from pathlib import Path
import gc
from src.datasets.registry import DATASET_REGISTRY
from src.preprocessing.filter_data import filter_data
from src.preprocessing.label_cleaning import clean_labels
from src.preprocessing.windowing import sliding_window_multilabel
from src.preprocessing.mapping import canonicalize_labels
from src.preprocessing.subject_cache import (
    already_processed, save_subject_windows
)
import numpy as np
from src.preprocessing.sensor_groups import (build_sensor_groups,
                                              get_all_feature_names)


MIN_WINDOWS = 50

def run_preprocessing(dataset_name, config, max_subjects=None):
    dataset = DATASET_REGISTRY[dataset_name](config)
    out_dir = Path(config["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    files = dataset.list_subject_files()
    if max_subjects is not None:
        files = files[:max_subjects]
    label_cols = config["label_cols"]

    sensor_groups = build_sensor_groups(config)

    col_to_keep = config.get("only_keep_columns", None)

    for i, f in enumerate(files):

        subject_id = list(f.keys())[0]

        if already_processed(out_dir, subject_id):
            print(f"[{i+1}/{len(files)}] {subject_id} already processed, skip")
            continue

        print(f"[{i+1}/{len(files)}] Processing {subject_id}")

        df = dataset.read_subject_raw(f)

        if df is None:
            print(f"  [SKIP] {subject_id} : too much missing labels")
            continue
    
        if not dataset._validate(df):
            print(f"  [SKIP] {subject_id} : not valid")
            continue

        if col_to_keep is not None:
            df = df[col_to_keep]

        df = clean_labels(df, dataset_name, label_cols, config["ffill_limit"], unknown_labels=config.get("unknown_labels", []))

        n_possible_windows = (len(df) - config["window_size"]) // config["step_size"] + 1
        if n_possible_windows < MIN_WINDOWS:
            print(f"  [SKIP] {subject_id} : not enough data ({n_possible_windows} possible windows < {MIN_WINDOWS} required)")
            continue

        for group in sensor_groups:
            df = filter_data(
                df,
                fs=config["sampling_rate"],
                cutoff_noise=config["cutoff_noise"],
                cutoff_gravity=config["cutoff_gravity"] if group["is_accel"] else None,
                axes=group["axes"],
                type=group["type"],
            )


        features = get_all_feature_names(config, sensor_groups, include_raw=True, include_temperature=True) + config.get("tabular_cols", [])


        data = df[features].values.astype(np.float32)
        labels_dict = {col: df[col].values for col in label_cols}


        X_windows, labels_windows = sliding_window_multilabel(
            data, labels_dict, config["window_size"], config["step_size"]
        )

        del df, data, labels_dict
        gc.collect()

        if X_windows is not None:
            # Windowing discards unknown-only windows first; canonicalization is
            # then configuration-defined and shared by ML, DL, and BioPM.
            labels_windows = {
                col: canonicalize_labels(values, col, config)
                for col, values in labels_windows.items()
            }
            save_subject_windows(out_dir, subject_id, X_windows, labels_windows, label_cols, feature_names=features)
            print(f"  [OK] {subject_id} : {len(X_windows)} saved windows")
            del X_windows, labels_windows

        gc.collect()


    print("\nPreprocessing done ! Windows saved in :", out_dir)
