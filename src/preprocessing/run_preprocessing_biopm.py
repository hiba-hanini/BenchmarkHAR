from pathlib import Path
import gc

import numpy as np
import pandas as pd

from src.datasets.registry import DATASET_REGISTRY
from src.preprocessing.filter_data import filter_data
from src.preprocessing.label_cleaning import clean_labels
from src.preprocessing.mapping import map_labels
from src.preprocessing.windowing import sliding_window_multilabel

from biopm.biopm.preprocessing import (
    detect_zero_crossings,
    pack_window,
    PreprocessConfig,
    save_subject_h5,
    resample_to_target_fs,
)

MIN_WINDOWS = 50


def _biopm_output_path(out_dir: Path, subject_id: str) -> Path:
    return out_dir / f"Data_MeLabel_{subject_id}.h5"


def _biopm_already_processed(out_dir: Path, subject_id: str) -> bool:
    return _biopm_output_path(out_dir, subject_id).exists()


def _encode_labels(labels, label_col, config) -> np.ndarray:
    return map_labels(
        pd.DataFrame({label_col: labels}),
        label_col,
        config,
    ).to_numpy(dtype=np.int32)


def _make_preprocess_config(fs, window_size, step_size, config) -> PreprocessConfig:
    return PreprocessConfig(
        target_fs=fs,
        window_sec=window_size / fs,
        slide_sec=step_size / fs,
        normalize_size=config["biopm"]["normalize_size"],
        pad_size=config["biopm"]["pad_size"],
    )


def _tokenize_and_save(
    output_dir: Path,
    subject_id: str,
    acc_raw_w: np.ndarray,
    acc_filt_w: np.ndarray,
    labels: np.ndarray,
    cfg: PreprocessConfig,
    time_index: np.ndarray,
) -> int:
    n_axes = acc_raw_w.shape[-1]
    gravity_w = acc_raw_w - acc_filt_w

    out_x_acc, out_x_grav, out_raw, out_labels = [], [], [], []

    for idx, window in enumerate(acc_filt_w):
        me_norm, me_info, pos, _, _ = detect_zero_crossings(
            vel=window,
            time_index=time_index,
            cfg=cfg,
        )
        x_acc = pack_window(me_norm, me_info, pos, cfg)

        out_x_acc.append(x_acc)
        out_x_grav.append(gravity_w[idx])
        out_raw.append(acc_raw_w[idx])
        out_labels.append(labels[idx])

    save_subject_h5(
        output_dir,
        subject_id,
        np.asarray(out_raw, dtype=np.float32),
        np.asarray(out_x_acc, dtype=np.float32),
        np.asarray(out_x_grav, dtype=np.float32),
        np.asarray(out_labels),
    )
    return len(out_labels)


def _process_subject_from_windows(
    subject_id: str,
    npz_path: Path,
    config: dict,
    output_dir: Path,
) -> int | None:
    label_col = config["label_to_predict"]
    raw_axes = config["acc_raw"]
    filtered_axes = [f"{axis}_filtered" for axis in raw_axes]

    fs = config["sampling_rate"]
    window_size = config["window_size"]
    step_size = config["step_size"]
    time_index = np.arange(window_size) / fs
    cfg = _make_preprocess_config(fs, window_size, step_size, config)

    with np.load(npz_path, allow_pickle=True) as data:
        X = data["X"]
        feature_names = data["feature_names"].tolist()
        window_labels = data[label_col]

    n_windows, _, n_features = X.shape
    if len(feature_names) != n_features:
        raise ValueError(
            f"{subject_id}: feature_names ({len(feature_names)}) "
            f"!= n_features ({n_features})"
        )

    raw_idx = [feature_names.index(ax) for ax in raw_axes]
    filt_idx = [feature_names.index(ax) for ax in filtered_axes]
    acc_raw_w = X[:, :, raw_idx].astype(np.float32)
    acc_filt_w = X[:, :, filt_idx].astype(np.float32)
    labels = _encode_labels(window_labels, label_col, config)

    n_saved = _tokenize_and_save(
        output_dir, subject_id, acc_raw_w, acc_filt_w, labels, cfg, time_index
    )
    return n_saved


def _process_subject_from_raw(
    subject_file: dict,
    subject_id: str,
    dataset,
    dataset_name: str,
    config: dict,
    output_dir: Path,
) -> int | None:
    label_col = config["label_to_predict"]
    label_cols = config["label_cols"]
    raw_axes = config["acc_raw"]
    filtered_axes = [f"{axis}_filtered" for axis in raw_axes]
    n_axes = len(raw_axes)

    fs = config["sampling_rate"]
    target_fs = config["biopm"]["target_fs"]
    window_size = config["biopm"]["ws_samples"]
    step_size = window_size // 2
    effective_fs = target_fs if target_fs else fs
    time_index = np.arange(window_size) / effective_fs
    cfg = _make_preprocess_config(effective_fs, window_size, step_size, config)

    df = dataset.read_subject_raw(subject_file)
    if df is None:
        print(f"  [SKIP] {subject_id} : too much missing labels")
        return None

    df = df[raw_axes + ["timestamp"] + label_cols + ["subject_id"]]
    df = clean_labels(
        df,
        dataset_name,
        label_cols,
        config["ffill_limit"],
        unknown_labels=config.get("unknown_labels", []),
    )

    if target_fs and target_fs != fs:
        time_array = df["timestamp"].values
        acc_raw = df[raw_axes].values
        label = df[label_col].values
        acc_new, t_new, label_new = resample_to_target_fs(
            time_array, acc_raw, label, target_fs
        )
        df = pd.DataFrame(acc_new, columns=raw_axes)
        df["timestamp"] = t_new
        df[label_col] = label_new
        df["subject_id"] = subject_id
        fs = target_fs

    n_possible_windows = (len(df) - window_size) // step_size + 1
    if n_possible_windows < MIN_WINDOWS:
        print(
            f"  [SKIP] {subject_id} : not enough data "
            f"({n_possible_windows} possible windows < {MIN_WINDOWS} required)"
        )
        return None

    df = filter_data(
        df,
        fs=fs,
        cutoff_noise=config["cutoff_noise"],
        cutoff_gravity=config["cutoff_gravity"],
        axes=raw_axes,
    )

    acc_raw_full = df[raw_axes].values.astype(np.float32)
    acc_filt_full = df[filtered_axes].values.astype(np.float32)
    gravity_full = acc_raw_full - acc_filt_full
    labels_full = df[label_col].values

    del df
    gc.collect()

    stacked = np.concatenate([acc_raw_full, acc_filt_full, gravity_full], axis=1)
    X_windows, labels_windows = sliding_window_multilabel(
        stacked, {label_col: labels_full}, window_size, step_size
    )

    del acc_raw_full, acc_filt_full, gravity_full, stacked
    gc.collect()

    if X_windows is None or len(X_windows) == 0:
        print(f"  [SKIP] {subject_id} : no windows generated")
        return None

    acc_raw_w = X_windows[:, :, :n_axes]
    acc_filt_w = X_windows[:, :, n_axes : 2 * n_axes]
    labels = _encode_labels(labels_windows[label_col], label_col, config)

    n_saved = _tokenize_and_save(
        output_dir, subject_id, acc_raw_w, acc_filt_w, labels, cfg, time_index
    )

    del X_windows, labels_windows, acc_raw_w, acc_filt_w
    gc.collect()
    return n_saved


def preprocessing_biopm(dataset_name, config, max_subjects=None):
    """
    BioPM preprocessing: for each subject, reuse existing .npz windows when
    available, otherwise preprocess from raw sensor files.
    """
    dataset = DATASET_REGISTRY[dataset_name](config)
    output_dir = Path(config["out_dir_biopm"])
    windows_dir = Path(config["out_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    files = dataset.list_subject_files()
    if max_subjects is not None:
        files = files[:max_subjects]
    print(f"{len(files)} subjects to process")

    for i, subject_file in enumerate(files):
        subject_id = list(subject_file.keys())[0]

        if _biopm_already_processed(output_dir, subject_id):
            print(f"[{i + 1}/{len(files)}] {subject_id} already processed, skip")
            continue

        npz_path = windows_dir / f"{subject_id}.npz"
        if npz_path.exists():
            print(f"[{i + 1}/{len(files)}] {subject_id} from windows ({npz_path.name})")
            n_saved = _process_subject_from_windows(
                subject_id, npz_path, config, output_dir
            )
        else:
            print(f"[{i + 1}/{len(files)}] {subject_id} from raw")
            n_saved = _process_subject_from_raw(
                subject_file, subject_id, dataset, dataset_name, config, output_dir
            )

        if n_saved is not None:
            print(f"  [OK] {subject_id} : {n_saved} saved windows")

    print("\nBioPM preprocessing done. Windows saved in:", output_dir)
