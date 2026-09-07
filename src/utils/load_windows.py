import gc
from pathlib import Path

import numpy as np


def _load_all_windows(config_data, max_subjects=None):
    windows_dir = Path(config_data["out_dir"])
    npz_files = sorted(windows_dir.glob("*.npz"))
    if max_subjects is not None:
        npz_files = npz_files[:int(max_subjects)]
        print(f"[LOAD] Limiting to first {max_subjects} subjects from {windows_dir}.")
    if not npz_files:
        raise ValueError(f"No .npz files found in {windows_dir}. Please run the preprocessing and feature extraction steps first.")

    X_list, y_list, subject_list = [], [], []
    label_col = config_data["label_to_predict"]

    for f in npz_files:
        data = np.load(f, allow_pickle=True)
        X_list.append(data["X"])
        y_list.append(data[label_col])
        subject_list.append(np.full(len(data["X"]), str(data["subject_id"])))
        del data
        gc.collect()

    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    subject_ids = np.concatenate(subject_list, axis=0)

    del X_list, y_list, subject_list
    gc.collect()
    print("X dtype:", X.dtype)
    print("X nbytes:", X.nbytes / 1024**3, "GB")
    return X, y, subject_ids