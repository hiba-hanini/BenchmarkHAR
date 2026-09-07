import gc
import json
from pathlib import Path

import numpy as np
import psutil

from src.utils.reproducibility import build_experiment_manifest


def log_ram(tag=""):
    mem = psutil.Process().memory_info().rss / 1e9
    print(f"  [RAM {tag}] {mem:.2f} GB")


def subject_output_path(out_dir: Path, subject_id: str) -> Path:
    return out_dir / f"{subject_id}.npz"


def subject_manifest_path(out_dir: Path, subject_id: str) -> Path:
    return out_dir / f"{subject_id}.manifest.json"


def already_processed(out_dir: Path, subject_id: str, *, current_manifest=None) -> bool:
    path = subject_output_path(out_dir, subject_id)
    manifest_path = subject_manifest_path(out_dir, subject_id)
    if not path.exists() or not manifest_path.exists():
        return False
    if current_manifest is None:
        return True
    try:
        with manifest_path.open("r", encoding="utf-8") as handle:
            existing = json.load(handle)
        return existing.get("experiment_id") == current_manifest.get("experiment_id")
    except Exception:
        return False


def save_subject_windows(out_dir: Path, subject_id, X_windows, labels_windows, label_cols, feature_names=None, current_manifest=None):
    save_dict = {"X": X_windows, "subject_id": subject_id}
    for col in label_cols:
        save_dict[col] = np.array(labels_windows[col], dtype=object)
    if feature_names is not None:
        save_dict["feature_names"] = np.array(feature_names, dtype=object)
    output_path = subject_output_path(out_dir, subject_id)
    np.savez_compressed(output_path, **save_dict)
    if current_manifest is not None:
        with subject_manifest_path(out_dir, subject_id).open("w", encoding="utf-8") as handle:
            json.dump(current_manifest, handle, indent=2, sort_keys=True)
            handle.write("\n")
