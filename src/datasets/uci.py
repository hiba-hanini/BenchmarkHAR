import numpy as np
import pandas as pd
from pathlib import Path
from src.datasets.base_dataset import BaseHARDataset


class UciHarDataset(BaseHARDataset):
    """Dataset loader for the UCI Smartphone HAR dataset."""

    def list_subject_files(self):
        files = []
        for split in ["train", "test"]:
            path = self.raw_path / split / f"X_{split}.txt"
            if path.exists():
                files.append(path)
        return files

    def read_subject_raw(self, f):
        # UCI HAR dataset uses separate files for X/y/subject and inertial signals.
        base_dir = Path(f).parent
        if f.name in {"X_train.txt", "X_test.txt"}:
            return self._read_main_split(f, base_dir)

        # We only need the main feature table for preprocessing and windowing.
        return None

    def _read_main_split(self, x_file: Path, base_dir: Path) -> pd.DataFrame:
        split = x_file.parent.name
        y_file = base_dir / f"y_{split}.txt"
        subject_file = base_dir / f"subject_{split}.txt"

        df_x = pd.read_csv(x_file, sep="\s+", header=None)
        df_y = pd.read_csv(y_file, sep="\s+", header=None, names=["activity_id"])
        df_subject = pd.read_csv(subject_file, sep="\s+", header=None, names=["subject_id"])

        df = pd.concat([df_subject, df_y, df_x], axis=1)

        feature_names = self._get_channel_names()
        if len(feature_names) != df_x.shape[1]:
            raise ValueError(
                f"Expected {len(feature_names)} features but found {df_x.shape[1]}"
            )
        df.columns = ["subject_id", "activity_id"] + feature_names

        df["timestamp"] = pd.NaT
        df["activity_id"] = df["activity_id"].astype(int)
        return df

    def _get_channel_names(self):
        return [
            f"feature_{i+1}" for i in range(561)
        ]

    def _get_sampling_rate(self):
        return self.config["sampling_rate"]
