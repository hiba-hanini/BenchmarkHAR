import pandas as pd
import numpy as np
from pathlib import Path
from src.datasets.base_dataset import BaseHARDataset

class Pamap2Dataset(BaseHARDataset):

    def __init__(self, config):
        super().__init__(config)
        self.raw_columns = config["raw_columns"]

    def _get_channel_names(self):
        return [
            'imu_hand_accel_16_x', 'imu_hand_accel_16_y', 'imu_hand_accel_16_z',
            'imu_hand_gyro_x', 'imu_hand_gyro_y', 'imu_hand_gyro_z',
            'imu_hand_temperature',
            'imu_chest_accel_16_x', 'imu_chest_accel_16_y', 'imu_chest_accel_16_z',
            'imu_chest_gyro_x', 'imu_chest_gyro_y', 'imu_chest_gyro_z',
            'imu_chest_temperature',
            'imu_ankle_accel_16_x', 'imu_ankle_accel_16_y', 'imu_ankle_accel_16_z',
            'imu_ankle_gyro_x', 'imu_ankle_gyro_y', 'imu_ankle_gyro_z',
            'imu_ankle_temperature',
        ]
    
    def _get_sampling_rate(self):
        return self.config["sampling_rate"]
    
    def list_subject_files(self):
        """Return the list of raw files, one per subject."""
        files = sorted(Path(self.raw_path).glob("*.dat"))

        subjects = []

        for f in files:
            subject_id = f.name.replace(".dat", "")
            subject_id = subject_id.replace("subject", "")
            subjects.append(
                {
                    subject_id: f
                }
            )

        return subjects

    def read_subject_raw(self, f) -> pd.DataFrame:

        subject_id = list(f.keys())[0]
        f = f[subject_id]

        df = pd.read_csv(
            f,
            sep=r"\s+",
            header=None
        )

        df.columns = self.raw_columns[:df.shape[1]]

        keep_cols = (
            [
                self.config["columns"]["timestamp"],
                self.config["columns"]["activity_id"]
            ]
            + self._get_channel_names()
        )

        df = df[keep_cols]

        df["subject_id"] = int(
            f.stem.replace("subject", "")
        )

        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")

        df = df.sort_values(
            ["subject_id", "timestamp"]
        )

        sensor_cols = self._get_channel_names()

        df[sensor_cols] = (
            df.groupby("subject_id")[sensor_cols]
            .transform(
                lambda x: x.interpolate(
                    method="linear",
                    limit_direction="both"
                )
            )
        )

        return df
