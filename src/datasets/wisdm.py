import pandas as pd
from pathlib import Path
from src.datasets.base_dataset import BaseHARDataset

class WisdmDataset(BaseHARDataset):

    def __init__(self, config):
        super().__init__(config)
        self.sensor = config["sensor_to_use"]

    def list_subject_files(self) -> list[dict]:
        """Return accel/gyro files grouped by subject.
        [ { subject_id: (accel_file, gyro_file) } ]
        """

        sensor_path = Path(self.raw_path) / self.sensor

        accel_files = {
            f.name.split("_")[1]: f
            for f in (sensor_path / "accel").glob("*.txt")
        }

        gyro_files = {
            f.name.split("_")[1]: f
            for f in (sensor_path / "gyro").glob("*.txt")
        }

        subjects = []

        for subject_id in sorted(accel_files.keys()):
            if subject_id in gyro_files:
                subjects.append(
                    {
                        subject_id: (
                            accel_files[subject_id],
                            gyro_files[subject_id],
                        )
                    }
                )

        return subjects

    def read_subject_raw(self, f) -> pd.DataFrame:
        subject_id = list(f.keys())[0]
        accel_file, gyro_file = f[subject_id]
        
        accel = pd.read_csv(accel_file, header=None, names=[
            "subject_id",
            "activity-label",
            "timestamp",
            "x",
            "y",
            "z",
            ],
            skipinitialspace=True,
            )

        gyro = pd.read_csv(gyro_file, header=None, names=[
            "subject_id",
            "activity-label",
            "timestamp",
            "gx",
            "gy",
            "gz",
            ],
            skipinitialspace=True,
            )
        
        df = accel.merge(gyro,
            on=[
                "subject_id",
                "activity-label",
                "timestamp",
            ],
            how="inner",
            )
        
        sensor_cols = ["x", "y", "z", "gx", "gy", "gz"]

        # some columns have str;
        for col in sensor_cols:
            if col in df.columns:
                df[col] = (
                    df[col]
                    .astype(str)
                    .str.replace(";", "", regex=False)
                    .str.strip()
                )
                df[col] = pd.to_numeric(df[col], errors="raise")
        

        df["subject_id"] = df["subject_id"].astype(str)

        df = df.sort_values("timestamp").reset_index(drop=True)

        df["timestamp"] = pd.to_datetime(df["timestamp"])

        return df

    def _get_channel_names(self):
        return ["x", "y", "z", "gx", "gy", "gz"]

    def _get_sampling_rate(self):
        return self.config["sampling_rate"]