import pandas as pd
from pathlib import Path
from src.datasets.base_dataset import BaseHARDataset

class Capture24Dataset(BaseHARDataset):

    def __init__(self, config):
        super().__init__(config)
        self.meta = pd.read_csv(config["meta_path"])
        self.dict_df = pd.read_csv(config["dict_path"])
        self.age_mapping = config["age_mapping"]

    def list_subject_files(self):
        """Return the list of raw files, one per subject."""
        files = sorted(Path(self.raw_path).glob("*.csv.gz"))

        subjects = []

        for f in files:
            subject_id = f.name.replace(".csv.gz", "")
            subjects.append(
                {
                    subject_id: f
                }
            )

        return subjects

    def read_subject_raw(self, f) -> pd.DataFrame:
        
        pid = list(f.keys())[0]
        file_path = f[pid]

        df = pd.read_csv(file_path, compression='gzip', low_memory=False)
        df['pid'] = pid

        df = df.merge(self.meta, on="pid", how="left")
        df = df.rename(columns={"pid": "subject_id"})
        
        dict_cols = self.config["dict_cols"]
        df = df.merge(self.dict_df[dict_cols], on="annotation", how="left")

        missing_pct = df["annotation"].isna().mean() * 100
        if missing_pct > self.config["max_missing_pct"]:
            return None  # signal to the caller to skip this subject

        df = df.sort_values("time").reset_index(drop=True)
        df["time"] = pd.to_datetime(df["time"])
        df = df.rename(columns={"time": "timestamp"})

        df["age_encoded"] = df["age"].map(self.age_mapping)
        df["sex_encoded"] = df["sex"].map({"F": 0, "M": 1})

        return df

    # Abstract methods of BaseHARDataset (used if using the "load global" mode)
    def _get_channel_names(self):
        return ["x", "y", "z"]

    def _get_sampling_rate(self):
        return self.config["sampling_rate"]