# src/datasets/base_dataset.py
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

@dataclass
class HARData:
    signals: np.ndarray  # (N, C)
    labels: dict  # {scheme_name: np.ndarray(N,)}
    subject_ids: np.ndarray  # (N,)
    timestamps: np.ndarray  # (N,) datetime64
    session_ids: np.ndarray  # (N,) filled after gap_detection
    sampling_rate: int
    channel_names: list
    label_map: dict


class BaseHARDataset(ABC):
    def __init__(self, config: dict):
        self.config = config
        self.raw_path = Path(config["raw_path"])
        self.label_cols = config.get("label_cols", [])

    @abstractmethod
    def list_subject_files(self):
        pass

    @abstractmethod
    def read_subject_raw(self, f):
        pass

    @abstractmethod
    def _get_channel_names(self) -> list:
        pass

    @abstractmethod
    def _get_sampling_rate(self) -> int:
        pass

    def _read_raw_files(self) -> list[pd.DataFrame]:
        raw_dfs = []
        for file_path in self.list_subject_files():
            df = self.read_subject_raw(file_path)
            if df is None:
                continue
            raw_dfs.append(df)
        return raw_dfs


    def _validate_timestamps(self, timestamps, subject_ids, sampling_rate):
        timestamps = pd.to_datetime(timestamps)

        assert np.issubdtype(timestamps.dtype, np.datetime64), (
            f"'timestamp' must have dtype datetime64, got {timestamps.dtype}."
        )

        expected_dt = 1.0 / sampling_rate

        for subject in np.unique(subject_ids):
            subject_ts = timestamps[subject_ids == subject].values

            if len(subject_ts) < 2:
                continue

            time_diffs = (
                np.diff(subject_ts)
                .astype("timedelta64[ns]")
                .astype(np.int64) / 1e9
            )

            assert (time_diffs >= 0).all(), (
                f"Non-increasing timestamps detected for subject {subject}."
            )

            tolerance = 0.045

            n_gaps = (np.abs(time_diffs - expected_dt) > tolerance).sum()

            if n_gaps > 0:
                gap_ratio = n_gaps / len(time_diffs)

                if gap_ratio <= 1e-2:
                    print(
                        f"Warning: {n_gaps} small timestamp gaps detected "
                        f"for subject {subject} ({gap_ratio:.2%})."
                    )
                else:
                    print(
                    f"[SKIP] Too many timestamp gaps for subject {subject} "
                    f"({gap_ratio:.2%}, {n_gaps} gaps)."
                    )
                    return False
        return True

    def _validate(self, data):
        if isinstance(data, pd.DataFrame):
            assert "subject_id" in data.columns, "DataFrame must contain subject_id."
            for channel in self._get_channel_names():
                assert channel in data.columns, f"Missing channel column '{channel}'."
            for label_col in self.label_cols:
                assert label_col in data.columns, f"Missing label column '{label_col}'."
            if data[self._get_channel_names()].isna().any().any():
                raise AssertionError("NaN values detected in signal columns.")
            
        
            assert "timestamp" in data.columns, "DataFrame must contain a 'timestamp' column."
            
            is_valid = self._validate_timestamps(
                data["timestamp"],
                data["subject_id"].values,
                self._get_sampling_rate()
            )

            return is_valid

        assert isinstance(data, HARData), "Expected HARData or pandas DataFrame."

        n = len(data.signals)
        assert len(data.subject_ids) == n, "Subject IDs are not aligned with signals."
        if data.timestamps is not None:
            assert len(data.timestamps) == n, "Timestamps are not aligned with signals."

        assert not np.isnan(data.signals).any(), "NaN values detected in signals."
        for scheme, labels in data.labels.items():
            assert len(labels) == n, f"Labels '{scheme}' are not aligned with signals."

        if data.timestamps is not None:
            self._validate_timestamps(
                data.timestamps,
                data.subject_ids,
                data.sampling_rate
            )
        return True


    def load(self) -> HARData:
        raw_dfs = self._read_raw_files()
        if not raw_dfs:
            raise ValueError("No raw data found for dataset.")

        df = pd.concat(raw_dfs, ignore_index=True)
        channels = self._get_channel_names()
        signals = df[channels].values.astype(np.float32)

        if len(self.label_cols) == 0:
            labels = {}
        elif len(self.label_cols) == 1:
            labels = self._harmonize_labels(df[self.label_cols[0]].values)
        else:
            labels = self._harmonize_labels(
                {col: df[col].values for col in self.label_cols}
            )

        timestamps = pd.to_datetime(df["timestamp"]).values if "timestamp" in df.columns else None

        data = HARData(
            signals=signals,
            labels=labels,
            subject_ids=df["subject_id"].values,
            timestamps=timestamps,
            session_ids=None,
            sampling_rate=self._get_sampling_rate(),
            channel_names=channels,
            label_map=self.config.get("common_label_map", {}),
        )
        self._validate(data)
        return data