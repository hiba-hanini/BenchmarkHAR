from scipy.signal import resample_poly
import numpy as np
import pandas as pd

def resample_dataframe(df, timestamp, target_fs):
    t = df[timestamp].values

    original_fs = 1 / np.mean(np.diff(t))

    ratio = int(original_fs / target_fs)

    out = {}

    for col in df.columns:
        if col == timestamp:
            continue

        out[col] = resample_poly(
            df[col].values,
            up=1,
            down=ratio
        )

    n = len(out[list(out.keys())[0]])

    out[timestamp] = np.arange(n) / target_fs + t[0]

    return pd.DataFrame(out)