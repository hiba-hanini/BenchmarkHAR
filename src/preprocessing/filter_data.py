import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt

def butter_lowpass_filter(data, cutoff, fs, order=4):
    nyq = 0.5 * fs
    assert cutoff < nyq, (
        f"cutoff frequency ({cutoff}Hz) must be below Nyquist "
        f"frequency ({nyq}Hz) for fs={fs}Hz"
    )
    b, a = butter(order, cutoff / nyq, btype='low')
    return filtfilt(b, a, data)

def calculate_magnitude(df,axes=("x","y","z"), type="acceleration"):
    # to numeric
    for axis in axes:
        df[axis] = pd.to_numeric(df[axis], errors='coerce')
    # calculate magnitude
    df[f'magnitude_{type}'] = np.sqrt(sum(df[a]**2 for a in axes))
    return df


def filter_data(df, fs, cutoff_noise, cutoff_gravity, axes=("x","y","z"), type="acceleration"):
    df = calculate_magnitude(df, axes, type)
    # if cutoff_noise doesnt respect the nyquist condition
    if cutoff_noise is not None and cutoff_noise < fs / 2:
        for axis in axes:
            clean = butter_lowpass_filter(df[axis].values, cutoff_noise, fs)
            if cutoff_gravity is not None:
                gravity = butter_lowpass_filter(clean, cutoff_gravity, fs)
                df[f'{axis}_filtered'] = clean - gravity
            else:
                df[f'{axis}_filtered'] = clean


    elif cutoff_gravity is not None and cutoff_gravity < fs / 2:
        for axis in axes:
            gravity = butter_lowpass_filter(df[axis], cutoff_gravity, fs)
            df[f'{axis}_filtered'] = df[axis] - gravity


    else:
        # No filter at all
        for axis in axes:
            df[f"{axis}_filtered"] = df[axis]

    df[f"magnitude_{type}_filtered"] = np.sqrt(
        sum(df[f"{a}_filtered"]**2 for a in axes)
    )
    
    return df