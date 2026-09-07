import numpy as np
from scipy import stats
from scipy.signal import find_peaks


def _safe_corr(a, b):
    if np.std(a) < 1e-10 or np.std(b) < 1e-10:
        return 0.0
    with np.errstate(invalid="ignore", divide="ignore"):
        c = np.corrcoef(a, b)[0, 1]
    return 0.0 if np.isnan(c) else c


def _basic_stats(x, col_name, fs, feats, compute_peaks=False):
    feats[f'{col_name}_mean']   = np.mean(x)
    feats[f'{col_name}_std']    = np.std(x)
    feats[f'{col_name}_min']    = np.min(x)
    feats[f'{col_name}_max']    = np.max(x)
    feats[f'{col_name}_range']  = np.max(x) - np.min(x)
    feats[f'{col_name}_median'] = np.median(x)
    feats[f'{col_name}_iqr']    = stats.iqr(x)
    feats[f'{col_name}_rms']    = np.sqrt(np.mean(x**2))
    feats[f'{col_name}_energy'] = np.mean(x**2)
    feats[f'{col_name}_mad']    = np.mean(np.abs(x - np.mean(x)))
    feats[f'{col_name}_zc']     = np.sum(np.diff(np.sign(x)) != 0)
    feats[f'{col_name}_perc25'] = np.percentile(x, 25)
    feats[f'{col_name}_perc75'] = np.percentile(x, 75)

    fft_vals = np.abs(np.fft.rfft(x))
    freqs = np.fft.rfftfreq(len(x), d=1 / fs)
    feats[f'{col_name}_fft_mean']   = np.mean(fft_vals)
    feats[f'{col_name}_fft_std']    = np.std(fft_vals)
    feats[f'{col_name}_fft_energy'] = np.mean(fft_vals**2)

    if len(fft_vals) > 2:
        order = np.argsort(fft_vals[1:])[::-1] + 1
        top2 = order[:2]
        feats[f'{col_name}_dominant_freq']    = freqs[top2[0]]
        feats[f'{col_name}_dominant_power']   = fft_vals[top2[0]]
        feats[f'{col_name}_dominant_freq_2']  = freqs[top2[1]] if len(top2) > 1 else 0.0
        feats[f'{col_name}_dominant_power_2'] = fft_vals[top2[1]] if len(top2) > 1 else 0.0
    else:
        feats[f'{col_name}_dominant_freq'] = feats[f'{col_name}_dominant_power'] = 0.0
        feats[f'{col_name}_dominant_freq_2'] = feats[f'{col_name}_dominant_power_2'] = 0.0

    p = fft_vals / (np.sum(fft_vals) + 1e-10)
    feats[f'{col_name}_spectral_entropy'] = stats.entropy(p + 1e-10)

    if compute_peaks:  # only for magnitudes
        peaks, props = find_peaks(x, prominence=0.01)
        feats[f'{col_name}_n_peaks'] = len(peaks)
        feats[f'{col_name}_median_peak_prominence'] = (
            np.median(props['prominences']) if len(peaks) > 0 else 0.0
        )


def _sensor_group_features(window, col_idx, group, feats):
    """
    Calculate correlation and orientation features for a sensor group (e.g., accelerometer axes).
    """
    axes = group["axes"] # example: ("accel_x", "accel_y", "accel_z")
    prefix = group.get("type", axes[0].rsplit("_", 1)[0])

    raw = {a: window[:, col_idx[a]] for a in axes if a in col_idx}
    filtered_names = [f"{a}_filtered" for a in axes]
    filtered = {a: window[:, col_idx[fn]] for a, fn in zip(axes, filtered_names) if fn in col_idx}

    if len(raw) == 3:
        ax, ay, az = [raw[a] for a in axes]
        feats[f'{prefix}_corr_xy'] = _safe_corr(ax, ay)
        feats[f'{prefix}_corr_xz'] = _safe_corr(ax, az)
        feats[f'{prefix}_corr_yz'] = _safe_corr(ay, az)

    if group.get("is_accel") and len(raw) == 3 and len(filtered) == 3:
        eps = 1e-8
        gx, gy, gz = [filtered[a] for a in axes]
        bx, by, bz = [raw[a] - filtered[a] for a in axes]  # body motion = brut - gravité(filtré)

        dyn_roll  = np.arctan2(by, np.sqrt(bx**2 + bz**2) + eps)
        dyn_pitch = np.arctan2(-bx, np.sqrt(by**2 + bz**2) + eps)
        dyn_yaw   = np.arctan2(bz, np.sqrt(bx**2 + by**2) + eps)
        feats[f'{prefix}_dyn_roll_mean']  = np.mean(dyn_roll)
        feats[f'{prefix}_dyn_roll_std']   = np.std(dyn_roll)
        feats[f'{prefix}_dyn_pitch_mean'] = np.mean(dyn_pitch)
        feats[f'{prefix}_dyn_pitch_std']  = np.std(dyn_pitch)
        feats[f'{prefix}_dyn_yaw_mean']   = np.mean(dyn_yaw)
        feats[f'{prefix}_dyn_yaw_std']    = np.std(dyn_yaw)

        grav_roll  = np.arctan2(gy, np.sqrt(gx**2 + gz**2) + eps)
        grav_pitch = np.arctan2(-gx, np.sqrt(gy**2 + gz**2) + eps)
        grav_yaw   = np.arctan2(gz, np.sqrt(gx**2 + gy**2) + eps)
        feats[f'{prefix}_grav_roll_mean']  = np.mean(grav_roll)
        feats[f'{prefix}_grav_pitch_mean'] = np.mean(grav_pitch)
        feats[f'{prefix}_grav_yaw_mean']   = np.mean(grav_yaw)

    mag_col = f'magnitude_{prefix}'
    if mag_col in col_idx:
        mag = window[:, col_idx[mag_col]]
        lag = 100
        if len(mag) > lag:
            feats[f'{prefix}_mag_autocorr_1s'] = _safe_corr(mag[:-lag], mag[lag:])
        else:
            feats[f'{prefix}_mag_autocorr_1s'] = 0.0


def extract_features(window, channel_names, sensor_groups=None, tabular_cols=None, fs=100):
    feats = {}
    col_idx = {name: i for i, name in enumerate(channel_names)}
    tabular_cols = tabular_cols or []

    for col_name in channel_names:
        if col_name in tabular_cols:
            feats[col_name] = window[0, col_idx[col_name]]
            continue
        x = window[:, col_idx[col_name]]
        is_magnitude = col_name.startswith("magnitude")
        _basic_stats(x, col_name, fs, feats, compute_peaks=is_magnitude)

    if sensor_groups:
        for group in sensor_groups:
            _sensor_group_features(window, col_idx, group, feats)

    return feats