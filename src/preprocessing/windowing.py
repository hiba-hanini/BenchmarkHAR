import numpy as np
import pandas as pd

def sliding_window_multilabel(data, labels_dict, window_size, step_size,
                                unknown_ratio_max=0.2):
    """
    data          : (n_rows, n_features)
    labels_dict   : {label_col: array(n_rows,)}
    Return X_windows (n_win, window_size, n_features) and labels_windows dict,
    """
    n_rows = len(data)
    X_windows, labels_windows = [], {col: [] for col in labels_dict}

    for start in range(0, n_rows - window_size + 1, step_size):
        end = start + window_size
        window_labels = {}
        valid = True
        for col, arr in labels_dict.items():
            w = arr[start:end]
            if (w == "Unknown").mean() > unknown_ratio_max:
                valid = False
                break
            mode = pd.Series(w[w != "Unknown"]).mode()
            if len(mode) == 0:
                valid = False
                break
            window_labels[col] = mode.iloc[0]
        if not valid:
            continue
        X_windows.append(data[start:end])
        for col in labels_dict:
            labels_windows[col].append(window_labels[col])

    if len(X_windows) == 0:
        return None, None
    return np.stack(X_windows).astype(np.float32), labels_windows