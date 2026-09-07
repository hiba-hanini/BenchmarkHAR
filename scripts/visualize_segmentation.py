import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from biopm.biopm.preprocessing import PreprocessConfig, detect_zero_crossings
from src.utils.load_config import load_config


def visualize_segmentation(dataset_name: str, subject_id: str, window_index: int):
    """
    Loads a specific signal window, runs zero-crossing segmentation on it,
    and plots the original signal with the detected Movement Elements (MEs) highlighted.
    """
    print(f"--- Visualizing segmentation for {dataset_name}, subject {subject_id}, window {window_index} ---")

    # 1. Load configurations
    config = load_config(dataset_name=dataset_name)
    out_dir = Path(config["out_dir"])
    plots_dir = Path("data/plots")
    plots_dir.mkdir(exist_ok=True)

    # 2. Load the preprocessed window data (.npz)
    npz_path = out_dir / f"{subject_id}.npz"
    if not npz_path.exists():
        print(f"ERROR: Preprocessed file not found at {npz_path}")
        print(f"Please run preprocessing first: python run.py --dataset {dataset_name} --step preprocessing")
        return

    data = np.load(npz_path, allow_pickle=True)
    X_windows = data['X']
    feature_names = data['feature_names'].tolist()

    if window_index >= len(X_windows):
        print(f"ERROR: window_index {window_index} is out of bounds. Subject {subject_id} has {len(X_windows)} windows.")
        return

    # 3. Select the specific window and identify filtered acceleration columns
    window = X_windows[window_index]

    # Identify the correct columns for filtered acceleration
    accel_cols = config.get("acc_raw", ["x", "y", "z"])  # Fallback for simpler configs
    filtered_cols = [f"{col}_filtered" for col in accel_cols]

    try:
        col_indices = [feature_names.index(col) for col in filtered_cols]
    except ValueError as e:
        print(f"ERROR: Could not find required filtered column in the data: {e}")
        print(f"Available columns: {feature_names}")
        return

    accel_filtered_window = window[:, col_indices]

    # 4. Configure and run BioPM's zero-crossing detection
    fs = config.get("sampling_rate")
    biopm_config = config.get("biopm", {})

    cfg = PreprocessConfig(
        target_fs=biopm_config.get("target_fs", fs),
        window_sec=config.get("window_size", 512) / fs,
        slide_sec=config.get("step_size", 256) / fs,
        normalize_size=biopm_config.get("normalize_size", 32),
        pad_size=biopm_config.get("pad_size", 192),
    )

    time_index = np.arange(len(accel_filtered_window)) / fs

    _, me_info, _, _, _ = detect_zero_crossings(
        vel=accel_filtered_window,
        time_index=time_index,
        cfg=cfg,
    )

    if me_info.empty:
        print("No movement elements were detected in this window.")
    else:
        print(f"Detected {len(me_info)} movement elements.")

    # 5. Plot the results
    fig, axes = plt.subplots(3, 1, figsize=(15, 9), sharex=True)
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    axis_labels = ['X-axis', 'Y-axis', 'Z-axis']

    for i in range(3):
        axes[i].plot(time_index, accel_filtered_window[:, i], color=colors[i], label=f'Filtered Accel {axis_labels[i]}')
        axes[i].set_ylabel("Acceleration (g)")
        axes[i].grid(True, which='both', linestyle='--', linewidth=0.5)

    palette = ['#ffadad', '#add8e6']  # Light red, light blue
    if not me_info.empty:
        for i, row in me_info.iterrows():
            axis_idx, start_idx, end_idx = int(row['axis']), int(row['start_point']), int(row['end_point'])
            start_time, end_time = start_idx / fs, end_idx / fs
            axes[axis_idx].axvspan(start_time, end_time, color=palette[i % 2], alpha=0.5)

    legend_patches = [mpatches.Patch(color=palette[0], alpha=0.5, label='Movement Element (Odd)'),
                      mpatches.Patch(color=palette[1], alpha=0.5, label='Movement Element (Even)')]
    for i in range(3):
        line, _ = axes[i].get_legend_handles_labels()
        if line: legend_patches.append(line[0])

    fig.legend(handles=legend_patches, loc='upper right')
    axes[2].set_xlabel("Time (s)")
    fig.suptitle(f'Zero-Crossing Segmentation for {dataset_name.upper()} - Subject {subject_id} - Window {window_index}', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    output_filename = f"segmentation_{dataset_name}_{subject_id}_win{window_index}.png"
    save_path = plots_dir / output_filename
    plt.savefig(save_path)
    print(f"Plot saved to: {save_path}")
    plt.close(fig)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize the zero-crossing segmentation of a signal window.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--dataset", type=str, required=True, help="Name of the dataset (e.g., 'pamap2').")
    parser.add_argument("--subject-id", type=str, required=True, help="ID of the subject to visualize.")
    parser.add_argument("--window-index", type=int, default=0, help="Index of the window to visualize.")
    args = parser.parse_args()
    visualize_segmentation(dataset_name=args.dataset, subject_id=args.subject_id, window_index=args.window_index)