from pathlib import Path
import pickle

import pandas as pd
import matplotlib.pyplot as plt


RESULTS_DIR = Path("data/results/capture24")

DL_MODELS = {
    "cnn",
    "lstm",
    "dnn",
    "rnn",
    "deepconv_lstm",
    "tinyhar",
    "astar_har",
}


def load_results(path):
    with open(path, "rb") as f:
        return pickle.load(f)


rows = []

# Structure:
# data/results/capture24/<model>/<experiment_id>/results.pkl
result_files = list(RESULTS_DIR.glob("*/*/results.pkl"))

print(f"Found {len(result_files)} results.pkl files")

for results_path in result_files:

    model = results_path.parent.parent.name
    experiment_id = results_path.parent.name

    print(f"\nProcessing: {model} / {experiment_id}")

    try:
        results = load_results(results_path)

        if not isinstance(results, pd.DataFrame):
            print(
                f"[WARNING] {results_path}: "
                f"expected DataFrame, got {type(results)}"
            )
            continue

        print("Columns:", list(results.columns))
        print("Shape:", results.shape)

        # -------------------------
        # F1
        # -------------------------
        if "f1_macro" in results.columns:
            f1 = results["f1_macro"].mean()
        else:
            f1 = None

        # -------------------------
        # Training time
        # -------------------------
        if "training_time_sec" in results.columns:
            training_time = results["training_time_sec"].sum()
            training_time_mean = results["training_time_sec"].mean()
        else:
            training_time = None
            training_time_mean = None

        # -------------------------
        # GPU memory
        # -------------------------
        if "gpu_memory_mb" in results.columns:
            gpu_memory = results["gpu_memory_mb"].max()
        else:
            gpu_memory = None

        rows.append({
            "model": model,
            "experiment_id": experiment_id,
            "f1_macro": f1,
            "training_time_sec": training_time,
            "training_time_mean_sec": training_time_mean,
            "gpu_memory_mb": gpu_memory,
        })

    except Exception as e:
        print(f"[WARNING] {results_path}: {e}")


# ============================================================
# Summary
# ============================================================

if not rows:
    raise RuntimeError(
        "No valid results were extracted. "
        "Check the structure and columns of results.pkl."
    )

df = pd.DataFrame(rows)

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print(df.to_string(index=False))


# Save summary
summary_path = RESULTS_DIR / "resource_f1_summary.csv"
df.to_csv(summary_path, index=False)

print(f"\nSaved summary to: {summary_path}")


# ============================================================
# 1. Training time vs F1
# ============================================================

plot_df = df.dropna(
    subset=["training_time_sec", "f1_macro"]
)

if not plot_df.empty:

    plt.figure(figsize=(8, 6))

    plt.scatter(
        plot_df["training_time_sec"],
        plot_df["f1_macro"]
    )

    for _, row in plot_df.iterrows():
        plt.annotate(
            row["model"],
            (
                row["training_time_sec"],
                row["f1_macro"]
            ),
            xytext=(5, 5),
            textcoords="offset points"
        )

    plt.xlabel("Total training time (s)")
    plt.ylabel("Macro F1")
    plt.title("Training Time vs. Macro F1")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    output_path = RESULTS_DIR / "training_time_vs_f1.png"
    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    print(f"Saved plot: {output_path}")

    plt.show()


# ============================================================
# 2. GPU memory vs F1 — DL only
# ============================================================

dl_df = df[
    df["model"].isin(DL_MODELS)
].dropna(
    subset=["gpu_memory_mb", "f1_macro"]
)

if not dl_df.empty:

    plt.figure(figsize=(8, 6))

    plt.scatter(
        dl_df["gpu_memory_mb"],
        dl_df["f1_macro"]
    )

    for _, row in dl_df.iterrows():
        plt.annotate(
            row["model"],
            (
                row["gpu_memory_mb"],
                row["f1_macro"]
            ),
            xytext=(5, 5),
            textcoords="offset points"
        )

    plt.xlabel("Peak GPU memory (MB)")
    plt.ylabel("Macro F1")
    plt.title("GPU Memory vs. Macro F1 — Deep Learning Models")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    output_path = RESULTS_DIR / "gpu_memory_vs_f1_dl.png"
    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    print(f"Saved plot: {output_path}")

    plt.show()
else:
    print("\nNo DL results with GPU memory information were found.")