import argparse
from pathlib import Path
import pandas as pd
from src.utils.evaluation import compute_pooled_metrics

ALL_MODELS = [
    "random_forest", "logistic_regression", "cnn", "lstm", "dnn", "rnn",
    "deepconv_lstm", "tinyhar", "astar_har", "biopm", "biopm_rf"
]

# Sort by length descending to match longer names first (e.g., 'biopm_rf' before 'biopm')
ALL_MODELS.sort(key=len, reverse=True)


def _parse_result_path(path: Path, results_dir: Path):
    relative = path.relative_to(results_dir)
    parts = relative.parts

    if len(parts) >= 3 and parts[-1] == "results.pkl":
        dataset_name = parts[0]
        model_name = parts[1]
        return dataset_name, model_name

    file_name = path.name
    for model in ALL_MODELS:
        suffix = f"_{model}_results.pkl"
        if file_name.endswith(suffix):
            dataset_name = file_name[:-len(suffix)]
            return dataset_name, model
    return None, None


def aggregate_results(dataset_filter=None):
    """
    Aggregates results from all model runs, calculates summary statistics,
    and saves them to a CSV file.
    """
    results_dir = Path("data/results")
    if not results_dir.exists():
        print(f"Results directory not found: {results_dir}")
        return

    nested_result_files = sorted(results_dir.rglob("results.pkl"))
    legacy_result_files = sorted(results_dir.glob("*_results.pkl"))
    result_files = nested_result_files if nested_result_files else legacy_result_files

    if not result_files:
        print(f"No result files found in {results_dir}")
        return

    summary_data = []

    for f in result_files:
        dataset_name, model_name = _parse_result_path(f, results_dir)

        if not dataset_name or not model_name:
            print(f"Could not parse result path, skipping: {f}")
            continue

        if dataset_filter and dataset_name != dataset_filter:
            continue

        try:
            df = pd.read_pickle(f)
        except Exception as e:
            print(f"Could not read {f}: {e}")
            continue

        f1_mean = df["f1_macro"].mean()
        f1_std = df["f1_macro"].std()
        acc_mean = df["accuracy"].mean()
        acc_std = df["accuracy"].std()
        n_folds = len(df)
        cv_strategy = df["cv_strategy"].iloc[0] if "cv_strategy" in df.columns and not df.empty else "N/A"

        f1_pooled = None
        if "y_true" in df.columns and "y_pred" in df.columns and cv_strategy == "LOSO":
            try:
                pooled = compute_pooled_metrics(
                    df["y_true"].tolist(), df["y_pred"].tolist(), cv_strategy="LOSO"
                )
                f1_pooled = pooled["f1_macro_pooled"]
            except Exception:
                f1_pooled = None

        summary_row = {
            "dataset": dataset_name,
            "model": model_name,
            "f1_macro_mean": f1_mean,
            "f1_macro_std": f1_std,
            "accuracy_mean": acc_mean,
            "accuracy_std": acc_std,
            "n_folds": n_folds,
            "cv_strategy": cv_strategy,
        }
        if f1_pooled is not None:
            summary_row["f1_macro_pooled"] = f1_pooled

        summary_data.append(summary_row)

    if not summary_data:
        print("No data to aggregate with the current filter.")
        return

    summary_df = pd.DataFrame(summary_data).sort_values(by=["dataset", "model"]).reset_index(drop=True)

    print("--- Aggregated Results ---")
    print(summary_df.to_string())

    output_path = Path("results_summary.csv")
    summary_df.to_csv(output_path, index=False, float_format="%.4f")
    print(f"\nSummary saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aggregate results from all models.")
    parser.add_argument(
        "--dataset", type=str, default=None,
        help="Optional: filter results for a specific dataset."
    )
    args = parser.parse_args()

    aggregate_results(dataset_filter=args.dataset)