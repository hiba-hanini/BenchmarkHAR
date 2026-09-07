import argparse
import subprocess
import sys

from src.utils.load_config import list_available_datasets

# These lists should be kept in sync with run.py
ML_MODELS = ["random_forest", "logistic_regression"]
DL_MODELS = ["cnn", "lstm", "dnn", "rnn", "deepconv_lstm", "astar_har", "tinyhar"]
#  "tinyhar", "astar_har"
FOUNDATION_MODELS = ["biopm", "biopm_rf"]
ALL_MODELS = ML_MODELS + DL_MODELS + FOUNDATION_MODELS

def run_all(datasets_to_run, models_to_run, max_subjects=None, seed=None, deterministic=None):
    """
    Runs the full training pipeline for specified datasets and models.
    """
    if not datasets_to_run:
        datasets_to_run = list_available_datasets()
    
    if not models_to_run:
        models_to_run = ALL_MODELS

    print(f"Will run on datasets: {datasets_to_run}")
    print(f"Will run with models: {models_to_run}")
    if max_subjects is not None:
        print(f"Will limit preprocessing to the first {max_subjects} subjects")

    total_runs = len(datasets_to_run) * len(models_to_run)
    current_run = 0

    for dataset in datasets_to_run:
        for model in models_to_run:
            current_run += 1
            print("\n" + "="*80)
            print(f"  RUN {current_run}/{total_runs}: model='{model}' on dataset='{dataset}'")
            print("="*80 + "\n")

            command = [
                sys.executable, "scripts/run.py",
                "--dataset", dataset,
                "--model", model,
                "--step", "full"
            ]
            if max_subjects is not None:
                command.extend(["--max-subjects", str(max_subjects)])
            if seed is not None:
                command.extend(["--seed", str(seed)])
            if deterministic is not None:
                command.extend(["--deterministic", str(deterministic)])

            try:
                subprocess.run(command, check=True, text=True)
                print(f"\nSUCCESS: model='{model}' on dataset='{dataset}' completed.")
            except subprocess.CalledProcessError as e:
                print(f"\nERROR: model='{model}' on dataset='{dataset}' failed with exit code {e.returncode}.")
                print("Continuing with the next run...")
                continue
            except FileNotFoundError:
                print(f"Error: Python executable '{sys.executable}' not found.")
                return

    print("\n" + "="*80)
    print("  RUN SUMMARY")
    print("="*80 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run all models on all datasets.")
    parser.add_argument(
        "--datasets", nargs='*', default=None,
        help="Specify which datasets to run. If not provided, runs on all available datasets."
    )
    parser.add_argument(
        "--models", nargs='*', default=None,
        help="Specify which models to run. If not provided, runs all models."
    )
    parser.add_argument(
        "--max-subjects", type=int, default=None,
        help="Limit preprocessing to the first N subjects for each dataset/model run."
    )
    parser.add_argument("--seed", type=int, default=None,
                        help="Experiment seed passed to every run.py invocation.")
    parser.add_argument("--deterministic", type=lambda value: value.lower() in {"1", "true", "yes", "on"}, default=None,
                        help="Force deterministic mode for each run.py invocation.")
    args = parser.parse_args()

    run_all(args.datasets, args.models, max_subjects=args.max_subjects, seed=args.seed, deterministic=args.deterministic)
