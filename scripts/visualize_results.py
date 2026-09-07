import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def visualize_results(input_file="results_summary.csv"):
    """
    Generates and saves bar plots comparing model performance from a summary CSV file.
    """
    input_path = Path(input_file)
    if not input_path.exists():
        print(f"Input file not found: {input_path}")
        print("Please run aggregate_results.py first to generate the summary file.")
        return

    df = pd.read_csv(input_path)

    output_dir = Path("data/plots")
    output_dir.mkdir(parents=True, exist_ok=True)

    datasets = df["dataset"].unique()

    for dataset in datasets:
        plt.figure(figsize=(12, 8))
        sns.set_theme(style="whitegrid")

        df_dataset = df[df["dataset"] == dataset].sort_values("f1_macro_mean", ascending=False)

        ax = sns.barplot(
            x="f1_macro_mean",
            y="model",
            data=df_dataset,
            hue="model",
            palette="viridis",
            dodge=False,
            legend=False
        )

        ax.set_title(f"Model Performance on {dataset.upper()} Dataset", fontsize=16)
        ax.set_xlabel("Mean F1-Score (Macro)", fontsize=12)
        ax.set_ylabel("Model", fontsize=12)
        ax.set_xlim(0, max(1.0, df_dataset["f1_macro_mean"].max() * 1.15))

        for i, (value, std) in enumerate(zip(df_dataset["f1_macro_mean"], df_dataset["f1_macro_std"])):
            ax.text(value + 0.01, i, f"{value:.3f} ± {std:.3f}", color='black', va='center')

        plt.tight_layout()
        
        plot_path = output_dir / f"{dataset}_performance_comparison.png"
        plt.savefig(plot_path)
        print(f"Saved plot to {plot_path}")
        plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize aggregated model results.")
    parser.add_argument(
        "--input-file", type=str, default="results_summary.csv",
        help="Path to the aggregated results CSV file."
    )
    args = parser.parse_args()

    visualize_results(input_file=args.input_file)