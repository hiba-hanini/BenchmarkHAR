from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt

results_dir = Path(
"/lustre09/project/6081099/haninih/BenchmarkHAR/data/models/pamap2/astar_har/758cead5928dcb741c4ded755822d61b4e59860125c7a4770bfa85188c8408c4"
)

files = sorted(results_dir.glob("fold_*_history.json"))

print(f"Found {len(files)} history files")

if not files:
    raise FileNotFoundError(
        f"No fold_*_history.json found in {results_dir}"
    )

all_losses = []

plt.figure(figsize=(8, 5))

for file in files:
    with open(file, "r") as f:
        data = json.load(f)

    loss = data["history"]["loss"]
    fold_idx = data["fold_index"]

    all_losses.append(loss)

    plt.plot(
        range(1, len(loss) + 1),
        loss,
        alpha=0.35,
        label=f"Fold {fold_idx}"
    )

# Mean and standard deviation
min_epochs = min(len(loss) for loss in all_losses)

losses = np.array([
    loss[:min_epochs]
    for loss in all_losses
])

mean_loss = losses.mean(axis=0)
std_loss = losses.std(axis=0)

epochs = np.arange(1, min_epochs + 1)

plt.plot(
    epochs,
    mean_loss,
    linewidth=2,
    label="Mean"
)

plt.fill_between(
    epochs,
    mean_loss - std_loss,
    mean_loss + std_loss,
    alpha=0.2
)

plt.xlabel("Epoch")
plt.ylabel("Training Loss")
plt.title("Training Loss Across PAMAP2 LOSO Folds")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

output_path = results_dir / "training_loss_loso.png"
plt.savefig(output_path, dpi=300)

print(f"Figure saved to: {output_path}")

plt.show()