from pathlib import Path
import yaml

CONFIGS_DIR = Path("configs")


def load_reproducibility_config() -> dict:
    """Load experiment-wide reproducibility settings with safe defaults."""
    config = {"seed": 42, "deterministic": True}
    path = CONFIGS_DIR / "reproducibility.yaml"
    if path.exists():
        with open(path) as f:
            config.update((yaml.safe_load(f) or {}).get("reproducibility", {}))
    return config


def load_config(dataset_name: str = None, model_name: str = None) -> dict:

    config = {}

    if dataset_name:
        dataset_path = CONFIGS_DIR / "datasets" / f"{dataset_name}.yaml"
        if not dataset_path.exists():
            raise FileNotFoundError(
                f"Available Datasets : {list_available_datasets()}"
            )
        with open(dataset_path) as f:
            config.update(yaml.safe_load(f) or {})

    if model_name:
        model_path = CONFIGS_DIR / "models" / f"{model_name}.yaml"
        if model_path.exists():
            with open(model_path) as f:
                config["model"] = yaml.safe_load(f) or {}

    return config


def list_available_datasets() -> list:
    return sorted(p.stem for p in (CONFIGS_DIR / "datasets").glob("*.yaml"))
