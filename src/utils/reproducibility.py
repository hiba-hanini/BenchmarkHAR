"""Experiment-level controls for reproducible BenchmarkHAR runs.

``PYTHONHASHSEED`` is recorded in the environment for child processes, but
Python reads it while starting the interpreter. Calling this module at runtime
cannot change the hash randomisation of the current process. Set it before
launching Python when that level of reproducibility is needed.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np


DEFAULT_SEED = 42
DEFAULT_DETERMINISTIC = True


def _normalize_for_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(k): _normalize_for_json(v)
            for k, v in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_normalize_for_json(v) for v in value]
    if isinstance(value, set):
        return sorted(_normalize_for_json(v) for v in value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def set_global_seed(seed: int, deterministic: bool = True) -> None:
    """Seed installed RNGs without requiring TensorFlow or PyTorch."""
    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:
        import tensorflow as tf
    except ImportError:
        pass
    else:
        tf.random.set_seed(seed)
        if hasattr(tf.keras, "utils"):
            try:
                tf.keras.utils.set_random_seed(seed)
            except Exception:
                pass
        if deterministic:
            enable_determinism = getattr(tf.config.experimental, "enable_op_determinism", None)
            if enable_determinism is not None:
                enable_determinism()

    try:
        import torch
    except ImportError:
        pass
    else:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            use_deterministic_algorithms = getattr(torch, "use_deterministic_algorithms", None)
            if use_deterministic_algorithms is not None:
                try:
                    use_deterministic_algorithms(True, warn_only=True)
                except TypeError:
                    use_deterministic_algorithms(True)


def get_environment_info() -> dict[str, Any]:
    """Return lightweight environment metadata, avoiding optional imports."""
    import platform

    info: dict[str, Any] = {
        "python": platform.python_version(),
        "numpy": np.__version__,
    }
    try:
        import sklearn
        info["sklearn"] = sklearn.__version__
    except ImportError:
        pass
    try:
        import tensorflow as tf
        info["tensorflow"] = tf.__version__
    except ImportError:
        pass
    try:
        import torch
        info["torch"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
    except ImportError:
        info["cuda_available"] = False
    return info


def resolve_reproducibility_config(seed: int | None = None, deterministic: bool | None = None) -> dict[str, Any]:
    from src.utils.load_config import load_reproducibility_config

    config = load_reproducibility_config()
    if seed is not None:
        config["seed"] = int(seed)
    if deterministic is not None:
        config["deterministic"] = bool(deterministic)
    return config


def compute_experiment_id(experiment_config: Mapping[str, Any]) -> str:
    resolved = _normalize_for_json(dict(experiment_config))
    payload = json.dumps(resolved, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_experiment_manifest(
    *,
    dataset: str,
    model: str | None,
    seed: int,
    deterministic: bool,
    resolved_configuration: Mapping[str, Any],
    artifact_path: str | Path,
    artifact_type: str,
    experiment_id: str | None = None,
) -> dict[str, Any]:
    gid = experiment_id or compute_experiment_id({
        "dataset": dataset,
        "model": model,
        "seed": seed,
        "deterministic": deterministic,
        "resolved_configuration": resolved_configuration,
    })
    manifest = {
        "experiment_id": gid,
        "dataset": dataset,
        "model": model,
        "seed": int(seed),
        "deterministic": bool(deterministic),
        "resolved_configuration": _normalize_for_json(dict(resolved_configuration)),
        "git_commit": _read_git_commit(),
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "artifact_path": str(Path(artifact_path)),
        "artifact_type": artifact_type,
    }
    return manifest


def _read_git_commit() -> str:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode == 0 and completed.stdout.strip():
            return completed.stdout.strip()
    except Exception:
        pass
    return "unknown"


def manifest_matches(manifest_a: Mapping[str, Any], manifest_b: Mapping[str, Any]) -> bool:
    candidate_keys = [
        "experiment_id",
        "dataset",
        "model",
        "seed",
        "deterministic",
        "resolved_configuration",
    ]
    for key in candidate_keys:
        left = _normalize_for_json(manifest_a.get(key))
        right = _normalize_for_json(manifest_b.get(key))
        if left != right:
            return False
    return True


def save_manifest(manifest: Mapping[str, Any], manifest_path: str | Path) -> None:
    path = Path(manifest_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(_normalize_for_json(dict(manifest)), handle, indent=2, sort_keys=True)
        handle.write("\n")


def load_manifest(manifest_path: str | Path) -> dict[str, Any] | None:
    path = Path(manifest_path)
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception:
        return None


def is_manifest_reusable(artifact_path: str | Path, manifest_path: str | Path, current_manifest: Mapping[str, Any]) -> bool:
    artifact = Path(artifact_path)
    manifest_file = Path(manifest_path)
    if not artifact.exists() or not manifest_file.exists():
        return False
    manifest = load_manifest(manifest_file)
    if manifest is None:
        return False
    return manifest_matches(manifest, current_manifest)


def get_result_path(dataset: str, model: str, experiment_config: Mapping[str, Any], base_dir: str = "data/results") -> Path:
    experiment_id = experiment_config.get("experiment_id") or compute_experiment_id(experiment_config)
    return Path(base_dir) / dataset / model / experiment_id / "results.pkl"


def get_model_artifact_dir(dataset: str, model: str, experiment_id: str, base_dir: str = "data/models") -> Path:
    return Path(base_dir) / dataset / model / experiment_id


def get_result_manifest_path(dataset: str, model: str, experiment_config: Mapping[str, Any], base_dir: str = "data/results") -> Path:
    return get_result_path(dataset, model, experiment_config, base_dir=base_dir).parent / "manifest.json"


def get_model_manifest_path(dataset: str, model: str, experiment_id: str, base_dir: str = "data/models") -> Path:
    return get_model_artifact_dir(dataset, model, experiment_id, base_dir=base_dir) / "manifest.json"
