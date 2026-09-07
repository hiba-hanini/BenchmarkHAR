import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


def canonical_label_metadata(config_data, label_col):
    """Return the configuration-defined canonical label vocabulary.

    ``label_mapping`` maps raw labels to either already-numeric class IDs or
    semantic names. Semantic names are assigned stable IDs in YAML insertion
    order, never from samples in a train or test fold.

    The configured ``unknown_labels`` list and ``unknown_label_policy`` are kept
    explicit so unseen labels cannot silently change the benchmark vocabulary.
    """
    source_mapping = config_data.get("label_mapping", {}).get(label_col)
    unknown_labels = config_data.get("unknown_labels", [])
    if isinstance(unknown_labels, str):
        unknown_labels = [unknown_labels]
    unknown_policy = config_data.get("unknown_label_policy", "raise")

    if source_mapping:
        source_mapping = {str(raw): value for raw, value in source_mapping.items()}
        values = list(dict.fromkeys(source_mapping.values()))
        if all(isinstance(value, (int, np.integer)) for value in values):
            value_to_id = {value: int(value) for value in values}
        else:
            value_to_id = {value: index for index, value in enumerate(values)}
        raw_to_id = {raw: value_to_id[value] for raw, value in source_mapping.items()}
        id_to_label = {class_id: str(value) for value, class_id in value_to_id.items()}
        return {
            "prediction_label": label_col,
            "raw_to_canonical": raw_to_id,
            "canonical_to_label": id_to_label,
            "class_ids": list(id_to_label),
            "unknown_labels": list(unknown_labels),
            "unknown_label_policy": unknown_policy,
        }

    class_ids = [int(value) for value in config_data.get("canonical_label_values", [])]
    if not class_ids:
        raise ValueError(
            f"Dataset configuration must define label_mapping or canonical_label_values for '{label_col}'."
        )
    return {
        "prediction_label": label_col,
        "raw_to_canonical": {str(value): value for value in class_ids},
        "canonical_to_label": {value: str(value) for value in class_ids},
        "class_ids": class_ids,
        "unknown_labels": list(unknown_labels),
        "unknown_label_policy": unknown_policy,
    }


def canonicalize_labels(labels, label_col, config_data) -> np.ndarray:
    """Map raw or already-canonical labels using configuration only.

    Unknown or unmapped labels are rejected by default so the benchmark task
    cannot silently expand the class vocabulary from test-only observations.
    """
    metadata = canonical_label_metadata(config_data, label_col)
    values = pd.Series(labels)
    if values.empty:
        return np.asarray([], dtype=np.int32)

    class_ids = set(int(class_id) for class_id in metadata["class_ids"])
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.notna().all():
        numeric_values = numeric.astype(np.int64).to_numpy()
        if set(int(value) for value in numeric_values).issubset(class_ids):
            return numeric_values.astype(np.int32)

    raw_lookup = metadata["raw_to_canonical"]
    mapped = values.astype(str).map(lambda value: raw_lookup.get(str(value), np.nan))
    if mapped.isna().any():
        unknown_values = values[mapped.isna()].astype(str).unique().tolist()
        policy = metadata.get("unknown_label_policy", "raise")
        if policy == "ignore":
            return mapped.astype(np.float64).to_numpy()
        raise ValueError(f"Unmapped labels for '{label_col}': {unknown_values}")
    return mapped.astype(np.int32).to_numpy()


def build_canonical_label_encoder(config_data, label_col) -> LabelEncoder:
    """Construct, rather than fit, an encoder from configured class IDs."""
    metadata = canonical_label_metadata(config_data, label_col)
    encoder = LabelEncoder()
    encoder.classes_ = np.asarray(metadata["class_ids"], dtype=np.int32)
    return encoder


def map_labels(df, label_col, config_data):
    return pd.Series(canonicalize_labels(df[label_col], label_col, config_data), index=df.index)
