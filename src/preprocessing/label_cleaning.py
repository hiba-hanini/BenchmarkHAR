import os
from pathlib import Path
import joblib
from sklearn.preprocessing import LabelEncoder


def clean_labels(df, dataset_name, label_cols, limit_rows, unknown_labels=None, encoder_dir = "encoders"):
    """ffill/bfill for missing labels with a limit_rows"""
    encoder_dir = Path("data") / "encoders" / dataset_name
    encoder_dir.mkdir(parents=True, exist_ok=True)

    for col in label_cols:
        if unknown_labels:
            df[col] = df[col].astype(str).replace(unknown_labels, "Unknown")
        df[col] = df.groupby("subject_id")[col].ffill(limit=limit_rows)
        df[col] = df.groupby("subject_id")[col].bfill(limit=limit_rows)
        df[col] = df[col].fillna("Unknown").astype(str).astype(object)


    return df