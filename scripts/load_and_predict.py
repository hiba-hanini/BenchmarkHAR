from pathlib import Path
import argparse
import json

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score

from src.utils.evaluation import compute_fold_metrics

from src.models.registry import get_model_class
from src.utils.load_windows import _load_all_windows
from src.utils.cv import get_cv_splitter
from src.utils.reproducibility import DEFAULT_SEED
from src.preprocessing.mapping import build_canonical_label_encoder, canonicalize_labels
import joblib



ML_MODELS = ["random_forest", "logistic_regression"]
DL_MODELS_KERAS = ["cnn", "lstm", "dnn", "rnn", "deepconv_lstm", "astar_har"]
DL_MODELS_PYTORCH = ["tinyhar"]
ALL_DL_MODELS = DL_MODELS_KERAS + DL_MODELS_PYTORCH


def load_and_predict(data_name, config_data, model_name, config_model, fold_index,
                     seed=DEFAULT_SEED):
    model_class = get_model_class(model_name)
    if model_class is None:
        raise ValueError(f"Model '{model_name}' not found in the registry.")

    cv_config = config_model.get("cv", {})

    if model_name in ML_MODELS:
        df_features_path = f"data/features/{data_name}_features_all.pkl"
        df_all = pd.read_pickle(df_features_path)

        label_col = config_data["label_to_predict"]
        le = build_canonical_label_encoder(config_data, label_col)
        df_all[label_col] = le.transform(canonicalize_labels(df_all[label_col], label_col, config_data))

        drop_cols = ["subject_id"] + config_data["label_cols"]
        X_all = df_all.drop(columns=drop_cols)
        y_all = df_all[config_data["label_to_predict"]]
        groups = df_all["subject_id"].values

        n_subjects = len(np.unique(groups))
        splitter, _ = get_cv_splitter(n_subjects, cv_config, seed=seed)
        splits = list(splitter.split(X_all, y_all, groups=groups))
        if fold_index >= len(splits):
            raise ValueError(f"Fold index {fold_index} is out of bounds for {len(splits)} splits.")
        
        _, test_idx = splits[fold_index]
        X_test, y_test = X_all.iloc[test_idx], y_all.iloc[test_idx]
        selector_path = Path(
            f"data/models/{data_name}/{model_name}/seed_{seed}/selected_features_fold_{fold_index}.json"
        )
        if not selector_path.exists():
            raise FileNotFoundError(f"Fold feature selector not found: {selector_path}")
        selected_features = json.loads(selector_path.read_text())
        missing_features = set(selected_features) - set(X_test.columns)
        if missing_features:
            raise ValueError(f"Saved selector references missing features: {sorted(missing_features)}")
        X_test = X_test.loc[:, selected_features]

    elif model_name in ALL_DL_MODELS:
        from src.models.deep.architectures import apply_standardize, prepare_input
        X_all, y_all, groups = _load_all_windows(config_data)
        le = joblib.load(Path(f"data/models/{data_name}/{model_name}/seed_{seed}/label_encoder.joblib"))
        y_all = le.transform(y_all)

        X_all = prepare_input(X_all, model_name)

        n_subjects = len(np.unique(groups))
        splitter, _ = get_cv_splitter(n_subjects, cv_config, seed=seed)
        splits = list(splitter.split(X_all, y_all, groups=groups))
        if fold_index >= len(splits):
            raise ValueError(f"Fold index {fold_index} is out of bounds for {len(splits)} splits.")

        _, test_idx = splits[fold_index]
        X_test, y_test = X_all[test_idx], y_all[test_idx]

        scaler_path = Path(f"data/models/{data_name}/{model_name}/seed_{seed}/scaler_fold_{fold_index}.joblib")
        if scaler_path.exists():
            scaler = joblib.load(scaler_path)
            X_test = apply_standardize(model_name, X_test, scaler)
    else:
        raise ValueError(f"Unknown model type for '{model_name}'")
        
    if model_name in ML_MODELS:    
        model_path = Path(f"data/models/{data_name}/{model_name}/seed_{seed}/fold_{fold_index}.joblib")
    elif model_name in DL_MODELS_KERAS:
        model_path = Path(f"data/models/{data_name}/{model_name}/seed_{seed}/fold_{fold_index}.keras")
    elif model_name in DL_MODELS_PYTORCH:
        model_path = Path(f"data/models/{data_name}/{model_name}/seed_{seed}/fold_{fold_index}.pt")
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found : {model_path}")

    model_config = dict(config_model)
    model_config["seed"] = seed
    print(model_config)
    model = model_class(model_config)
    model.load(model_path)


    y_pred = model.predict(X_test)

    y_test_decoded = le.inverse_transform(y_test)
    y_pred_decoded = le.inverse_transform(y_pred)

    metrics = compute_fold_metrics(
        y_test, y_pred,
        canonical_classes=list(range(len(le.classes_))),
        zero_division=0,
    )
    f1 = metrics["f1_macro"]
    acc = metrics["accuracy"]

    print(f"[RELOAD CHECK] {data_name} | {model_name} | fold_index={fold_index}")
    print(f"  n_test = {len(X_test)}")
    print(f"  F1 Macro = {f1:.4f} | Accuracy = {acc:.4f}")

    return {
        "f1_macro": f1,
        "accuracy": acc,
        "present_classes": metrics["present_classes"],
        "f1_macro_canonical": metrics.get("f1_macro_canonical"),
        "y_true": y_test_decoded,
        "y_pred": y_pred_decoded,
    }


if __name__ == "__main__":
    import yaml

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--fold-index", required=True, type=int, help="Index of the fold to load (from 0).")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help="Seed namespace used when the model was trained.")
    args = parser.parse_args()

    with open(f"configs/datasets/{args.dataset}.yaml") as f:
        config_data = yaml.safe_load(f)
    with open(f"configs/models/{args.model}.yaml") as f:
        config_model = yaml.safe_load(f)

    load_and_predict(args.dataset, config_data, args.model, config_model, args.fold_index, args.seed)
