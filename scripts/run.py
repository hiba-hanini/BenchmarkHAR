import argparse
from src.utils.load_config import load_config, list_available_datasets, load_reproducibility_config
from src.utils.reproducibility import (
    compute_experiment_id,
    get_result_manifest_path,
    get_result_path,
    is_manifest_reusable,
    save_manifest,
    set_global_seed,
)
from pathlib import Path


ML_MODELS = ["random_forest", "logistic_regression"]
DL_MODELS = ["cnn", "lstm", "dnn", "rnn", "deepconv_lstm", "tinyhar", "astar_har"]
FOUNDATION_MODELS = ["biopm", "biopm_rf"]


def has_matching_result(path, seed, deterministic, dataset=None, model=None, resolved_config=None):
    """Only reuse a full-pipeline result produced with the same settings."""
    if not path.exists():
        return False
    import pandas as pd
    try:
        results = pd.read_pickle(path)
        if results.empty:
            return False
        match = (
            "seed" in results
            and "deterministic" in results
            and "train_subject_ids" in results
            and "test_subject_ids" in results
            and int(results["seed"].iloc[0]) == seed
            and bool(results["deterministic"].iloc[0]) == deterministic
        )
        if not match:
            return False
        if dataset is None or model is None or resolved_config is None:
            return True
        manifest_path = get_result_manifest_path(dataset, model, {"dataset": dataset, "model": model, "seed": seed, "deterministic": deterministic, "resolved_configuration": resolved_config})
        return manifest_path.exists() and is_manifest_reusable(path, manifest_path, {
            "dataset": dataset,
            "model": model,
            "seed": seed,
            "deterministic": deterministic,
            "resolved_configuration": resolved_config,
        })
    except Exception:
        return False


def run_preprocessing_model(dataset, data_config, model=None, max_subjects=None):
    if model in FOUNDATION_MODELS:
        from src.preprocessing.run_preprocessing_biopm import preprocessing_biopm
        preprocessing_biopm(dataset, data_config, max_subjects=max_subjects)
    else:
        from src.preprocessing.run_preprocessing import run_preprocessing
        run_preprocessing(dataset, data_config, max_subjects=max_subjects)


def preprocessing_already_done(data_config, model=None):
    if model in FOUNDATION_MODELS:
        out_dir_biopm = Path(data_config.get("out_dir_biopm", ""))
        if not out_dir_biopm.exists():
            return False
        return any(out_dir_biopm.glob("Data_MeLabel_*.h5"))

    out_dir = Path(data_config.get("out_dir", ""))
    if not out_dir.exists():
        return False
    return any(out_dir.glob("*.npz"))


def main():
    
    parser = argparse.ArgumentParser(description="Benchmark HAR -generic pipeline ")
    
    parser.add_argument(
        "--dataset", type=str, required=True,
        help=f"Available : {list_available_datasets()}"
    )

    parser.add_argument(
        "--step", type=str, default="full", choices=["preprocessing", "feature_extraction", "dimension_reduction", "run_model", "full", "run_model_rf"],
        help="Step to run"
    )

    parser.add_argument(
        "--model", type=str, choices=ML_MODELS + DL_MODELS + FOUNDATION_MODELS,
        help="Model to run (only for the 'run_model' step)"
    )

    parser.add_argument(
        "--max-subjects", type=int, default=None,
        help="Limit preprocessing to the first N subjects"
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Experiment seed; overrides configs/reproducibility.yaml"
    )
    parser.add_argument(
        "--deterministic",
        type=lambda value: value.lower() in {"1", "true", "yes", "on"},
        default=None,
        help="Force deterministic mode for this run.",
    )
    parser.add_argument(
        "--skip-preprocessing",
        action="store_true",
        help="Skip preprocessing if the window cache already exists for this dataset.",
    )

    args = parser.parse_args()


    reproducibility = load_reproducibility_config()
    seed = args.seed if args.seed is not None else reproducibility["seed"]
    deterministic = bool(args.deterministic if args.deterministic is not None else reproducibility.get("deterministic", True))
    set_global_seed(seed, deterministic)
    print("=" * 50)
    print("Reproducibility")
    print(f"Seed: {seed}")
    print(f"Deterministic mode: {deterministic}")
    print("=" * 50)

    data_config = load_config(dataset_name=args.dataset)
    if args.model is not None:
        model_config = load_config(model_name=args.model)["model"]
    else:
        model_config = None
    resolved_config = {
        "dataset": args.dataset,
        "model": args.model,
        "seed": seed,
        "deterministic": deterministic,
        "data": data_config,
        "model_config": model_config,
    }
    experiment_id = compute_experiment_id(resolved_config)
    print(f"Experiment ID: {experiment_id}")

    if args.step in ("preprocessing"):
        print(f"--- Preprocessing : {args.dataset} ---")
        run_preprocessing_model(args.dataset, data_config, args.model, max_subjects=args.max_subjects)

    if args.step in ("feature_extraction"):
        print(f"--- Feature Extraction : {args.dataset} ---")
        # if the preprocessing step has not been run, we need to run it first
        from src.features.run_feature_extraction import run_feature_extraction
        run_feature_extraction(args.dataset, data_config, max_subjects=args.max_subjects, experiment_id=experiment_id, resolved_config=resolved_config)

    if args.step in ("dimension_reduction"):
        print(f"--- Dimension Reduction : {args.dataset} ---")
        from src.dimension_reduction.run_dimension_reduction import run_correlation_filter
        run_correlation_filter(args.dataset, data_config)

    if args.step in ("full"):
        print(f"--- Full Pipeline : {args.model} on {args.dataset} ---")
        if args.model is None:
            raise ValueError("Please specify a model to run with --model")

        results_path = get_result_path(args.dataset, args.model, {
            "dataset": args.dataset,
            "model": args.model,
            "seed": seed,
            "deterministic": deterministic,
            "resolved_configuration": resolved_config,
            "experiment_id": experiment_id,
        })
        manifest_path = results_path.parent / "manifest.json"
        if has_matching_result(results_path, seed, deterministic, dataset=args.dataset, model=args.model, resolved_config=resolved_config):
            print(f"Matching results already exist at '{results_path}', skipping full pipeline.")
            return
        if results_path.exists():
            print(f"Existing results at '{results_path}' are stale or mismatched for this experiment; rerunning.")
        if manifest_path.exists():
            print(f"Manifest exists but does not match current experiment identity at '{manifest_path}'; rebuilding output.")

        skip_preprocessing = args.skip_preprocessing or preprocessing_already_done(data_config, model=args.model)
        if skip_preprocessing:
            print(f"--- Preprocessing already available for {args.dataset}; skipping. ---")
        else:
            print(f"--- Preprocessing : {args.dataset} ---")
            run_preprocessing_model(args.dataset, data_config, args.model, max_subjects=args.max_subjects)
        if args.model in ML_MODELS:
            from src.models.run_model_ml import run_model as run_model_ml
            if not (Path(f"data/features/{args.dataset}_features_all.pkl").exists()):
                print(f"--- Feature Extraction : {args.dataset} ---")
                from src.features.run_feature_extraction import run_feature_extraction
                run_feature_extraction(args.dataset, data_config, max_subjects=args.max_subjects)

            print(f"--- Run Model : {args.model} on {args.dataset} ---")
            run_model_ml(
                args.dataset,
                data_config,
                args.model,
                model_config,
                max_subjects=args.max_subjects,
                seed=seed,
                deterministic=deterministic,
                experiment_id=experiment_id,
                resolved_config=resolved_config,
            )
        elif args.model in FOUNDATION_MODELS:
            from src.models.run_biopm import run_biopm
            from src.models.run_biopm_rf import run_biopm_rf
            if args.model == "biopm":
                run_biopm(args.dataset, data_config, model_config, max_subjects=args.max_subjects,
                          seed=seed, deterministic=deterministic, experiment_id=experiment_id, resolved_config=resolved_config)
            elif args.model == "biopm_rf":
                run_biopm_rf(args.dataset, data_config, model_config, max_subjects=args.max_subjects,
                             seed=seed, deterministic=deterministic, experiment_id=experiment_id, resolved_config=resolved_config)
        elif args.model in DL_MODELS :
            from src.models.run_model_dl import run_model_dl
            run_model_dl(args.dataset, data_config, args.model, model_config, max_subjects=args.max_subjects,
                         seed=seed, deterministic=deterministic, experiment_id=experiment_id, resolved_config=resolved_config)
        else :
            raise ValueError({args.model}, " not found")
            

    
    if args.step in ("run_model"):
        print(f"--- Run Model : {args.model} on {args.dataset} ---")
        if args.model is None:
            raise ValueError("Please specify a model to run with --model")
        # if ML model : rum_model_ml, if DL model : run_model_dl, 
        if args.model in ML_MODELS:
            from src.models.run_model_ml import run_model as run_model_ml
            run_model_ml(args.dataset, data_config, args.model, model_config, max_subjects=args.max_subjects,
                         seed=seed, deterministic=deterministic, experiment_id=experiment_id, resolved_config=resolved_config)
        elif args.model in DL_MODELS:
            from src.models.run_model_dl import run_model_dl
            run_model_dl(args.dataset, data_config, args.model, model_config, max_subjects=args.max_subjects,
                         seed=seed, deterministic=deterministic, experiment_id=experiment_id, resolved_config=resolved_config)
        elif args.model in FOUNDATION_MODELS:
            from src.models.run_biopm import run_biopm
            from src.models.run_biopm_rf import run_biopm_rf
            if args.model == "biopm":
                run_biopm(args.dataset, data_config, model_config, max_subjects=args.max_subjects,
                          seed=seed, deterministic=deterministic, experiment_id=experiment_id, resolved_config=resolved_config)
            elif args.model == "biopm_rf":
                run_biopm_rf(args.dataset, data_config, model_config, max_subjects=args.max_subjects,
                             seed=seed, deterministic=deterministic, experiment_id=experiment_id, resolved_config=resolved_config)


if __name__ == "__main__":
    main()
