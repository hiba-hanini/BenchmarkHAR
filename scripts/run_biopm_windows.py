import argparse
from src.utils.load_config import load_config, list_available_datasets
from src.preprocessing.run_preprocessing_biopm import preprocessing_biopm


def main():
    
    parser = argparse.ArgumentParser(description="Benchmark HAR -generic pipeline ")
    
    parser.add_argument(
        "--dataset", type=str, required=True,
        help=f"Available : {list_available_datasets()}"
    )

    parser.add_argument(
        "--step", type=str, default="full", choices=["preprocessing"]
    )

    args = parser.parse_args()


    data_config = load_config(dataset_name=args.dataset)

    print(f"--- Preprocessing : {args.dataset} ---")
    preprocessing_biopm(args.dataset, data_config)


if __name__ == "__main__":
    main()