import argparse

import datasets
import numpy as np
from combined_dataset import ALL_DATASETS, DATA_PATH, ROOT_PATH

INDEX_PATH = ROOT_PATH / "data" / "datasets" / "indexes"

SEED = 123


def create_train_val_splits(val_size):
    for dataset_name in ALL_DATASETS:
        if (INDEX_PATH / dataset_name).exists():
            print(f"{dataset_name} split already exists, skipping...")
            continue
        ds = datasets.load_from_disk(DATA_PATH / dataset_name)["train"]
        np.random.seed(SEED)
        indexes = np.random.permutation(len(ds))
        val_threshold = int(len(ds) * val_size)
        val_indexes = indexes[:val_threshold]
        train_indexes = indexes[val_threshold:]
        data_dir = INDEX_PATH / dataset_name
        data_dir.mkdir(exist_ok=True, parents=True)
        np.save(data_dir / "train_indexes.npy", train_indexes)
        np.save(data_dir / "validation_indexes.npy", val_indexes)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Create train val splits from train datasets")
    parser.add_argument(
        "--val-size",
        default=0.10,
        type=float,
        help="Value in [0,1] corresponding to val split proportion",
    )
    args = parser.parse_args()
    create_train_val_splits(args.val_size)
