import argparse

import datasets
import numpy as np
from combined_dataset import ALL_DATASETS, DATA_PATH, ROOT_PATH
from joblib import Parallel, delayed
from tqdm.auto import tqdm

INDEX_FILTERED_PATH = ROOT_PATH / "data" / "datasets" / "indexes_filtered"
INDEX_PATH = ROOT_PATH / "data" / "datasets" / "indexes"


SEED = 123


def validate_elem(elem, max_duration):
    try:
        # use range to make it faster
        _ = elem["audio"].get_samples_played_in_range(0, 0.01)
    except RuntimeError as e:
        return False
    else:
        if elem["duration"] >= max_duration:
            return False
        return True


def process_dataset_indexes(ds, indexes, n_jobs=16, max_duration=60.0):
    indexes_mask = Parallel(n_jobs=n_jobs, backend="threading")(
        delayed(validate_elem)(elem, max_duration) for elem in tqdm(ds)
    )
    return indexes[indexes_mask]


def validate_and_filter_indexes(n_jobs, max_duration):
    """
    Validates audio files and removes indexes that have broken or too long audio,
    which were not detected previously, to ensure that there are
    no errors during training.
    """
    for dataset_name in ALL_DATASETS:
        if (INDEX_FILTERED_PATH / dataset_name).exists():
            print(f"{dataset_name} filtered split already exists, skipping...")
            continue
        ds = datasets.load_from_disk(DATA_PATH / dataset_name)["train"]
        data_dir = INDEX_PATH / dataset_name
        data_filtered_dir = INDEX_FILTERED_PATH / dataset_name
        data_filtered_dir.mkdir(exist_ok=True, parents=True)

        train_indexes = np.load(data_dir / "train_indexes.npy")
        val_indexes = np.load(data_dir / "validation_indexes.npy")

        # sort for faster elem access
        train_indexes = np.sort(train_indexes)
        val_indexes = np.sort(val_indexes)

        train_split = ds.select(train_indexes)
        val_split = ds.select(val_indexes)

        filtered_train_indexes = process_dataset_indexes(
            train_split,
            train_indexes,
            n_jobs=n_jobs,
            max_duration=max_duration,
        )
        filtered_val_indexes = process_dataset_indexes(
            val_split, val_indexes, n_jobs=n_jobs, max_duration=max_duration
        )

        print(f"Filtered {dataset_name}")
        print(
            f"Original/Filtered train: {train_indexes.shape}/{filtered_train_indexes.shape}"
        )
        print(
            f"Original/Filtered val: {val_indexes.shape}/{filtered_val_indexes.shape}"
        )

        np.save(data_filtered_dir / "train_indexes.npy", filtered_train_indexes)
        np.save(data_filtered_dir / "validation_indexes.npy", filtered_val_indexes)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Validate audio files in datasets")
    parser.add_argument(
        "--n-jobs",
        default=16,
        type=int,
        help="Number of joblib workers",
    )
    parser.add_argument(
        "--max-duration",
        default=60.0,
        type=float,
        help="Max allowed duration for an audio. (Default: 60s)",
    )
    args = parser.parse_args()
    validate_and_filter_indexes(n_jobs=args.n_jobs, max_duration=args.max_duration)
