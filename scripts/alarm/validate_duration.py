import argparse

import datasets
import numpy as np
from combined_dataset import ALL_DATASETS
from joblib import Parallel, delayed
from tqdm.auto import tqdm

from utils import DATA_PATH, PROMPT_PATH

SEED = 123


def validate_duration():
    """
    Prints max audio duration
    """
    for dataset_name in ALL_DATASETS:
        ds = datasets.load_from_disk(DATA_PATH / dataset_name)["train"]
        durations = np.array(ds["duration"])
        max_val_index = durations.argmax()
        max_val = durations[max_val_index]
        min_val_index = durations.argmin()
        min_val = durations[min_val_index]
        print(f"{dataset_name}: max duration {max_val} s at {max_val_index}")
        print(f"{dataset_name}: min duration {min_val} s at {min_val_index}")


if __name__ == "__main__":
    validate_duration()
