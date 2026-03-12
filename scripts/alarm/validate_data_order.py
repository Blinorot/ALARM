import argparse

import datasets
import numpy as np
from combined_dataset import ALL_DATASETS
from joblib import Parallel, delayed
from tqdm.auto import tqdm

from utils import DATA_PATH, PROMPT_PATH

SEED = 123


def validate_data_order():
    """
    Validates that raw and raw_with_prompts datasets are in the same order.
    """
    for dataset_name in ALL_DATASETS:
        ds = datasets.load_from_disk(DATA_PATH / dataset_name)["train"]
        prompt_name = f"{dataset_name}_with_prompts"
        prompt_ds = datasets.load_from_disk(PROMPT_PATH / prompt_name)["train"]

        print("Lengths:", len(ds), len(prompt_ds))
        assert len(ds) == len(prompt_ds), "Datasets are of different lengths"

        for elem, prompt_elem in tqdm(
            zip(ds, prompt_ds), desc=f"{dataset_name}", total=len(ds)
        ):
            assert elem["dataset_index"] == prompt_elem["dataset_index"]
            assert elem["audio_description"] == prompt_elem["audio_description"]


if __name__ == "__main__":
    validate_data_order()
