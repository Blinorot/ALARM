import re
from pathlib import Path

import datasets
import numpy as np

ROOT_PATH = Path(__file__).resolve().parents[2]


def parse_full_dataset_name(full_Dataset_name):
    pattern = (
        r"^"
        r"(?P<base>.+?)"  # base dataset (lazy)
        r"(?P<rephrased>_rephrased(?:_\d+)?)?"  # optional _rephrased or _rephrased_#
        r"(?P<filtered>_filtered)?"  # optional _filtered
        r"_(?P<model>[^_]+)"  # _model_name (lazy)
        r"_(?P<idx>\d+)$"  # final integer
    )
    match = re.match(pattern, full_Dataset_name, re.VERBOSE)
    if match:
        base_dataset_name = match.group("base")
        if match.group("filtered"):
            base_dataset_name += match.group("filtered")
        model_name = match.group("model")
    return base_dataset_name, model_name


def select_required_columns(dataset: datasets.Dataset):
    all_columns = dataset.column_names
    # take any split
    split = next(iter(all_columns))
    # text_field is always named "context"
    if "prompt" in all_columns[split]:
        dataset = dataset.rename_column("prompt", "context")
    # audio text description (or description of its content)
    # is always named "audio_description"
    if "question" in all_columns[split]:
        dataset = dataset.rename_column("question", "audio_description")
    dataset = dataset.select_columns(
        [
            "audio",
            "context",
            "audio_description",
            "llm_answer_with_context",
            "llm_answer_no_context",
        ]
    )
    return dataset


def get_all_paths(data_dir="data"):
    all_paths = {}
    all_paths["DATA_PATH"] = ROOT_PATH / data_dir / "datasets" / "raw"
    all_paths["PROMPT_PATH"] = ROOT_PATH / data_dir / "datasets" / "raw_with_prompts"
    all_paths["RESPONSE_PATH"] = ROOT_PATH / data_dir / "datasets" / "responses"
    all_paths["INDEXES_FILTERED_PATH"] = (
        ROOT_PATH / data_dir / "datasets" / "indexes_filtered"
    )
    all_paths["INDEXES_PATH"] = ROOT_PATH / data_dir / "datasets" / "indexes"
    return all_paths


def load_merged_dataset(full_dataset_name, split, data_dir="data"):
    base_dataset_name, model_name = parse_full_dataset_name(full_dataset_name)

    all_paths = get_all_paths(data_dir)

    base_path = all_paths["DATA_PATH"] / base_dataset_name
    indexes_path = all_paths["INDEXES_PATH"] / base_dataset_name
    indexes_filtered_path = all_paths["INDEXES_FILTERED_PATH"] / base_dataset_name
    if indexes_filtered_path.exists():
        indexes_path = indexes_filtered_path  # prefer filtered indexes

    prompt_path = all_paths["PROMPT_PATH"] / f"{base_dataset_name}_with_prompts"
    response_path = all_paths["RESPONSE_PATH"] / model_name / full_dataset_name
    base_dataset = datasets.load_from_disk(base_path)

    if prompt_path.exists():
        prompt_dataset = datasets.load_from_disk(prompt_path)
        base_dataset = merge_datasets(base_dataset, prompt_dataset)

    if response_path.exists():
        response_dataset = datasets.load_from_disk(response_path)
        base_dataset = merge_datasets(base_dataset, response_dataset)
        response_index_path = (
            all_paths["INDEXES_FILTERED_PATH"] / model_name / full_dataset_name
        )
        if response_index_path.exists():
            indexes_path = response_index_path  # prefer filtered response
    else:
        raise ValueError("Response path does not exist, dataset cannot be used")

    base_dataset = select_required_columns(base_dataset)

    if indexes_path.exists():
        split_path = indexes_path / f"{split}_indexes.npy"
        indexes = np.load(split_path)
        indexes = np.sort(indexes)  # for faster getitem
        base_dataset = base_dataset["train"].select(indexes)
    else:
        base_dataset = base_dataset[split]

    return base_dataset


def merge_datasets(dataset1, dataset2):
    """
    Row-wise merge of two datasets with the same order.
    Datasets are with splits.
    """
    merged_dataset = {}

    assert dataset1.keys() == dataset2.keys()

    for split in dataset1.keys():
        ds_split1 = dataset1[split]
        ds_split2 = dataset2[split]

        merged_dataset[split] = merge_splits(ds_split1, ds_split2)

    merged_dataset = datasets.DatasetDict(merged_dataset)
    return merged_dataset


def merge_splits(ds_split1, ds_split2):
    """
    Row-wise merge of two dataset splits with the same order.
    """
    # to be sure
    assert ds_split1[0]["dataset_index"] == ds_split2[0]["dataset_index"]
    assert ds_split1[-1]["dataset_index"] == ds_split2[-1]["dataset_index"]

    ds_split1, ds_split2 = remove_duplicated_columns(ds_split1, ds_split2)
    merged_split = datasets.concatenate_datasets([ds_split1, ds_split2], axis=1)
    return merged_split


def remove_duplicated_columns(ds_split1, ds_split2):
    columns1 = set(ds_split1.column_names)
    columns2 = set(ds_split2.column_names)

    new_columns2 = [elem for elem in (columns2 - columns1)]
    ds_split2 = ds_split2.select_columns(new_columns2)
    return ds_split1, ds_split2
