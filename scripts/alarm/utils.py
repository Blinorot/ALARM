from pathlib import Path

import datasets

ROOT_PATH = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT_PATH / "data" / "datasets" / "raw"
PROMPT_PATH = ROOT_PATH / "data" / "datasets" / "raw_with_prompts"
RESPONSE_PATH = ROOT_PATH / "data" / "datasets" / "responses"


def load_merged_dataset(dataset_name, model_name, max_tokens):
    base_path = DATA_PATH / dataset_name
    prompt_path = PROMPT_PATH / f"{dataset_name}_with_prompts"
    response_path = (
        RESPONSE_PATH / model_name / f"{dataset_name}_{model_name}_{max_tokens}"
    )
    base_dataset = datasets.load_from_disk(base_path)

    if prompt_path.exists():
        prompt_dataset = datasets.load_from_disk(prompt_path)
        base_dataset = merge_datasets(base_dataset, prompt_dataset)

    if response_path.exists():
        response_dataset = datasets.load_from_disk(response_path)
        base_dataset = merge_datasets(base_dataset, response_dataset)

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
