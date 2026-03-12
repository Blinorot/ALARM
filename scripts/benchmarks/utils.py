from pathlib import Path

import datasets

ROOT_PATH = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT_PATH / "data" / "datasets" / "raw"
PROMPT_PATH = ROOT_PATH / "data" / "datasets" / "raw_with_prompts"
RESPONSE_PATH = ROOT_PATH / "data" / "datasets" / "generated"


def load_merged_dataset(
    dataset_name, model_name, checkpoint_name, max_tokens, max_thinking_tokens, seed
):
    base_path = DATA_PATH / dataset_name
    response_name = get_response_name(
        dataset_name, checkpoint_name, max_tokens, max_thinking_tokens, seed
    )
    response_path = RESPONSE_PATH / model_name / response_name
    base_dataset = datasets.load_from_disk(base_path)

    response_dataset = datasets.load_from_disk(response_path)
    base_dataset = merge_datasets(base_dataset, response_dataset)

    return base_dataset


def get_response_name(
    dataset_name,
    checkpoint_name,
    max_tokens,
    max_thinking_tokens,
    seed,
):
    response_name = f"{dataset_name}"
    if isinstance(checkpoint_name, list):
        save_checkpoint_name = "_".join(
            [name.replace("/", "_") for name in checkpoint_name]
        )
    else:
        save_checkpoint_name = checkpoint_name.replace("/", "_")
    if seed is not None:
        save_checkpoint_name += f"_seed_{seed}"
    response_name += f"_{save_checkpoint_name}_{max_tokens}_{max_thinking_tokens}"
    return response_name


def merge_datasets(dataset1, dataset2):
    """
    Row-wise merge of two datasets with the same order.
    Datasets are with splits.
    """
    merged_dataset = {}

    for split in dataset1.keys():
        if split not in dataset2.keys():
            print(
                f"WARNING: split {split} is missing in the second dataset, skipping..."
            )
            continue
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
