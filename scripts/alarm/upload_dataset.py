import argparse
import re
from pathlib import Path

import datasets
import numpy as np
from combined_dataset import ALL_DATASETS, ROOT_PATH


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
    # text_field is always named "context"
    if "prompt" in all_columns:
        dataset = dataset.rename_column("prompt", "context")
    # audio text description (or description of its content)
    # is always named "audio_description"
    if "question" in all_columns:
        dataset = dataset.rename_column("question", "audio_description")
    dataset = dataset.select_columns(
        [
            "dataset_name",
            "audio",
            "context",
            "audio_description",
            "llm_answer_with_context",
            "llm_answer_no_context",
            "unique_id_1",
            "unique_id_2",
            "dataset_index",
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


def load_merged_dataset(full_dataset_name, split, data_dir="data", use_indexes=True):
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

    if use_indexes and indexes_path.exists():
        split_path = indexes_path / f"{split}_indexes.npy"
        indexes = np.load(split_path)
        indexes = np.sort(indexes)  # for faster getitem
        base_dataset = base_dataset["train"].select(indexes)
    else:
        base_dataset = base_dataset[split]

    base_dataset = base_dataset.add_column(
        "dataset_name", [base_dataset_name] * len(base_dataset)
    )
    base_dataset = select_required_columns(base_dataset)

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


def get_response_name(
    dataset_name,
    rephrase,
    use_checker,
    actual_model_name,
    max_tokens,
    max_thinking_tokens,
):
    response_name = f"{dataset_name}"
    if rephrase:
        response_name += "_rephrased"
        if max_thinking_tokens >= 0:
            response_name += f"_{max_thinking_tokens}"
    if use_checker:
        response_name += "_filtered"
    response_name += f"_{actual_model_name}_{max_tokens}"
    return response_name


def get_train_and_val_datasets(args, use_indexes=True):
    model_name = args.model_name.split("/")[-1]

    train_ds_list = []
    for dataset_name in ALL_DATASETS:
        full_dataset_name = get_response_name(
            dataset_name,
            args.rephrase,
            args.use_checker,
            model_name,
            args.max_tokens,
            args.max_thinking_tokens,
        )
        ds = load_merged_dataset(
            full_dataset_name,
            split="train",
            data_dir=args.data_dir,
            use_indexes=use_indexes,
        )
        train_ds_list.append(ds)

    heysquad_name = "heysquad_human_filtered" if use_indexes else "heysquad_human"

    for dataset_name in [heysquad_name, "instructs2s"]:
        full_dataset_name = get_response_name(
            dataset_name, False, False, model_name, args.max_tokens, None
        )
        ds = load_merged_dataset(
            full_dataset_name,
            split="train",
            data_dir=args.data_dir,
            use_indexes=use_indexes,
        )
        train_ds_list.append(ds)
    train_ds = datasets.concatenate_datasets(train_ds_list)

    if use_indexes:
        val_ds_list = []
        for dataset_name in ALL_DATASETS:
            full_dataset_name = get_response_name(
                dataset_name,
                args.rephrase,
                args.use_checker,
                model_name,
                args.max_tokens,
                args.max_thinking_tokens,
            )
            ds = load_merged_dataset(
                full_dataset_name,
                split="validation",
                data_dir=args.data_dir,
                use_indexes=use_indexes,
            )
            val_ds_list.append(ds)

    full_dataset_name = get_response_name(
        heysquad_name, False, False, model_name, args.max_tokens, None
    )
    val_ds = load_merged_dataset(
        full_dataset_name,
        split="validation",
        data_dir=args.data_dir,
        use_indexes=use_indexes,
    )
    if use_indexes:
        val_ds_list.append(val_ds)
        val_ds = datasets.concatenate_datasets(val_ds_list)

    return train_ds, val_ds


def combine_datasets(args) -> datasets.DatasetDict:
    train_ds, val_ds = get_train_and_val_datasets(args, use_indexes=False)
    filtered_train_ds, filtered_val_ds = get_train_and_val_datasets(
        args, use_indexes=True
    )

    full_ds_dict = {
        "raw_train": train_ds,
        "raw_validation": val_ds,
        "train": filtered_train_ds,
        "validation": filtered_val_ds,
    }

    full_ds = datasets.DatasetDict(full_ds_dict)
    print(full_ds)
    return full_ds


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Upload dataset to HuggingFace")
    parser.add_argument(
        "--model-name",
        default="Qwen/Qwen3-4B-Thinking-2507",
        type=str,
        help="LLM name (Default: Qwen/Qwen3-4B-Thinking-2507)",
    )
    parser.add_argument(
        "--data-dir",
        default="data",
        type=str,
        help="data dir name (Default: data)",
    )
    parser.add_argument(
        "--max-tokens",
        default=512,
        type=int,
        help="Max tokens for generation (Default: 512)",
    )
    parser.add_argument(
        "--max-thinking-tokens",
        default=1536,
        type=int,
        help="Max tokens for internal thinking (Default: 1536)",
    )
    parser.add_argument(
        "--no-use-checker",
        dest="use_checker",
        action="store_false",
        help="Disable the checker",
    )
    parser.set_defaults(use_checker=True)
    parser.add_argument(
        "--no-rephrase",
        dest="rephrase",
        action="store_false",
        help="Disable the rephrase stage",
    )
    parser.set_defaults(rephrase=True)

    parser.add_argument(
        "--username",
        default="Blinorot",
        type=str,
        help="HF Username",
    )

    parser.add_argument(
        "--repo-name",
        default="ALARM-Corpora",
        type=str,
        help="HF repo name",
    )

    parser.add_argument(
        "--public",
        dest="public",
        action="store_true",
        help="Make the dataset public",
    )
    parser.set_defaults(public=False)

    parser.add_argument(
        "--remove-audio",
        dest="remove_audio",
        action="store_true",
        help="Remove audio before upload",
    )
    parser.set_defaults(remove_audio=False)

    args = parser.parse_args()

    full_ds = combine_datasets(args)

    if args.remove_audio:
        full_ds = full_ds.remove_columns("audio")

    full_ds.push_to_hub(
        repo_id=f"{args.username}/{args.repo_name}",
        private=not args.public,
    )
