import argparse

import datasets
from combined_dataset import ALL_DATASETS, COLUMNS_TO_TAKE, ROOT_PATH


def load_instruction_ds(data_path, dataset_name, split):
    ds = datasets.load_from_disk(data_path / dataset_name)[split]
    ds = ds.rename_column("question", "audio_description")
    ds = ds.select_columns(COLUMNS_TO_TAKE)
    ds = ds.add_column("dataset_name", [dataset_name] * len(ds))
    return ds


def load_full_dataset_from_disk(args):
    data_path = ROOT_PATH / args.data_dir / "datasets" / "raw"

    ds_list = []
    for dataset_name in ALL_DATASETS:
        ds = datasets.load_from_disk(data_path / dataset_name)["train"]
        ds = ds.select_columns(COLUMNS_TO_TAKE)
        ds = ds.add_column("dataset_name", [dataset_name] * len(ds))
        ds_list.append(ds)

    heysquad_train = load_instruction_ds(data_path, "heysquad_human", "train")
    heysquad_val = load_instruction_ds(data_path, "heysquad_human", "validation")
    instructs2s_train = load_instruction_ds(data_path, "instructs2s", "train")

    # we concatenate all, filtration of indexes will be handled my merging
    full_train_ds = datasets.concatenate_datasets(
        [heysquad_train, instructs2s_train, *ds_list]
    )
    full_val_ds = datasets.concatenate_datasets([heysquad_val, *ds_list])

    return {"train": full_train_ds, "validation": full_val_ds}


def add_audio_column(row, full_ds, full_ds_lookup):
    key = (row["unique_id_1"], row["unique_id_2"], row["dataset_name"])
    if key not in full_ds_lookup:
        raise KeyError(f"Missing key in full dataset: {key}")
    audio = full_ds[full_ds_lookup[key]]["audio"]
    return {"audio": audio}


def merge_datasets(alarm_ds_dict, full_ds_dict):
    merged_ds_dict = {}
    for split in alarm_ds_dict.keys():
        merged_ds = alarm_ds_dict[split]
        full_ds = full_ds_dict[split]

        # find index in full_ds for each unique_pair
        full_ds_lookup = {}
        for i in range(len(full_ds)):
            row = full_ds[i]
            key = (row["unique_id_1"], row["unique_id_2"], row["dataset_name"])
            if key in full_ds_lookup:
                raise ValueError(f"Duplicate key in full_ds[{split}]: {key}")
            full_ds_lookup[key] = i

        fn_kwargs = {"full_ds": full_ds, "full_ds_lookup": full_ds_lookup}

        # add missing audio column
        # all other columns are already in merged_ds
        merged_ds = merged_ds.map(
            add_audio_column,
            fn_kwargs=fn_kwargs,
            desc=f"Adding audio column to split {split}",
        )
        merged_ds_dict[split] = merged_ds

    return datasets.DatasetDict(merged_ds_dict)


def download_and_merge_datasets(args):
    full_ds_dict = load_full_dataset_from_disk(args)
    alarm_corpora = datasets.load_dataset("Blinorot/ALARM-Corpora")

    alarm_ds_dict = {}
    if args.merge_filtered:
        alarm_ds_dict["train"] = alarm_corpora["train"]
        alarm_ds_dict["validation"] = alarm_corpora["validation"]
    else:
        alarm_ds_dict["train"] = alarm_corpora["raw_train"]
        alarm_ds_dict["validation"] = alarm_corpora["raw_validation"]

    merged_ds = merge_datasets(alarm_ds_dict, full_ds_dict)
    merged_ds.save_to_disk(ROOT_PATH / args.data_dir / "datasets" / args.save_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Merge local dataset with HuggingFace")
    parser.add_argument(
        "--data-dir",
        default="data",
        type=str,
        help="data dir name (Default: data)",
    )
    parser.add_argument(
        "--save-name",
        default="alarm_merged",
        type=str,
        help="name for the merged dataset (Default: alarm_merged)",
    )
    parser.add_argument(
        "--merge-filtered",
        dest="merge_filtered",
        action="store_true",
        help="merge already filtered splits",
    )
    parser.set_defaults(merge_filtered=False)

    args = parser.parse_args()
    download_and_merge_datasets(args)
