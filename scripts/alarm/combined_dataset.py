import pprint
import random
from argparse import ArgumentParser
from collections import defaultdict
from pathlib import Path

import datasets
import pyarrow.compute as pc
from tqdm.auto import tqdm

ROOT_PATH = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT_PATH / "data" / "datasets" / "raw"
DATASET_NAME = "combined"

SPEECH_DATASETS = [
    "cameo",
    "globe_v3",
    "vctk",
    "vocalsound",
    "ascend",
    "disfluencyspeech",
    "nisqa",
    "mucs",
    "mls",
    "librispeech",
    "asvspoof19",
    "asvspoof5",
    "noisy_vctk",
]
COMPLICATED_SPEECH_DATASETS = [
    "voxceleb1",
    "alimeeting",
    "amicorpus",
    "meld",
    "dailytalk",
    "librimix",
]
EVENT_DATASETS = [
    "audiocaps",
    "clotho",
]
MUSIC_DATASETS = [
    "gtzan",
    "opensinger",
    "singmos",
    "mridangam",
    "nsynth",
    "fma",
    "sonicmaster",
]
ENVIRONMENT_DATASETS = [
    "esc50",
    "fsd50k",
    "audioset",
]

COLUMNS_TO_TAKE = [
    "audio",
    "audio_description",
    "unique_id_1",
    "unique_id_2",
    "dataset_index",
]
ALL_DATASETS = [
    *SPEECH_DATASETS,
    *COMPLICATED_SPEECH_DATASETS,
    *EVENT_DATASETS,
    *MUSIC_DATASETS,
    *ENVIRONMENT_DATASETS,
]


def combine_group_datasets(dataset_group, n_examples):
    full_ds = []
    all_hours = 0
    group_examples = defaultdict(list)
    random.seed(1)
    for dataset_name in dataset_group:
        ds = datasets.load_from_disk(DATA_PATH / dataset_name)["train"]
        duration_col = ds.data.column("duration")
        audio_hours = pc.sum(duration_col).as_py() / 3600
        all_hours += audio_hours
        print(dataset_name, ds.shape, audio_hours)

        ds = ds.select_columns(COLUMNS_TO_TAKE)
        ds = ds.add_column("dataset_name", [dataset_name] * len(ds))
        # ds = ds.cast_column("unique_id_1", datasets.Value("string"))
        # ds = ds.cast_column("unique_id_2", datasets.Value("string"))

        full_ds.append(ds)

        if n_examples > 0:
            indexes = random.sample(range(len(ds)), n_examples)
            for elem in ds.select(indexes):
                example = (elem["audio_description"], elem["audio"])
                group_examples[dataset_name].append(example)

    full_ds = datasets.concatenate_datasets(full_ds)
    print(f"Group Total: {full_ds.shape}, Hours: {all_hours}")
    return full_ds, all_hours, group_examples


def combine_datasets(n_examples):
    dataset_groups = {
        "speech": SPEECH_DATASETS,
        "environment": ENVIRONMENT_DATASETS,
        "music": MUSIC_DATASETS,
        "event": EVENT_DATASETS,
        "complicated_speech": COMPLICATED_SPEECH_DATASETS,
    }
    all_ds = []
    total_hours = 0
    dataset_name_examples = {}
    for name, dataset_group in dataset_groups.items():
        print(f"Processing {name} dataset group...")
        full_ds, all_hours, group_examples = combine_group_datasets(
            dataset_group, n_examples
        )
        all_ds.append(full_ds)
        total_hours += all_hours
        dataset_name_examples.update(**group_examples)

    final_ds = datasets.DatasetDict({"train": datasets.concatenate_datasets(all_ds)})
    print(f"Final Total: {final_ds.shape}, Hours: {total_hours}")
    # final_ds.save_to_disk(DATA_PATH / DATASET_NAME)

    if n_examples > 0:
        for k, v in dataset_name_examples.items():
            desc = "\n\n".join([elem[0] for elem in v])
            print(f"====={k}=====\n===\n{desc}\n==========")

        return final_ds, dataset_name_examples

    return final_ds


if __name__ == "__main__":
    parser = ArgumentParser("Combine all datasets into a big one")

    parser.add_argument(
        "--n-examples",
        type=int,
        default=0,
        help="How many examples per dataset to return",
    )

    args = parser.parse_args()

    combine_datasets(args.n_examples)
