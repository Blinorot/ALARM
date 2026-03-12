from collections import defaultdict

import datasets
from tqdm.auto import tqdm

from utils import DATA_PATH, SAMPLING_RATE, get_audio_description

# https://arxiv.org/abs/2304.13689
DATASET_NAME = "heysquad_human"


def download_dataset():
    ds = datasets.load_dataset(
        "yijingwu/HeySQuAD_human",
        revision="2a9aff6032a76185b98eff703144a2a896313e49",
    )

    ds_dict = {}

    for split in ds:
        ds_split = ds[split]
        ds_split = ds_split.select_columns(
            ["audio", "question", "context", "id", "is_impossible"]
        )
        ds_split = ds_split.map(
            lambda x: {
                "unique_id_1": str(x["id"]),
                "unique_id_2": str(x["is_impossible"]),
            },
        )
        # sort to ensure order is the same
        ds_split = ds_split.sort(["unique_id_1", "unique_id_2"])
        ds_split = ds_split.cast_column(
            "audio", datasets.Audio(sampling_rate=SAMPLING_RATE)
        )
        ds_split = ds_split.add_column("dataset_index", list(range(len(ds_split))))
        ds_dict[split] = ds_split

    ds = datasets.DatasetDict(ds_dict)
    ds.save_to_disk(DATA_PATH / DATASET_NAME)


if __name__ == "__main__":
    download_dataset()
