from collections import defaultdict

import datasets
from tqdm.auto import tqdm

from utils import DATA_PATH, SAMPLING_RATE, get_audio_description

# https://arxiv.org/abs/2505.11051
DATASET_NAME = "cameo"
SPLITS = [
    "cafe",
    "crema_d",
    "emns",
    "emozionalmente",
    "enterface",
    "jl_corpus",
    "mesd",
    "nemo",
    "oreau",
    "pavoque",
    "ravdess",
    "resd",
    "subesco",
]


def process_elem(elem):
    processed_elem = {}
    elem_info = {}
    elem_info["text"] = elem["transcription"].strip()
    elem_info["emotion"] = elem["emotion"].title()
    elem_info["gender"] = elem["gender"].title()
    elem_info["age"] = elem["age"]
    elem_info["language"] = elem["language"].title()

    audio = elem["audio"]

    duration = audio.get_all_samples().duration_seconds
    elem_info["duration"] = duration

    audio_description = get_audio_description(text_type="speech", **elem_info)

    processed_elem["audio_description"] = audio_description
    for k, v in elem_info.items():
        processed_elem[k] = v

    processed_elem["cameo_subset"] = elem["dataset"]
    processed_elem["unique_id_1"] = str(elem["dataset"])

    processed_elem["file_id"] = elem["file_id"]
    processed_elem["unique_id_2"] = str(elem["file_id"])

    processed_elem["audio"] = audio
    return processed_elem


def download_dataset():
    all_ds = []
    for split in SPLITS:
        ds = datasets.load_dataset(
            "amu-cai/CAMEO",
            split=split,
            revision="38e9e968deb4636377e76b28bbb2062f92b898ab",
        )

        all_columns = set(ds.column_names)
        good_columns = {
            "text",
            "emotion",
            "gender",
            "language",
            "age",
            "audio",
            "audio_description",
            "duration",
            "file_id",
            "cameo_subset",
            "unique_id_1",
            "unique_id_2",
        }
        remove_columns = all_columns - good_columns

        ds = ds.map(
            process_elem,
            desc=f"Processing {split}",
            remove_columns=remove_columns,
        )
        all_ds.append(ds)

    ds = datasets.concatenate_datasets(all_ds)
    # sort to ensure the same index across machines
    ds = ds.sort(["unique_id_1", "unique_id_2"])
    ds = ds.cast_column("audio", datasets.Audio(sampling_rate=SAMPLING_RATE))
    ds = ds.add_column("dataset_index", list(range(len(ds))))
    ds = datasets.DatasetDict({"train": ds})
    ds.save_to_disk(DATA_PATH / DATASET_NAME)


if __name__ == "__main__":
    download_dataset()
