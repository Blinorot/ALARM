from collections import defaultdict

import datasets
from tqdm.auto import tqdm

from utils import DATA_PATH, SAMPLING_RATE, get_audio_description

# https://ieeexplore.ieee.org/document/7952261
DATASET_NAME = "audioset"


def process_elem(elem):
    processed_elem = {}
    elem_info = {}
    elem_info["text"] = ", ".join(elem["json"]["label"])

    audio = elem["wav"]

    duration = audio.get_all_samples().duration_seconds
    elem_info["duration"] = duration

    audio_description = get_audio_description(text_type="caption", **elem_info)

    processed_elem["audio_description"] = audio_description
    for k, v in elem_info.items():
        processed_elem[k] = v

    processed_elem["video_id"] = elem["json"]["id"]
    processed_elem["unique_id_2"] = str(elem["json"]["id"])
    processed_elem["key"] = elem["__key__"]
    processed_elem["unique_id_1"] = str(elem["__key__"])

    processed_elem["audio"] = audio
    return processed_elem


def download_dataset():
    ds = datasets.load_dataset(
        "confit/audioset-16khz-wds",
        "2m",
        split="train",
        revision="4a40121a0063dd58d5d15f40f03c496eb3013906",
    )

    all_columns = set(ds.column_names)
    good_columns = {
        "text",
        "audio",
        "audio_description",
        "duration",
        "video_id",
        "key",
        "unique_id_1",
        "unique_id_2",
    }
    remove_columns = all_columns - good_columns

    ds = ds.map(
        process_elem,
        desc="Processing AudioSet",
        remove_columns=remove_columns,
    )

    # sort to ensure the same index across machines
    ds = ds.sort(["unique_id_1", "unique_id_2"])
    ds = ds.cast_column("audio", datasets.Audio(sampling_rate=SAMPLING_RATE))
    ds = ds.add_column("dataset_index", list(range(len(ds))))
    ds = datasets.DatasetDict({"train": ds})
    ds.save_to_disk(DATA_PATH / DATASET_NAME)


if __name__ == "__main__":
    download_dataset()
