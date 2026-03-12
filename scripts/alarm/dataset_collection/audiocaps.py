from collections import defaultdict

import datasets
from tqdm.auto import tqdm

from utils import DATA_PATH, SAMPLING_RATE, get_audio_description

# https://aclanthology.org/N19-1011/
DATASET_NAME = "audiocaps"


def process_elem(elem):
    processed_elem = {}
    elem_info = {}
    elem_info["text"] = elem["caption"].strip()

    audio = elem["audio"]

    duration = audio.get_all_samples().duration_seconds
    elem_info["duration"] = duration

    audio_description = get_audio_description(text_type="caption", **elem_info)

    processed_elem["audio_description"] = audio_description
    for k, v in elem_info.items():
        processed_elem[k] = v

    processed_elem["audiocap_id"] = elem["audiocap_id"]
    processed_elem["unique_id_1"] = str(elem["audiocap_id"])

    processed_elem["youtube_id"] = elem["youtube_id"]
    processed_elem["unique_id_2"] = str(elem["youtube_id"])

    processed_elem["audio"] = audio
    return processed_elem


def download_dataset():
    ds = datasets.load_dataset(
        "OpenSound/AudioCaps",
        split="train",
        revision="b29b3243d6ce49c2cd0d48d4b5f0701ae7969ded",
    )

    all_columns = set(ds.column_names)
    good_columns = {
        "text",
        "audio",
        "audio_description",
        "duration",
        "audiocap_id",
        "youtube_id",
        "unique_id_1",
        "unique_id_2",
    }
    remove_columns = all_columns - good_columns

    ds = ds.map(
        process_elem,
        desc="Processing AudioCaps",
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
