from collections import defaultdict

import datasets
from tqdm.auto import tqdm

from utils import DATA_PATH, SAMPLING_RATE, get_audio_description

# https://arxiv.org/abs/2112.06223
DATASET_NAME = "ascend"


def process_elem(elem):
    processed_elem = {}
    elem_info = {}
    elem_info["text"] = elem["transcription"].strip()

    audio = elem["audio"]

    duration = audio.get_all_samples().duration_seconds
    elem_info["duration"] = duration
    language = elem["language"]
    if language == "mixed":
        language = "zh-en code-switching"
    elem_info["language"] = language

    audio_description = get_audio_description(text_type="speech", **elem_info)

    processed_elem["audio_description"] = audio_description
    for k, v in elem_info.items():
        processed_elem[k] = v

    processed_elem["id"] = elem["id"]
    processed_elem["unique_id_1"] = str(elem["id"])

    processed_elem["path"] = elem["path"]
    processed_elem["unique_id_2"] = str(elem["path"])

    processed_elem["audio"] = audio
    return processed_elem


def download_dataset():
    ds = datasets.load_dataset(
        "CAiRE/ASCEND",
        split="train",
        revision="737e9800ae31be9932ba8464c80366559bd28424",
    )

    all_columns = set(ds.column_names)
    good_columns = {
        "text",
        "audio",
        "audio_description",
        "duration",
        "id",
        "path",
        "unique_id_1",
        "unique_id_2",
    }
    remove_columns = all_columns - good_columns

    ds = ds.map(
        process_elem,
        desc="Processing ASCEND",
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
