from collections import defaultdict

import datasets
from tqdm.auto import tqdm

from utils import DATA_PATH, SAMPLING_RATE, get_audio_description

# https://arxiv.org/abs/2406.14875
DATASET_NAME = "globe_v3"


def process_elem(elem):
    processed_elem = {}
    elem_info = {}
    elem_info["text"] = elem["common_voice_sentence"].strip()
    elem_info["mos"] = elem["utmos"]
    elem_info["snr"] = elem["snr"]

    is_predicted = False

    if elem["common_voice_gender"] != "nan":
        elem_info["gender"] = elem["common_voice_gender"].title()
    else:
        elem_info["gender"] = elem["predicted_gender"].title()
        is_predicted = True

    if elem["common_voice_accents"] != "nan":
        elem_info["accent"] = elem["common_voice_accents"]
    else:
        elem_info["accent"] = elem["predicted_accent"]
        is_predicted = True

    if elem["common_voice_age"] != "nan":
        elem_info["age"] = elem["common_voice_age"]
    else:
        elem_info["age"] = elem["predicted_age"]
        is_predicted = True

    audio = elem["audio"]

    duration = audio.get_all_samples().duration_seconds
    elem_info["duration"] = duration

    audio_description = get_audio_description(text_type="speech", **elem_info)

    processed_elem["audio_description"] = audio_description
    for k, v in elem_info.items():
        processed_elem[k] = v

    sentence_path = elem["common_voice_path"]
    processed_elem["common_voice_path"] = sentence_path
    processed_elem["unique_id_1"] = str(sentence_path)

    sentence_id = elem["common_voice_sentence_id"]
    processed_elem["common_voice_sentence_id"] = sentence_id
    processed_elem["unique_id_2"] = str(sentence_id)

    processed_elem["audio"] = audio
    processed_elem["is_predicted"] = "Yes" if is_predicted else "No"
    return processed_elem


def download_dataset():
    ds = datasets.load_dataset(
        "MushanW/GLOBE_V3",
        split="train",
        revision="0eb6d96fca5ef8fc01d0bb4dc1721d0c193f1e72",
    )

    all_columns = set(ds.column_names)
    good_columns = {
        "text",
        "mos",
        "snr",
        "gender",
        "accent",
        "age",
        "audio",
        "audio_description",
        "duration",
        "common_voice_path",
        "common_voice_sentence_id",
        "unique_id_1",
        "unique_id_2",
    }
    remove_columns = all_columns - good_columns

    ds = ds.map(
        process_elem,
        desc="Processing Globe",
        remove_columns=remove_columns,
        num_proc=4,  # we will sort the data anyway
    )
    print("Shape with artificial data", ds.shape)
    ds = ds.filter(lambda x: x["is_predicted"] == "No")
    print("Shape after filering", ds.shape)
    ds = ds.remove_columns("is_predicted")

    # sort to ensure the same index across machines
    ds = ds.sort(["unique_id_1", "unique_id_2"])
    ds = ds.cast_column("audio", datasets.Audio(sampling_rate=SAMPLING_RATE))
    ds = ds.add_column("dataset_index", list(range(len(ds))))
    ds = datasets.DatasetDict({"train": ds})
    ds.save_to_disk(DATA_PATH / DATASET_NAME)


if __name__ == "__main__":
    download_dataset()
