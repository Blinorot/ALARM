from collections import defaultdict

import datasets
from tqdm.auto import tqdm

from utils import DATA_PATH, SAMPLING_RATE, get_audio_description

# https://arxiv.org/abs/1911.01601
DATASET_NAME = "asvspoof19"

ATTACK2TYPE = {
    "A01": "Text-To-Speech using WaveNet",
    "A02": "Text-To-Speech using WORLD",
    "A03": "Text-To-Speech using WORLD",
    "A04": "Text-To-Speech using Waveform Concatenation based on MaryTTS",
    "A05": "Voice Conversion using VAE and WORLD",
    "A06": "Transfer-function based Voice Conversion using Spectral Filtering and Overlap-and-Add",
    "-": "None because this is a real audio",
}


def process_elem(elem):
    processed_elem = {}
    elem_info = {}
    elem_info["text"] = "Speech"

    audio = elem["audio"]

    duration = audio.get_all_samples().duration_seconds
    elem_info["duration"] = duration
    system_id = elem["system_id"]
    elem_info["is bona fide or spoof"] = "bona fide" if system_id == "-" else "spoof"
    elem_info["spoof algorithm description"] = ATTACK2TYPE[system_id]

    audio_description = get_audio_description(text_type="caption", **elem_info)

    processed_elem["audio_description"] = audio_description
    for k, v in elem_info.items():
        processed_elem[k] = v

    processed_elem["speaker_id"] = elem["speaker_id"]
    processed_elem["unique_id_1"] = str(elem["speaker_id"])

    processed_elem["audio_file_name"] = elem["audio_file_name"]
    processed_elem["unique_id_2"] = str(elem["audio_file_name"])

    processed_elem["audio"] = audio
    return processed_elem


def download_dataset():
    ds = datasets.load_dataset(
        "Bisher/ASVspoof_2019_LA",
        split="train",
        revision="aea92dd83a9c56e070c0b1e9f02e7c0d96216a4c",
    )

    all_columns = set(ds.column_names)
    good_columns = {
        "text",
        "audio",
        "audio_description",
        "speaker_id",
        "audio_file_name",
        "system_id",
        "" "unique_id_1",
        "unique_id_2",
    }
    remove_columns = all_columns - good_columns

    ds = ds.map(
        process_elem,
        desc="Processing ASVspoof19",
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
