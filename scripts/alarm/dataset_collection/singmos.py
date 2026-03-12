import json
import os
import shutil
from collections import defaultdict
from pathlib import Path

import datasets
import pandas as pd
import torchaudio
from huggingface_hub import snapshot_download
from tqdm.auto import tqdm

from utils import DATA_PATH, SAMPLING_RATE, cli_download, get_audio_description

# https://arxiv.org/abs/2406.10911
DATASET_NAME = "singmos"


def download_dataset():
    dataset_dict = defaultdict(list)

    data_dir = DATA_PATH / DATASET_NAME
    data_dir.mkdir(exist_ok=True, parents=True)

    downloaded_data = snapshot_download(
        repo_id="TangRain/SingMOS",
        repo_type="dataset",
        revision="a0fac9f752747a44536e1df3be9ecd260a947d5e",
    )
    downloaded_data = Path(downloaded_data)

    score_data_path = downloaded_data / "info" / "score.json"
    with score_data_path.open("r") as f:
        score_data = json.load(f)
        score_data = score_data["utterance"]

    split_path = downloaded_data / "info" / "split.json"
    with split_path.open("r") as f:
        split_data = json.load(f)
        split_data = split_data["singmos"]["train"]

    sys_info_path = downloaded_data / "info" / "sys_info.json"
    with sys_info_path.open("r") as f:
        sys_info = json.load(f)

    for f_id in tqdm(split_data):
        system = score_data[f_id]["sys_id"]
        score = score_data[f_id]["score"]["mos"]
        score = float(score)
        score = round(score, 2)

        system_type = sys_info[system]["type"]
        if system_type == "svs":
            system_type = "Singning Voice Synthesis"
        elif system_type == "svc":
            system_type = "Singing Voice Conversion"
        elif system_type == "vocoder":
            system_type = "Vocoder"
        elif system_type == "gt":
            system_type = "Not generated"

        if system_type == "Not generated":
            audio_type = "Real"
        else:
            audio_type = "Synthesized"

        text = "Singing"

        audio_path = str(downloaded_data / "wavs" / f"{f_id}.wav")

        audio_frames, audio_sr = torchaudio.load(audio_path)
        duration = audio_frames.shape[-1] / audio_sr

        elem_info = {}

        elem_info["duration"] = duration
        elem_info["text"] = text.strip()
        elem_info["mos"] = score
        elem_info["real or synthesized"] = audio_type
        elem_info["type of generative algorithm"] = system_type

        audio_description = get_audio_description(text_type="caption", **elem_info)

        dataset_dict["audio_description"].append(audio_description)
        for k, v in elem_info.items():
            dataset_dict[k].append(v)
        dataset_dict["audio"].append(audio_path)
        dataset_dict["f_id"].append(f_id)
        dataset_dict["unique_id_1"].append(str(f_id))
        dataset_dict["system_id"].append(system)
        dataset_dict["unique_id_2"].append(str(system))

    ds = datasets.Dataset.from_dict(dataset_dict)
    ds = ds.sort(["unique_id_1", "unique_id_2"])
    ds = ds.cast_column("audio", datasets.Audio(sampling_rate=SAMPLING_RATE))
    ds = ds.add_column("dataset_index", list(range(len(ds))))
    ds = datasets.DatasetDict({"train": ds})
    ds.save_to_disk(DATA_PATH / DATASET_NAME)


if __name__ == "__main__":
    download_dataset()
