import json
import os
import shutil
import subprocess
from collections import defaultdict
from pathlib import Path

import datasets
import pandas as pd
import torchaudio
from huggingface_hub import snapshot_download
from tqdm.auto import tqdm

from utils import DATA_PATH, SAMPLING_RATE, cli_download, get_audio_description

# https://aclanthology.org/2024.acl-long.109/
DATASET_NAME = "airbench"
URL_LINKS = {
    "metadata": "https://huggingface.co/datasets/qyang1021/AIR-Bench-Dataset/resolve/df92055c320edc10183f930d0df72740e9b33147/Foundation/Foundation_meta.json",
}


def download_dataset():
    dataset_dict = defaultdict(list)

    data_dir = DATA_PATH / DATASET_NAME
    data_dir.mkdir(exist_ok=True, parents=True)

    tmp_dir = data_dir / "tmp"
    audio_tmp_dir = tmp_dir / "tmp_audio"
    tmp_dir.mkdir(exist_ok=True, parents=True)
    audio_tmp_dir.mkdir(exist_ok=True, parents=True)

    # snapshot_download(
    #     repo_id="qyang1021/AIR-Bench-Dataset",
    #     repo_type="dataset",
    #     allow_patterns="Foundation/**",
    #     local_dir=str(tmp_dir),
    #     local_dir_use_symlinks=False,
    #     revision="7766a6f775ab20013e957353961007ab157a5d46",
    # )

    metadata_dir = data_dir / "metadata"
    metadata_dir.mkdir(exist_ok=True, parents=True)
    metadata_path = metadata_dir / "test_metadata.json"
    cli_download(URL_LINKS["metadata"], output=str(metadata_path))

    with metadata_path.open("r") as f:
        metadata = json.load(f)

    dataset_index = 0

    print(len(metadata))

    # correct_metadata = []

    data_max_duration = 0
    MAX_DURATION = 60  # we crop all audio to 60s

    unique_audio_path = set()
    unique_new_audio_path = set()

    for elem in tqdm(metadata):
        audio_fname = elem["path"]
        task = elem["task_name"]
        task_dataset_name = elem["dataset_name"]

        audio_path = str(
            tmp_dir / "Foundation" / f"{task}_{task_dataset_name}" / audio_fname
        )
        if task == "Audio_Grounding":
            # https://github.com/OFA-Sys/AIR-Bench/blob/983247666e2a25519b5c122810c2bcd91f26fd8d/Inference_Foundation.py#L31
            audio_path = audio_path[:-3] + "flac"

        audio, sr = torchaudio.load(audio_path)

        data_max_duration = max(data_max_duration, audio.shape[-1] / sr)
        if audio.shape[0] != 0:
            audio = audio.mean(dim=0)
            audio = audio.unsqueeze(0)
        audio = torchaudio.functional.resample(audio, sr, SAMPLING_RATE)
        audio = audio[
            ..., : SAMPLING_RATE * MAX_DURATION
        ]  # crop length to MAX_DURATION
        new_audio_path = (
            audio_tmp_dir / f"{dataset_index}_{task}_{task_dataset_name}_{audio_fname}"
        )
        new_audio_path = str(new_audio_path.with_suffix(".wav"))
        torchaudio.save(new_audio_path, audio, sample_rate=SAMPLING_RATE)

        unique_audio_path.add(audio_path)
        unique_new_audio_path.add(new_audio_path)

        f_id = str(elem["uniq_id"])
        question = str(elem["question"])

        # fix choices format
        choices = []
        for letter in ["a", "b", "c", "d"]:
            key = f"choice_{letter}"
            if key in elem.keys():
                choices.append(elem[key])
        # for consistency
        choices = [str(choice) for choice in choices]
        answer = str(elem["answer_gt"])

        dataset_dict["question"].append(question)
        dataset_dict["choices"].append(choices)
        dataset_dict["answer"].append(answer)
        dataset_dict["audio"].append(new_audio_path)
        dataset_dict["id"].append(f_id)
        dataset_dict["dataset_index"].append(dataset_index)

        dataset_index += 1

    #     correct_metadata.append(elem)

    # with metadata_path.open("w") as f:
    #     json.dump(correct_metadata, f, indent=4)

    print("Data max duration", data_max_duration)
    print(len(unique_audio_path), len(unique_new_audio_path))
    print(len(metadata), dataset_index)

    ds = datasets.Dataset.from_dict(dataset_dict)
    ds = ds.sort("dataset_index")  # to be sure
    ds = ds.cast_column("audio", datasets.Audio(sampling_rate=SAMPLING_RATE))
    ds = datasets.DatasetDict({"test": ds})
    ds.save_to_disk(DATA_PATH / DATASET_NAME, max_shard_size="200MB")

    # delete raw data
    # keep metadata file
    # shutil.rmtree(tmp_dir)


if __name__ == "__main__":
    download_dataset()
