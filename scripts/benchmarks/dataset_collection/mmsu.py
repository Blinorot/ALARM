import json
import os
import shutil
import subprocess
from collections import defaultdict

import datasets
import pandas as pd
import torchaudio
from huggingface_hub import snapshot_download
from tqdm.auto import tqdm

from utils import DATA_PATH, SAMPLING_RATE, cli_download, get_audio_description

# https://openreview.net/forum?id=yHzCDP1tXw
DATASET_NAME = "mmsu"
URL_LINKS = {
    "metadata": "https://huggingface.co/datasets/ddwang2000/MMSU/resolve/2568d7b939286b19a0be93b767886e29c7874445/question/mmsu.jsonl",
}


def download_dataset():
    dataset_dict = defaultdict(list)

    data_dir = DATA_PATH / DATASET_NAME
    data_dir.mkdir(exist_ok=True, parents=True)

    tmp_dir = data_dir / "tmp"
    tmp_dir.mkdir(exist_ok=True, parents=True)

    snapshot_download(
        repo_id="ddwang2000/MMSU",
        repo_type="dataset",
        allow_patterns="audio/**",
        local_dir=str(tmp_dir),
        local_dir_use_symlinks=False,
        revision="4b230b59ac5ae4cd074b4efddfc459496000111b",
    )

    metadata_dir = data_dir / "metadata"
    metadata_dir.mkdir(exist_ok=True, parents=True)
    metadata_lines_path = metadata_dir / "test_metadata.jsonl"
    metadata_path = metadata_dir / "test_metadata.json"
    cli_download(URL_LINKS["metadata"], output=str(metadata_lines_path))

    with metadata_lines_path.open("r") as f:
        metadata = []
        for line in f:
            meta_line = json.loads(line)

            # fix choices format
            choices = []
            for letter in ["a", "b", "c", "d"]:
                key = f"choice_{letter}"
                if key in meta_line.keys():
                    choices.append(meta_line[key])
            meta_line["choices"] = choices

            metadata.append(meta_line)

    with metadata_path.open("w") as f:
        json.dump(metadata, f)

    dataset_index = 0

    for elem in tqdm(metadata):
        audio_fname = elem["audio_path"][1:]
        audio_path = str(tmp_dir / audio_fname)

        assert os.path.exists(audio_path), f"{audio_path} not found"

        f_id = str(elem["id"])
        question = elem["question"]

        choices = elem["choices"]
        # for consistency
        choices = [str(choice) for choice in choices]
        answer = str(elem["answer_gt"])

        dataset_dict["question"].append(question)
        dataset_dict["choices"].append(choices)
        dataset_dict["answer"].append(answer)
        dataset_dict["audio"].append(audio_path)
        dataset_dict["id"].append(f_id)
        dataset_dict["dataset_index"].append(dataset_index)

        dataset_index += 1

    ds = datasets.Dataset.from_dict(dataset_dict)
    ds = ds.sort("dataset_index")  # to be sure
    ds = ds.cast_column("audio", datasets.Audio(sampling_rate=SAMPLING_RATE))
    ds = datasets.DatasetDict({"test": ds})
    ds.save_to_disk(DATA_PATH / DATASET_NAME)

    # delete raw data
    # keep metadata file
    shutil.rmtree(tmp_dir)


if __name__ == "__main__":
    download_dataset()
