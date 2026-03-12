import json
import os
import shutil
import subprocess
from collections import defaultdict

import datasets
import pandas as pd
import torchaudio
from tqdm.auto import tqdm

from utils import DATA_PATH, SAMPLING_RATE, cli_download, get_audio_description

# https://arxiv.org/abs/2505.13032
DATASET_NAME = "mmar"
URL_LINKS = {
    "data": "https://huggingface.co/datasets/BoJack/MMAR/resolve/3bd051123480e80d273ae9e8e9f1653f49010ac7/mmar-audio.tar.gz",
    "metadata": "https://huggingface.co/datasets/BoJack/MMAR/resolve/3bd051123480e80d273ae9e8e9f1653f49010ac7/MMAR-meta.json",
}


def download_dataset():
    dataset_dict = defaultdict(list)

    data_dir = DATA_PATH / DATASET_NAME
    data_dir.mkdir(exist_ok=True, parents=True)

    arc_path = data_dir / "data.tar.gz"
    cli_download(URL_LINKS["data"], output=str(arc_path), method="wget")
    cores = os.cpu_count()
    subprocess.run(
        [
            "tar",
            f"--use-compress-program=pigz -d -p {cores}",
            "-xf",
            str(arc_path),
            "-C",
            str(data_dir),
        ],
        check=True,
    )
    os.remove(arc_path)

    metadata_dir = data_dir / "metadata"
    metadata_dir.mkdir(exist_ok=True, parents=True)
    metadata_path = metadata_dir / "test_metadata.json"
    cli_download(URL_LINKS["metadata"], output=str(metadata_path))

    with metadata_path.open("r") as f:
        metadata = json.load(f)

    dataset_index = 0

    for elem in tqdm(metadata):
        audio_fname = elem["audio_path"][2:]
        audio_path = str(data_dir / audio_fname)

        f_id = elem["id"]
        question = elem["question"]
        choices = elem["choices"]
        # for consistency
        choices = [str(choice) for choice in choices]
        answer = elem["answer"]

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
    shutil.rmtree(data_dir / "audio")


if __name__ == "__main__":
    download_dataset()
