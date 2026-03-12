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

# https://arxiv.org/abs/2410.19168
DATASET_NAME = "mmau"
URL_LINKS = {
    "data_test-mini": "https://drive.google.com/file/d/1fERNIyTa0HWry6iIG1X-1ACPlUlhlRWA/view?usp=sharing",
    "metadata_test-mini": "https://raw.githubusercontent.com/Sakshi113/MMAU/8e0ed4cd58363e546dda52cc21f9a733e684bae3/mmau-test-mini.json",
    "data_test": "https://huggingface.co/datasets/gamma-lab-umd/MMAU-test/resolve/8e835a9f64ed6c703b3c9ddb6d423d9ab697061e/test-audios.tar.gz",
    "metadata_test": "https://raw.githubusercontent.com/Sakshi113/MMAU/8e0ed4cd58363e546dda52cc21f9a733e684bae3/mmau-test.json",
}


def download_dataset():
    data_dir = DATA_PATH / DATASET_NAME
    data_dir.mkdir(exist_ok=True, parents=True)
    metadata_dir = data_dir / "metadata"
    metadata_dir.mkdir(exist_ok=True, parents=True)

    dataset_splits = {}
    for set, method in zip(["test-mini", "test"], ["gdown_file", "wget"]):
        arc_path = data_dir / f"{set}-audios.tar.gz"
        cli_download(URL_LINKS[f"data_{set}"], output=str(arc_path), method=method)
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

        metadata_path = metadata_dir / f"{set}_metadata.json"
        cli_download(URL_LINKS[f"metadata_{set}"], output=str(metadata_path))

        with metadata_path.open("r") as f:
            metadata = json.load(f)

        dataset_dict = defaultdict(list)
        dataset_index = 0

        max_duration = 0
        for elem in tqdm(metadata):
            audio_fname = elem["audio_id"][2:]
            audio_path = str(data_dir / audio_fname)

            audio_frames, audio_sr = torchaudio.load(audio_path)
            duration = audio_frames.shape[-1] / audio_sr
            max_duration = max(duration, max_duration)

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
        dataset_splits[set] = ds

        print("MAX DURATION", max_duration)

    ds = datasets.DatasetDict(dataset_splits)
    ds.save_to_disk(DATA_PATH / DATASET_NAME, max_shard_size="200MB")

    # delete raw data
    # keep metadata file
    shutil.rmtree(data_dir / "test-audios")
    shutil.rmtree(data_dir / "test-mini-audios")


if __name__ == "__main__":
    download_dataset()
