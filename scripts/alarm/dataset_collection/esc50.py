import json
import os
import shutil
from collections import defaultdict

import datasets
import pandas as pd
import torchaudio
from tqdm.auto import tqdm

from utils import DATA_PATH, SAMPLING_RATE, cli_download, get_audio_description

# https://dl.acm.org/doi/10.1145/2733373.2806390
DATASET_NAME = "esc50"
URL_LINK = "https://github.com/karoldvl/ESC-50/archive/master.zip"
META_URL_LINK = "https://raw.githubusercontent.com/karolpiczak/ESC-50/refs/heads/master/meta/esc50.csv"


def download_dataset():
    dataset_dict = defaultdict(list)

    data_dir = DATA_PATH / DATASET_NAME
    data_dir.mkdir(exist_ok=True, parents=True)

    arc_path = data_dir / "esc50.zip"
    cli_download(URL_LINK, output=str(arc_path))
    shutil.unpack_archive(arc_path, data_dir)
    os.remove(arc_path)

    metadata_path = data_dir / "meta.csv"
    cli_download(META_URL_LINK, output=str(metadata_path))

    metadata = pd.read_csv(metadata_path)
    for _, row in metadata.iterrows():
        f_id = row["filename"]
        src = row["src_file"]
        text = row["category"]

        audio_path = str(data_dir / "ESC-50-master" / "audio" / f_id)

        audio_frames, audio_sr = torchaudio.load(audio_path)
        duration = audio_frames.shape[-1] / audio_sr

        elem_info = {}

        elem_info["duration"] = duration
        elem_info["text"] = text.strip()

        audio_description = get_audio_description(text_type="caption", **elem_info)

        dataset_dict["audio_description"].append(audio_description)
        for k, v in elem_info.items():
            dataset_dict[k].append(v)
        dataset_dict["audio"].append(audio_path)
        dataset_dict["filename"].append(f_id)
        dataset_dict["unique_id_1"].append(str(f_id))
        dataset_dict["src_file"].append(src)
        dataset_dict["unique_id_2"].append(str(src))

    ds = datasets.Dataset.from_dict(dataset_dict)
    ds = ds.sort(["unique_id_1", "unique_id_2"])
    ds = ds.cast_column("audio", datasets.Audio(sampling_rate=SAMPLING_RATE))
    ds = ds.add_column("dataset_index", list(range(len(ds))))
    ds = datasets.DatasetDict({"train": ds})
    ds.save_to_disk(DATA_PATH / DATASET_NAME)

    # delete raw data
    shutil.rmtree(data_dir / "ESC-50-master")
    os.remove(str(data_dir / "meta.csv"))


if __name__ == "__main__":
    download_dataset()
