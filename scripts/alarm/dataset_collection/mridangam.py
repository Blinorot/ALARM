import json
import os
import shutil
from collections import defaultdict

import datasets
import pandas as pd
import torchaudio
from tqdm.auto import tqdm

from utils import DATA_PATH, SAMPLING_RATE, cli_download, get_audio_description

# https://ieeexplore.ieee.org/abstract/document/6637633
DATASET_NAME = "mridangam"
URL_LINK = (
    "https://zenodo.org/records/4068196/files/mridangam_stroke_1.5.zip?download=1"
)


def download_dataset():
    dataset_dict = defaultdict(list)

    data_dir = DATA_PATH / DATASET_NAME
    data_dir.mkdir(exist_ok=True, parents=True)

    arc_path = data_dir / "mridangam.zip"
    cli_download(URL_LINK, output=str(arc_path))
    shutil.unpack_archive(arc_path, data_dir)
    os.remove(arc_path)

    full_data_dir = data_dir / "mridangam_stroke_1.5"

    for tonic in os.listdir(full_data_dir):
        tonic_dir = full_data_dir / tonic
        if not tonic_dir.is_dir():
            continue
        for song in tqdm(os.listdir(tonic_dir), desc=f"Processing {tonic}..."):
            if not song.endswith(".wav"):
                continue

            audio_path = str(tonic_dir / song)

            metadata = audio_path.split("_")[-1]
            stroke = metadata.split("-")[0]

            audio_frames, audio_sr = torchaudio.load(audio_path)
            duration = audio_frames.shape[-1] / audio_sr

            elem_info = {}

            elem_info["duration"] = duration
            elem_info["text"] = "Mridangam"
            elem_info["tonic"] = tonic
            elem_info["stroke"] = stroke.title()

            audio_description = get_audio_description(text_type="caption", **elem_info)

            dataset_dict["audio_description"].append(audio_description)
            for k, v in elem_info.items():
                dataset_dict[k].append(v)
            dataset_dict["audio"].append(audio_path)
            # extra word so it does not look like the corresponding metadata field
            # needed for prompt creation
            dataset_dict["unique_id_1"].append(f"tonic: {tonic}")
            dataset_dict["song_id"].append(song)
            dataset_dict["unique_id_2"].append(str(song))

    ds = datasets.Dataset.from_dict(dataset_dict)
    ds = ds.sort(["unique_id_1", "unique_id_2"])
    ds = ds.cast_column("audio", datasets.Audio(sampling_rate=SAMPLING_RATE))
    ds = ds.add_column("dataset_index", list(range(len(ds))))
    ds = datasets.DatasetDict({"train": ds})
    ds.save_to_disk(DATA_PATH / DATASET_NAME)

    # delete raw data
    shutil.rmtree(data_dir / "mridangam_stroke_1.5")
    shutil.rmtree(data_dir / "__MACOSX")


if __name__ == "__main__":
    download_dataset()
