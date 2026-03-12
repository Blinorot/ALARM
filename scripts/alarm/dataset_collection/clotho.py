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

# https://arxiv.org/abs/1910.09387
DATASET_NAME = "clotho"
URL_LINKS = {
    "data": "https://zenodo.org/records/3490684/files/clotho_audio_development.7z?download=1",
    "captions": "https://zenodo.org/records/3490684/files/clotho_captions_development.csv?download=1",
    "metadata": "https://zenodo.org/records/3490684/files/clotho_metadata_development.csv?download=1",
}


def download_dataset():
    dataset_dict = defaultdict(list)

    data_dir = DATA_PATH / DATASET_NAME
    data_dir.mkdir(exist_ok=True, parents=True)

    arc_path = data_dir / "data.7z"
    cli_download(URL_LINKS["data"], output=str(arc_path), method="wget")
    subprocess.run(["7z", "x", str(arc_path), f"-o{data_dir}", "-y"], check=True)
    os.remove(arc_path)

    metadata_path = data_dir / "metadata.csv"
    captions_path = data_dir / "captions.csv"
    cli_download(URL_LINKS["captions"], output=str(captions_path))
    cli_download(URL_LINKS["metadata"], output=str(metadata_path))

    captions = pd.read_csv(captions_path)
    metadata = pd.read_csv(metadata_path)

    for _, row in tqdm(captions.iterrows(), total=captions.shape[0]):
        fname = row["file_name"]
        tags = metadata.loc[metadata["file_name"] == fname].iloc[0]["keywords"]
        tags = tags.replace(";", "/")

        audio_path = str(data_dir / "development" / fname)

        audio_frames, audio_sr = torchaudio.load(audio_path)
        duration = audio_frames.shape[-1] / audio_sr

        for index in range(1, 6):
            caption_id = f"caption_{index}"
            text = row[caption_id].strip()

            elem_info = {}

            elem_info["duration"] = duration
            elem_info["text"] = text
            elem_info["tags"] = tags

            audio_description = get_audio_description(text_type="caption", **elem_info)

            dataset_dict["audio_description"].append(audio_description)
            for k, v in elem_info.items():
                dataset_dict[k].append(v)
            dataset_dict["audio"].append(audio_path)
            dataset_dict["file_name"].append(fname)
            dataset_dict["unique_id_1"].append(str(fname))
            dataset_dict["caption_id"].append(caption_id)
            dataset_dict["unique_id_2"].append(str(caption_id))

    ds = datasets.Dataset.from_dict(dataset_dict)
    ds = ds.sort(["unique_id_1", "unique_id_2"])
    ds = ds.cast_column("audio", datasets.Audio(sampling_rate=SAMPLING_RATE))
    ds = ds.add_column("dataset_index", list(range(len(ds))))
    ds = datasets.DatasetDict({"train": ds})
    ds.save_to_disk(DATA_PATH / DATASET_NAME)

    # delete raw data
    shutil.rmtree(data_dir / "development")
    os.remove(metadata_path)
    os.remove(captions_path)


if __name__ == "__main__":
    download_dataset()
