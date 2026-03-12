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

# https://ieeexplore.ieee.org/abstract/document/9645159
DATASET_NAME = "fsd50k"
URL_LINKS = {
    "part1": "https://zenodo.org/records/4060432/files/FSD50K.dev_audio.z01?download=1",
    "part2": "https://zenodo.org/records/4060432/files/FSD50K.dev_audio.z02?download=1",
    "part3": "https://zenodo.org/records/4060432/files/FSD50K.dev_audio.z03?download=1",
    "part4": "https://zenodo.org/records/4060432/files/FSD50K.dev_audio.z04?download=1",
    "part5": "https://zenodo.org/records/4060432/files/FSD50K.dev_audio.z05?download=1",
    "part6": "https://zenodo.org/records/4060432/files/FSD50K.dev_audio.zip?download=1",
    "ground_truth": "https://zenodo.org/records/4060432/files/FSD50K.ground_truth.zip?download=1",
    "metadata": "https://zenodo.org/records/4060432/files/FSD50K.metadata.zip?download=1",
}


def download_dataset():
    dataset_dict = defaultdict(list)

    data_dir = DATA_PATH / DATASET_NAME
    data_dir.mkdir(exist_ok=True, parents=True)

    arc_names = []
    for part_id in range(1, 7):
        if part_id == 6:
            end = "zip"
        else:
            end = f"z0{part_id}"
        arc_path = data_dir / f"FSD50K.dev_audio.{end}"
        arc_names.append(str(arc_path))
        cli_download(URL_LINKS[f"part{part_id}"], output=str(arc_path), method="wget")
    final_arc_path = data_dir / "FSD50K_full.zip"
    subprocess.run(
        ["zip", "-s", "0", arc_names[-1], "--out", str(final_arc_path)], check=True
    )
    shutil.unpack_archive(final_arc_path, data_dir)
    for arc_name in arc_names:
        os.remove(arc_name)
    os.remove(final_arc_path)

    metadata_path = data_dir / "metadata.zip"
    ground_truth_path = data_dir / "captions.zip"
    cli_download(URL_LINKS["ground_truth"], output=str(ground_truth_path))
    cli_download(URL_LINKS["metadata"], output=str(metadata_path))
    shutil.unpack_archive(metadata_path, data_dir)
    shutil.unpack_archive(ground_truth_path, data_dir)
    os.remove(metadata_path)
    os.remove(ground_truth_path)

    wav_dir = data_dir / "FSD50K.dev_audio"
    labels_path = data_dir / "FSD50K.ground_truth" / "dev.csv"
    metadata_path = data_dir / "FSD50K.metadata" / "collection" / "collection_dev.csv"
    labels_data = pd.read_csv(labels_path)
    metadata = pd.read_csv(metadata_path)

    labels_data = labels_data.loc[labels_data["split"] == "train"]

    for _, row in tqdm(labels_data.iterrows(), total=labels_data.shape[0]):
        fname = row["fname"]
        labels = row["labels"].split(",")

        fdata = metadata.loc[metadata["fname"] == fname].iloc[0]
        if isinstance(fdata, str):
            collection_labels = fdata["labels"].split(",")
        else:
            collection_labels = []

        full_labels = []
        for elem in collection_labels:
            if elem not in labels:
                full_labels.append(elem)
        full_labels.extend(labels)

        main_label = full_labels[0].replace("_", " ")
        labels = "/".join(full_labels).replace("_", " ")

        audio_path = str(wav_dir / f"{fname}.wav")

        audio_frames, audio_sr = torchaudio.load(audio_path)
        duration = audio_frames.shape[-1] / audio_sr

        elem_info = {}

        elem_info["duration"] = duration
        elem_info["text"] = main_label
        elem_info["sound categories"] = labels

        audio_description = get_audio_description(text_type="caption", **elem_info)

        dataset_dict["audio_description"].append(audio_description)
        for k, v in elem_info.items():
            dataset_dict[k].append(v)
        dataset_dict["audio"].append(audio_path)
        dataset_dict["fname"].append(fname)
        dataset_dict["unique_id_1"].append(str(fname))
        # extra word so it does not look like the corresponding metadata field
        # needed for prompt creation
        dataset_dict["unique_id_2"].append(f"labels: {labels}")

    ds = datasets.Dataset.from_dict(dataset_dict)
    ds = ds.sort(["unique_id_1", "unique_id_2"])
    ds = ds.cast_column("audio", datasets.Audio(sampling_rate=SAMPLING_RATE))
    ds = ds.add_column("dataset_index", list(range(len(ds))))
    ds = datasets.DatasetDict({"train": ds})
    ds.save_to_disk(DATA_PATH / DATASET_NAME)

    # delete raw data
    shutil.rmtree(data_dir / "FSD50K.dev_audio")
    shutil.rmtree(data_dir / "FSD50K.ground_truth")
    shutil.rmtree(data_dir / "FSD50K.metadata")


if __name__ == "__main__":
    download_dataset()
