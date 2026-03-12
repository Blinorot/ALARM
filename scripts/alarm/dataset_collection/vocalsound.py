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

# https://ieeexplore.ieee.org/abstract/document/9746828
DATASET_NAME = "vocalsound"
URL_LINK = "https://www.dropbox.com/s/c5ace70qh1vbyzb/vs_release_16k.zip?dl=1"

# see paper Section 3
HEALTH_QUESTION = (
    "has a cold/allergy/other health-related symptoms that might affect speech"
)

# BROKEN FILES
# Broken file: m0600_0_laughter.wav
# Broken file: m0600_0_sigh.wav
# Broken file: m0600_0_sniff.wav
# Broken file: m0600_0_sneeze.wav
# Broken file: m0600_0_cough.wav
# Broken file: m0600_0_throatclearing.wav
# Broken file: m0072_0_laughter.wav
# Broken file: m0072_0_throatclearing.wav
# Broken file: f0098_0_sneeze.wav
# Broken file: f2298_1_sniff.wav
# Broken file: f2298_0_sniff.wav
# Broken file: f2298_1_throatclearing.wav
# Broken file: f2298_0_sneeze.wav
# Broken file: f2298_1_sneeze.wav
# Broken file: f2298_1_laughter.wav
# Broken file: f2298_1_sigh.wav
# Broken file: f2298_0_sigh.wav
# Broken file: f2298_0_laughter.wav
# Broken file: f2298_1_cough.wav
# Broken file: f2298_0_cough.wav
# Broken file: f2298_0_throatclearing.wav
# Broken file: f0066_0_cough.wav
# Broken file: f3310_0_throatclearing.wav
# Broken file: f3310_0_sniff.wav
# Broken file: f3310_0_laughter.wav
# Broken file: f3310_0_sigh.wav
# Broken file: f3310_0_cough.wav
# Broken file: f3310_0_sneeze.wav
# Broken file: m2077_0_sneeze.wav
# Broken file: m2077_0_laughter.wav
# Broken file: m2077_0_sigh.wav
# Broken file: m2077_0_throatclearing.wav
# Broken file: m2077_0_cough.wav
# Broken file: m2077_0_sniff.wav


def download_dataset():
    dataset_dict = defaultdict(list)

    data_dir = DATA_PATH / DATASET_NAME
    data_dir.mkdir(exist_ok=True, parents=True)

    arc_path = data_dir / "vocalsound.zip"
    cli_download(URL_LINK, output=str(arc_path))
    shutil.unpack_archive(arc_path, data_dir)
    os.remove(arc_path)

    metadata = pd.read_csv(data_dir / "meta" / "tr_meta.csv", header=None)

    speaker_files = defaultdict(list)
    for file in tqdm(os.listdir(data_dir / "audio_16k"), desc="Parsing dir"):
        speaker_id = file.split("_")[0]
        speaker_files[speaker_id].append(file)

    for _, row in tqdm(metadata.iterrows(), total=metadata.shape[0]):
        speaker_id = row[0]
        gender = row[1].title()
        age = row[2]
        # country = row[3]
        # native_language = row[4]
        health_condition = row[5].title()

        for file in speaker_files[speaker_id]:
            audio_path = data_dir / "audio_16k" / file

            fname = audio_path.stem
            audio_type = fname.split("_")[-1].title()
            if audio_type == "Throatclearing":
                audio_type = "Throat clearing"

            audio_path = str(audio_path)
            try:
                audio_frames, audio_sr = torchaudio.load(audio_path)
            except RuntimeError:
                print(f"Broken file: {file}")
                continue

            duration = audio_frames.shape[-1] / audio_sr
            audio_path = str(audio_path)

            elem_info = {}

            elem_info["duration"] = duration
            elem_info["text"] = audio_type
            # the exact age might be too hard to guess by hearing
            elem_info["age"] = f"{(age // 10) * 10}s"
            elem_info["gender"] = gender
            # we believe this data is impossible to guess by hearing here
            # elem_info["country"] = country
            # elem_info["native language"] = native_language
            elem_info[HEALTH_QUESTION] = health_condition

            audio_description = get_audio_description(text_type="caption", **elem_info)

            dataset_dict["audio_description"].append(audio_description)
            for k, v in elem_info.items():
                dataset_dict[k].append(v)
            dataset_dict["audio"].append(audio_path)
            dataset_dict["speaker_id"].append(speaker_id)
            dataset_dict["unique_id_1"].append(str(speaker_id))
            dataset_dict["file_id"].append(file)
            dataset_dict["unique_id_2"].append(str(file))

    ds = datasets.Dataset.from_dict(dataset_dict)
    ds = ds.sort(["unique_id_1", "unique_id_2"])
    ds = ds.cast_column("audio", datasets.Audio(sampling_rate=SAMPLING_RATE))
    ds = ds.add_column("dataset_index", list(range(len(ds))))
    ds = datasets.DatasetDict({"train": ds})
    ds.save_to_disk(DATA_PATH / DATASET_NAME)

    # delete raw data
    shutil.rmtree(data_dir / "__MACOSX")
    shutil.rmtree(data_dir / "audio_16k")
    shutil.rmtree(data_dir / "datafiles")
    shutil.rmtree(data_dir / "meta")
    os.remove(data_dir / "LICENSE")
    os.remove(data_dir / "readme.txt")
    os.remove(data_dir / "class_labels_indices_vs.csv")


if __name__ == "__main__":
    download_dataset()
