import hashlib
import json
import os
import random
import shutil
import subprocess
from collections import defaultdict

import datasets
import pandas as pd
import torch
import torchaudio
from private_keys import VOXCELEB1_LINKS, VOXCELEB2_LINKS
from tqdm.auto import tqdm

from utils import DATA_PATH, SAMPLING_RATE, cli_download, get_audio_description

# https://www.robots.ox.ac.uk/~vgg/data/voxceleb/
# get dataset download links from
# https://mm.kaist.ac.kr/datasets/voxceleb/index.html
DATASET1_NAME = "voxceleb1"
DATASET2_NAME = "voxceleb2"

VOXCELEB1_HASH = {
    "A": "e395d020928bc15670b570a21695ed96",
    "B": "bbfaaccefab65d82b21903e81a8a8020",
    "C": "017d579a2a96a077f40042ec33e51512",
    "D": "7bb1e9f70fddc7a678fa998ea8b3ba19",
}

VOXCELEB2_HASH = {
    "A": "da070494c573e5c0564b1d11c3b20577",
    "B": "17fe6dab2b32b48abaf1676429cdd06f",
    "C": "1de58e086c5edf63625af1cb6d831528",
    "D": "5a043eb03e15c5a918ee6a52aad477f9",
    "E": "cea401b624983e2d0b2a87fb5d59aa60",
    "F": "fc886d9ba90ab88e7880ee98effd6ae9",
    "G": "d160ecc3f6ee3eed54d55349531cb42e",
    "H": "6b84a81b9af72a9d9eecbb3b1f602e65",
}

URL1_META = "https://openslr.trmal.net/resources/49/vox1_meta.csv"
URL2_META = "https://openslr.trmal.net/resources/49/vox2_meta.csv"

# train pairs list
# replace with a link to our pairs for reproducibility
# if "", pairs will be generated automatically, but may differ from ours
TRAIN_LIST_URL = ""

N_AUDIO = 50  # number of pairs of a given type


# https://github.com/clovaai/voxceleb_trainer/blob/master/dataprep.py
## ========== ===========
## MD5SUM
## ========== ===========
def md5(fname):
    hash_md5 = hashlib.md5()
    with open(fname, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def get_all_speaker_audio(wav_dir, speaker_id):
    speaker_audio = []
    speaker_dir = wav_dir / speaker_id
    for video_dir in os.listdir(speaker_dir):
        for wav_file in os.listdir(speaker_dir / video_dir):
            if not wav_file.endswith(".wav"):
                continue
            audio_path = f"{speaker_id}/{video_dir}/{wav_file}"
            speaker_audio.append(audio_path)
    return speaker_audio


def sample_pairs(speaker1_audio, speaker2_audio, n_pairs=1):
    n_pairs = min(n_pairs, len(speaker1_audio), len(speaker2_audio))
    speaker1_sample = random.sample(speaker1_audio, n_pairs)
    speaker2_sample = random.sample(speaker2_audio, n_pairs)
    return list(zip(speaker1_sample, speaker2_sample))


def create_train_pairs(metadata, speaker2audio, path):
    train_pairs = []

    male_set = {*metadata[metadata["Gender"] == "m"].iloc[:, 0].values}
    female_set = {*metadata[metadata["Gender"] == "f"].iloc[:, 0].values}

    # fix seed
    random.seed(123)
    for _, row in tqdm(
        metadata.iterrows(), total=metadata.shape[0], desc="Getting train pairs"
    ):
        speaker_id = row.iloc[0]

        # N_AUDIO self-pairs
        speaker_audio = speaker2audio[speaker_id]
        self_pairs = sample_pairs(speaker_audio, speaker_audio, n_pairs=N_AUDIO)

        # N_AUDIO male speakers
        valid_male = male_set - {speaker_id}
        valid_male = sorted(valid_male)
        chosen_male = random.sample(valid_male, N_AUDIO)
        male_pairs = []
        for male_id in chosen_male:
            pair = sample_pairs(speaker_audio, speaker2audio[male_id], n_pairs=1)[0]
            male_pairs.append(pair)

        # N_AUDIO female speakers
        valid_female = female_set - {speaker_id}
        valid_female = sorted(valid_female)
        chosen_female = random.sample(valid_female, N_AUDIO)
        female_pairs = []
        for female_id in chosen_female:
            pair = sample_pairs(speaker_audio, speaker2audio[female_id], n_pairs=1)[0]
            female_pairs.append(pair)

        train_pairs.extend(self_pairs)
        train_pairs.extend(male_pairs)
        train_pairs.extend(female_pairs)

    print(f"Created {len(train_pairs)} pairs")

    with path.open("w") as f_out:
        for pair in train_pairs:
            line = f"{pair[0]}\t{pair[1]}\n"
            f_out.write(line)


def download_dataset(dataset_name):
    if dataset_name == DATASET1_NAME:
        dataset_links = VOXCELEB1_LINKS
        dataset_hash = VOXCELEB1_HASH
        meta_link = URL1_META
    else:
        dataset_links = VOXCELEB2_LINKS
        dataset_hash = VOXCELEB2_HASH
        meta_link = URL2_META

    dataset_dict = defaultdict(list)

    data_dir = DATA_PATH / dataset_name
    data_dir.mkdir(exist_ok=True, parents=True)

    for letter, link in dataset_links.items():
        arc_path = data_dir / f"voxceleb_part_{letter}"
        print(f"Downloading part {letter}")
        cli_download(link, output=str(arc_path), method="wget_no_check")

        hashsum = md5(str(arc_path))
        assert hashsum == dataset_hash[letter], f"Incorrect hash for {letter}"

    all_arc_path = str(data_dir / "voxceleb_part_*")
    arc_path = str(data_dir / "voxceleb.zip")
    subprocess.run(
        " ".join(["cat", all_arc_path, ">", arc_path]), shell=True, check=True
    )
    shutil.unpack_archive(arc_path, data_dir)
    for letter in dataset_links.keys():
        os.remove(str(data_dir / f"voxceleb_part_{letter}"))
    os.remove(arc_path)

    meta_path = data_dir / "meta.csv"
    cli_download(meta_link, output=str(meta_path))
    metadata = pd.read_csv(meta_path, sep="\t")
    metadata = metadata.loc[metadata["Set"] == "dev"]

    wav_dir = data_dir / "wav"

    train_pairs = []
    speaker_data = {}
    speaker2audio = {}
    for _, row in tqdm(
        metadata.iterrows(),
        total=metadata.shape[0],
        desc="Getting speaker2audio and metadata",
    ):
        # get speaker metadata
        speaker_id = row.iloc[0]
        gender = "Male" if row.iloc[2] == "m" else "Female"
        nationality = row.iloc[3]
        speaker_data[speaker_id] = {"gender": gender, "nationality": nationality}

        speaker_audio = get_all_speaker_audio(wav_dir, speaker_id)
        speaker2audio[speaker_id] = speaker_audio

    train_list_path = data_dir / "train_list.txt"
    if TRAIN_LIST_URL == "":
        print("TRAIN LIST URL not provided. Generating train pairs...")
        create_train_pairs(metadata, speaker2audio, train_list_path)
    else:
        cli_download(TRAIN_LIST_URL, output=str(train_list_path))

    train_pairs = []
    with train_list_path.open("r") as f_in:
        for line in f_in.readlines():
            speaker1, speaker2 = line.strip().split("\t")
            train_pairs.append((speaker1, speaker2))

    print(f"N train pairs: {len(train_pairs)}")

    tmp_dir = data_dir / "tmp"
    tmp_dir.mkdir(exist_ok=True, parents=True)

    for speaker1_file, speaker2_file in tqdm(train_pairs):
        audio1_path = wav_dir / speaker1_file
        audio2_path = wav_dir / speaker2_file

        audio1, sr1 = torchaudio.load(audio1_path)
        audio2, sr2 = torchaudio.load(audio2_path)

        # convert to mono
        if audio1.shape[0] != 1:
            audio1 = audio1.mean(dim=0).unsqueeze(0)
        if audio2.shape[0] != 1:
            audio2 = audio2.mean(dim=0).unsqueeze(0)

        # resample
        if sr1 != SAMPLING_RATE:
            audio1 = torchaudio.functional.resample(audio1, sr1, SAMPLING_RATE)
        if sr2 != SAMPLING_RATE:
            audio2 = torchaudio.functional.resample(audio2, sr2, SAMPLING_RATE)

        duration1 = audio1.shape[-1] / SAMPLING_RATE
        duration2 = audio2.shape[-1] / SAMPLING_RATE

        start_time1 = 0
        end_time1 = duration1
        start_time2 = duration1
        end_time2 = duration1 + duration2

        speaker1_info = "the first speaker"
        speaker2_info = "the second speaker"

        joint_audio = torch.cat([audio1, audio2], dim=-1)
        joint_name = (
            speaker1_file.replace("/", "_")
            + "_"
            + speaker2_file.replace("/", "_")
            + ".wav"
        )
        joint_path = str(tmp_dir / joint_name)
        torchaudio.save(joint_path, joint_audio, sample_rate=SAMPLING_RATE)

        duration = [
            (start_time1, end_time1, speaker1_info),
            (start_time2, end_time2, speaker2_info),
        ]
        text = ["Speech", "Speech"]

        speaker1_id = speaker1_file.split("/")[0]
        speaker2_id = speaker2_file.split("/")[0]
        speaker1_data = speaker_data[speaker1_id]
        speaker2_data = speaker_data[speaker2_id]

        elem_info = {}

        elem_info["duration"] = duration
        elem_info["text"] = text
        elem_info["the speakers are the same"] = (
            "Yes" if speaker1_id == speaker2_id else "No"
        )
        elem_info["the first speaker's gender"] = speaker1_data["gender"]
        elem_info["the second speaker's gender"] = speaker2_data["gender"]
        elem_info["the first speaker's nationality"] = speaker1_data["nationality"]
        elem_info["the second speaker's nationality"] = speaker2_data["nationality"]
        elem_info["the number of speakers"] = 2

        audio_description = get_audio_description(text_type="caption", **elem_info)

        dataset_dict["audio_description"].append(audio_description)

        elem_info["duration"] = end_time2
        elem_info["text"] = ";".join(text)
        for k, v in elem_info.items():
            dataset_dict[k].append(v)
        dataset_dict["audio"].append(joint_path)
        dataset_dict["speaker1_file"].append(speaker1_file)
        dataset_dict["unique_id_1"].append(str(speaker1_file))
        dataset_dict["speaker2_file"].append(speaker2_file)
        dataset_dict["unique_id_2"].append(str(speaker2_file))

    ds = datasets.Dataset.from_dict(dataset_dict)
    ds = ds.sort(["unique_id_1", "unique_id_2"])
    ds = ds.cast_column("audio", datasets.Audio(sampling_rate=SAMPLING_RATE))
    ds = ds.add_column("dataset_index", list(range(len(ds))))
    ds = datasets.DatasetDict({"train": ds})
    ds.save_to_disk(DATA_PATH / dataset_name)

    # delete raw data
    shutil.rmtree(data_dir / "wav")
    shutil.rmtree(data_dir / "tmp")
    os.remove(data_dir / "meta.csv")


if __name__ == "__main__":
    download_dataset(DATASET1_NAME)
    # download_dataset(DATASET2_NAME)
