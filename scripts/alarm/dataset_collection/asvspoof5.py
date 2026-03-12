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

# https://arxiv.org/abs/2408.08739
DATASET_NAME = "asvspoof5"
URL_LINKS = {
    "a": "https://huggingface.co/datasets/jungjee/asvspoof5/resolve/37a01fa724b6329c2232d26842313abdf935bebc/flac_T_aa.tar",
    "b": "https://huggingface.co/datasets/jungjee/asvspoof5/resolve/37a01fa724b6329c2232d26842313abdf935bebc/flac_T_ab.tar",
    "c": "https://huggingface.co/datasets/jungjee/asvspoof5/resolve/37a01fa724b6329c2232d26842313abdf935bebc/flac_T_ac.tar",
    "d": "https://huggingface.co/datasets/jungjee/asvspoof5/resolve/37a01fa724b6329c2232d26842313abdf935bebc/flac_T_ad.tar",
    "e": "https://huggingface.co/datasets/jungjee/asvspoof5/resolve/37a01fa724b6329c2232d26842313abdf935bebc/flac_T_ae.tar",
    "protocol": "https://huggingface.co/datasets/jungjee/asvspoof5/resolve/main/ASVspoof5_protocols.tar",
}

ATTACK2TYPE = {
    "A01": "Text-To-Speech using GlowTTS",
    "A02": "Text-To-Speech using GlowTTS",
    "A03": "Text-To-Speech using GlowTTS",
    "A04": "Text-To-Speech using GradTTS",
    "A05": "Text-To-Speech using GradTTS",
    "A06": "Text-To-Speech using GradTTS",
    "A07": "Text-To-Speech using FastPitch",
    "A08": "Text-To-Speech using VITS",
    "bonafide": "None because this is a real audio",
}


def download_dataset():
    dataset_dict = defaultdict(list)

    data_dir = DATA_PATH / DATASET_NAME
    data_dir.mkdir(exist_ok=True, parents=True)

    for doc, link in URL_LINKS.items():
        arc_path = data_dir / f"{doc}.tar"
        cli_download(URL_LINKS[doc], output=str(arc_path), method="wget")
        subprocess.run(["tar", "-xf", str(arc_path), "-C", str(data_dir)], check=True)
        os.remove(arc_path)

    metadata_path = data_dir / "ASVspoof5.train.tsv"
    with metadata_path.open("r") as f:
        metadata = f.readlines()[:-1]

    for line in tqdm(metadata):
        speaker_id, f_id, gender, _, _, _, _, system_id, _, _ = line.split()

        audio_path = str(data_dir / "flac_T" / f"{f_id}.flac")

        audio_frames, audio_sr = torchaudio.load(audio_path)
        duration = audio_frames.shape[-1] / audio_sr

        elem_info = {}

        elem_info["duration"] = duration
        elem_info["text"] = "Speech"
        elem_info["is bona fide or spoof"] = (
            "bona fide" if system_id == "-" else "spoof"
        )
        elem_info["spoof algorithm description"] = ATTACK2TYPE[system_id]

        audio_description = get_audio_description(text_type="caption", **elem_info)

        dataset_dict["audio_description"].append(audio_description)
        for k, v in elem_info.items():
            dataset_dict[k].append(v)
        dataset_dict["audio"].append(audio_path)
        dataset_dict["f_id"].append(f_id)
        dataset_dict["unique_id_1"].append(str(f_id))
        dataset_dict["speaker_id"].append(speaker_id)
        dataset_dict["unique_id_2"].append(str(speaker_id))

    ds = datasets.Dataset.from_dict(dataset_dict)
    ds = ds.sort(["unique_id_1", "unique_id_2"])
    ds = ds.cast_column("audio", datasets.Audio(sampling_rate=SAMPLING_RATE))
    ds = ds.add_column("dataset_index", list(range(len(ds))))
    ds = datasets.DatasetDict({"train": ds})
    ds.save_to_disk(DATA_PATH / DATASET_NAME)

    # delete raw data
    for fname in os.listdir(data_dir):
        if "ASVspoof5" in fname:
            os.remove(data_dir / fname)
    shutil.rmtree(data_dir / "flac_T")


if __name__ == "__main__":
    download_dataset()
