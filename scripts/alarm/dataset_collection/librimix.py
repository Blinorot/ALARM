import json
import os
import shutil
import subprocess
import sys
from collections import defaultdict

import datasets
import pandas as pd
from torchcodec.decoders import AudioDecoder
from tqdm.auto import tqdm

from utils import DATA_PATH, SAMPLING_RATE, cli_download, get_audio_description

# https://arxiv.org/abs/2005.11262
# based on https://github.com/JorisCos/LibriMix
DATASET_NAME = "librimix"
URL_LINK = {
    "train-clean-100": "https://openslr.trmal.net/resources/12/train-clean-100.tar.gz",
    "train-clean-360": "https://openslr.trmal.net/resources/12/train-clean-360.tar.gz",
    "wham": "https://my-bucket-a8b4b49c25c811ee9a7e8bba05fa24c7.s3.amazonaws.com/wham_noise.zip",
    "repo": "https://raw.githubusercontent.com/JorisCos/LibriMix/eac7cf0b24ebadd50e7167c7be98c2366b1be0c4/",
}

META_FILES = [
    "Libri2Mix/libri2mix_train-clean-100.csv",
    "Libri2Mix/libri2mix_train-clean-360.csv",
    "Libri3Mix/libri3mix_train-clean-100.csv",
    "Libri3Mix/libri3mix_train-clean-360.csv",
    "Libri2Mix/libri2mix_train-clean-100_info.csv",
    "Libri2Mix/libri2mix_train-clean-360_info.csv",
    "Libri3Mix/libri3mix_train-clean-100_info.csv",
    "Libri3Mix/libri3mix_train-clean-360_info.csv",
]
ORDER2TEXT = {
    1: "first",
    2: "second",
    3: "third",
}


def add_elem_to_dataset(
    dataset_dict,
    audio_path,
    mixture_id,
    partition_id,
    joint_text_list,
    joint_duration_list,
    speaker2gender,
    speaker2order,
    speaker2snr,
):
    elem_info = {}

    processed_joint_duration_list = []
    min_start_time = -1
    all_speakers = set()
    for start_time, end_time, speaker_id in joint_duration_list:
        if min_start_time == -1:  # the first time
            min_start_time = start_time
        start_time -= min_start_time
        end_time -= min_start_time
        speaker_info = f"the {speaker2order[speaker_id]} speaker"
        processed_joint_duration_list.append((start_time, end_time, speaker_info))
        all_speakers.add(speaker_id)

    for speaker_id in all_speakers:
        order = speaker2order[speaker_id]
        gender = speaker2gender[speaker_id]
        snr = speaker2snr[speaker_id]
        elem_info[f"the {order} speaker's gender"] = gender
        elem_info[f"the {order} speaker's mixture snr"] = snr

    elem_info["the number of speakers"] = len(all_speakers)

    elem_info["duration"] = processed_joint_duration_list
    elem_info["text"] = joint_text_list

    audio_description = get_audio_description(text_type="speech", **elem_info)

    dataset_dict["audio_description"].append(audio_description)

    elem_info["duration"] = elem_info["duration"][-1][1]
    elem_info["text"] = ";".join(elem_info["text"])

    # since elem_info depends on the number of speakers
    # we need to add all possible keys
    # to ensure that all fields are present
    for _, order in ORDER2TEXT.items():
        key = f"the {order} speaker's gender"
        elem_info[key] = elem_info.get(key, "")
        key = f"the {order} speaker's mixture snr"
        elem_info[key] = elem_info.get(key, 0.0)

    for k, v in elem_info.items():
        dataset_dict[k].append(v)
    dataset_dict["audio"].append(audio_path)
    dataset_dict["mixture_id"].append(mixture_id)
    dataset_dict["unique_id_2"].append(str(mixture_id))
    dataset_dict["partition_id"].append(partition_id)
    dataset_dict["unique_id_1"].append(str(partition_id))


def download_dataset():
    dataset_dict = defaultdict(list)

    data_dir = DATA_PATH / DATASET_NAME
    data_dir.mkdir(exist_ok=True, parents=True)

    for subset in ["train-clean-100", "train-clean-360"]:
        url_link = URL_LINK[subset]

        arc_path = data_dir / f"{subset}.tar.gz"
        cli_download(url_link, output=str(arc_path), method="wget")
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

    arc_path = data_dir / "wham.zip"
    cli_download(URL_LINK["wham"], output=str(arc_path), method="wget")
    shutil.unpack_archive(arc_path, data_dir)

    metadata_dir = data_dir / "metadata"
    metadata_dir.mkdir(exist_ok=True, parents=True)
    for meta_file in META_FILES:
        meta_url = URL_LINK["repo"] + "metadata/" + meta_file
        meta_dir, meta_name = meta_file.split("/")
        save_dir = metadata_dir / meta_dir
        save_dir.mkdir(exist_ok=True, parents=True)
        save_path = save_dir / meta_name
        cli_download(meta_url, output=save_path, method="wget")

    script_url = URL_LINK["repo"] + "scripts/create_librimix_from_metadata.py"
    script_path = data_dir / "script.py"
    cli_download(script_url, output=script_path, method="wget")

    librispeech_dir = data_dir / "LibriSpeech"
    librimix_outdir = data_dir / "LibriMix"
    wham_dir = data_dir / "wham_noise"
    metadata_dir = data_dir / "metadata"

    librimix_outdir.mkdir(exist_ok=True, parents=True)
    # create mixtures
    for n_src in [2, 3]:
        # fix metadata
        # correct wav name should not have spXX at the end
        n_src_metadata = metadata_dir / f"Libri{n_src}Mix"

        wrong_metadata_path = n_src_metadata / f"libri{n_src}mix_train-clean-360.csv"
        df = pd.read_csv(wrong_metadata_path)
        df["noise_path"] = df["noise_path"].str.replace(
            r"sp\d{2}\.wav$", ".wav", regex=True
        )
        df.to_csv(wrong_metadata_path)

        subprocess.run(
            [
                sys.executable,
                str(script_path),
                "--librispeech_dir",
                str(librispeech_dir),
                "--wham_dir",
                str(wham_dir),
                "--metadata_dir",
                str(n_src_metadata),
                "--librimix_outdir",
                str(librimix_outdir),
                "--n_src",
                str(n_src),
                "--freqs",
                "16k",
                "--modes",
                "max",
                "--types",
                "mix_clean",
                "mix_both",
                "mix_single",
            ],
            check=True,
        )

    speakers_meta_path = librispeech_dir / "SPEAKERS.TXT"
    speaker2gender = {}
    with speakers_meta_path.open("r") as f:
        all_lines = f.readlines()
        speaker_lines = []
        for line in all_lines:
            if len(line) == 0:
                continue
            if line[0] == ";":
                continue
            speaker_lines.append(line)
        for line in speaker_lines:
            line_split = line.split("|")
            speaker_id = line_split[0].strip()
            gender = line_split[1].strip()
            if gender == "M":
                gender = "Male"
            else:
                gender = "Female"
            speaker2gender[speaker_id] = gender

    for n_src in [2, 3]:
        librimix_dir = librimix_outdir / f"Libri{n_src}Mix" / "wav16k" / "max"

        for subset in ["train-100", "train-360"]:
            # take only clean data
            filelist_path = (
                librimix_dir / "metadata" / f"metrics_{subset}_mix_clean.csv"
            )
            filelist = pd.read_csv(filelist_path)
            desc = f"Processing Libri{n_src}Mix {subset} mix clean..."
            for _, row in tqdm(filelist.iterrows(), total=filelist.shape[0], desc=desc):
                mixture_id = row["mixture_ID"]
                utterance_ids = mixture_id.split("_")
                utterance_snrs = row.iloc[1:].values

                text_list = []
                duration_list = []
                speaker2snr = {}
                speaker2order = {}
                order = 1

                for utterance_id, utterance_snr in zip(utterance_ids, utterance_snrs):
                    speaker_id, book_id, _ = utterance_id.split("-")
                    if speaker_id not in speaker2order.keys():
                        speaker2order[speaker_id] = ORDER2TEXT[order]
                        order += 1

                        speaker2snr[speaker_id] = round(utterance_snr, 3)

                    if subset == "train-100":
                        libri_subset = "train-clean-100"
                    else:
                        libri_subset = "train-clean-360"

                    audio_dir = librispeech_dir / libri_subset / speaker_id / book_id
                    audio_path = audio_dir / f"{utterance_id}.flac"

                    # torchaudio does not work for some reason
                    audio_metadata = AudioDecoder(audio_path).get_all_samples()
                    duration = audio_metadata.duration_seconds

                    start_time = 0
                    end_time = duration

                    duration_list.append((start_time, end_time, speaker_id))

                    transcription_list_path = (
                        audio_dir / f"{speaker_id}-{book_id}.trans.txt"
                    )
                    with transcription_list_path.open("r") as f:
                        transcription_list = f.readlines()
                        for line in transcription_list:
                            line_split = line.split(" ")
                            text_utterance_id = line_split[0]
                            if text_utterance_id != utterance_id:
                                continue
                            text = " ".join(line_split[1:])
                            text = text.lower().strip()
                    text_list.append(text)

                audio_path = librimix_dir / subset / "mix_clean" / f"{mixture_id}.wav"
                assert audio_path.exists()

                add_elem_to_dataset(
                    dataset_dict=dataset_dict,
                    audio_path=str(audio_path),
                    mixture_id=mixture_id,
                    partition_id=f"{subset}_mix_clean",
                    joint_text_list=text_list,
                    joint_duration_list=duration_list,
                    speaker2gender=speaker2gender,
                    speaker2order=speaker2order,
                    speaker2snr=speaker2snr,
                )

    ds = datasets.Dataset.from_dict(dataset_dict)
    ds = ds.sort(["unique_id_1", "unique_id_2"])
    ds = ds.cast_column("audio", datasets.Audio(sampling_rate=SAMPLING_RATE))
    ds = ds.add_column("dataset_index", list(range(len(ds))))
    ds = datasets.DatasetDict({"train": ds})
    ds.save_to_disk(DATA_PATH / DATASET_NAME)

    # delete raw data
    shutil.rmtree(librimix_outdir)
    shutil.rmtree(librispeech_dir)
    shutil.rmtree(wham_dir)
    shutil.rmtree(metadata_dir)
    os.remove(data_dir / "script.py")


if __name__ == "__main__":
    download_dataset()
