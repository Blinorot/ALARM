import json
import os
import shutil
import subprocess
from collections import defaultdict

import datasets
from torchcodec.decoders import AudioDecoder
from tqdm.auto import tqdm

from utils import DATA_PATH, SAMPLING_RATE, cli_download, get_audio_description

# https://ieeexplore.ieee.org/document/7178964
DATASET_NAME = "librispeech"
URL_LINK = {
    "train-clean-100": "https://openslr.trmal.net/resources/12/train-clean-100.tar.gz",
    "train-clean-360": "https://openslr.trmal.net/resources/12/train-clean-360.tar.gz",
    "train-other-500": "https://openslr.trmal.net/resources/12/train-other-500.tar.gz",
}


def download_dataset():
    dataset_dict = defaultdict(list)

    data_dir = DATA_PATH / DATASET_NAME
    data_dir.mkdir(exist_ok=True, parents=True)

    for subset, url_link in URL_LINK.items():
        subset_dir = data_dir / subset
        subset_dir.mkdir(exist_ok=True, parents=True)

        arc_path = subset_dir / "data.tar.gz"
        cli_download(url_link, output=str(arc_path), method="wget")
        cores = os.cpu_count()
        subprocess.run(
            [
                "tar",
                f"--use-compress-program=pigz -d -p {cores}",
                "-xf",
                str(arc_path),
                "-C",
                str(subset_dir),
            ],
            check=True,
        )
        os.remove(arc_path)

        full_data_dir = subset_dir / "LibriSpeech"
        metadata_path = full_data_dir / "SPEAKERS.TXT"
        speaker2gender = {}
        with metadata_path.open("r") as f:
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

        full_data_dir = full_data_dir / subset

        for speaker_id in tqdm(os.listdir(full_data_dir), desc=f"Processing {subset}"):
            speaker_dir = full_data_dir / speaker_id
            for book_id in os.listdir(speaker_dir):
                transcription_list_path = (
                    speaker_dir / book_id / f"{speaker_id}-{book_id}.trans.txt"
                )
                utterance2text = {}
                with transcription_list_path.open("r") as f:
                    transcription_list = f.readlines()
                    for line in transcription_list:
                        line_split = line.split(" ")
                        utterance_id = line_split[0]
                        text = " ".join(line_split[1:])
                        text = text.lower().strip()
                        utterance2text[utterance_id] = text

                for utterance_id, text in utterance2text.items():
                    audio_path = speaker_dir / book_id / f"{utterance_id}.flac"

                    # torchaudio does not work for some reason
                    audio_metadata = AudioDecoder(audio_path).get_all_samples()
                    duration = audio_metadata.duration_seconds
                    audio_path = str(audio_path)

                    elem_info = {}

                    elem_info["duration"] = duration
                    elem_info["text"] = text.strip()
                    elem_info["gender"] = speaker2gender[speaker_id]

                    audio_description = get_audio_description(
                        text_type="speech", **elem_info
                    )

                    dataset_dict["audio_description"].append(audio_description)
                    for k, v in elem_info.items():
                        dataset_dict[k].append(v)
                    dataset_dict["audio"].append(audio_path)
                    dataset_dict["utterance_id"].append(utterance_id)
                    dataset_dict["unique_id_2"].append(str(utterance_id))
                    dataset_dict["speaker_id"].append(speaker_id)
                    dataset_dict["unique_id_1"].append(str(speaker_id))

    ds = datasets.Dataset.from_dict(dataset_dict)
    ds = ds.sort(["unique_id_1", "unique_id_2"])
    ds = ds.cast_column("audio", datasets.Audio(sampling_rate=SAMPLING_RATE))
    ds = ds.add_column("dataset_index", list(range(len(ds))))
    ds = datasets.DatasetDict({"train": ds})
    ds.save_to_disk(DATA_PATH / DATASET_NAME)

    # delete raw data
    for subset in URL_LINK.keys():
        subset_dir = data_dir / subset
        shutil.rmtree(subset_dir)


if __name__ == "__main__":
    download_dataset()
