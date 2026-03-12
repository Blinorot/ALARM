import json
import os
import shutil
import subprocess
from collections import defaultdict

import datasets
from torchcodec.decoders import AudioDecoder
from tqdm.auto import tqdm

from utils import DATA_PATH, SAMPLING_RATE, cli_download, get_audio_description

# https://arxiv.org/abs/2012.03411
DATASET_NAME = "mls"
URL_LINK = {
    "german": "https://dl.fbaipublicfiles.com/mls/mls_german_opus.tar.gz",
    "dutch": "https://dl.fbaipublicfiles.com/mls/mls_dutch_opus.tar.gz",
    "french": "https://dl.fbaipublicfiles.com/mls/mls_french_opus.tar.gz",
    "spanish": "https://dl.fbaipublicfiles.com/mls/mls_spanish_opus.tar.gz",
    "italian": "https://dl.fbaipublicfiles.com/mls/mls_italian_opus.tar.gz",
    "portuguese": "https://dl.fbaipublicfiles.com/mls/mls_portuguese_opus.tar.gz",
    "polish": "https://dl.fbaipublicfiles.com/mls/mls_polish_opus.tar.gz",
}


def download_dataset():
    dataset_dict = defaultdict(list)

    data_dir = DATA_PATH / DATASET_NAME
    data_dir.mkdir(exist_ok=True, parents=True)

    for language, url_link in URL_LINK.items():
        language_dir = data_dir / language
        language_dir.mkdir(exist_ok=True, parents=True)

        arc_path = language_dir / "data.tar.gz"
        cli_download(url_link, output=str(arc_path), method="wget")
        cores = os.cpu_count()
        subprocess.run(
            [
                "tar",
                f"--use-compress-program=pigz -d -p {cores}",
                "-xf",
                str(arc_path),
                "-C",
                str(language_dir),
            ],
            check=True,
        )
        os.remove(arc_path)

        full_data_dir = language_dir / f"mls_{language}_opus"
        metadata_path = full_data_dir / "metainfo.txt"
        speaker2gender = {}
        with metadata_path.open("r") as f:
            lines = f.readlines()[1:-1]
            for line in lines:
                line_split = line.split("|")
                speaker_id = line_split[0].strip()
                gender = line_split[1].strip()
                if gender == "M":
                    gender = "Male"
                else:
                    gender = "Female"
                speaker2gender[speaker_id] = gender

        transcripts_path = full_data_dir / "train" / "transcripts.txt"
        with transcripts_path.open("r") as f:
            lines = f.readlines()[:-1]

        audio_dir = full_data_dir / "train" / "audio"

        for line in tqdm(lines, desc=f"Processing {language}"):
            utterance_id, text = line.split("\t")

            speaker_id, book_id, _ = utterance_id.split("_")

            audio_path = audio_dir / speaker_id / book_id / f"{utterance_id}.opus"

            # torchaudio does not work for some reason
            audio_metadata = AudioDecoder(audio_path).get_all_samples()
            duration = audio_metadata.duration_seconds
            audio_path = str(audio_path)

            elem_info = {}

            elem_info["duration"] = duration
            elem_info["text"] = text.strip()
            elem_info["gender"] = speaker2gender[speaker_id]

            audio_description = get_audio_description(text_type="speech", **elem_info)

            dataset_dict["audio_description"].append(audio_description)
            for k, v in elem_info.items():
                dataset_dict[k].append(v)
            dataset_dict["audio"].append(audio_path)
            dataset_dict["utterance_id"].append(utterance_id)
            dataset_dict["unique_id_2"].append(str(utterance_id))
            dataset_dict["language"].append(language)
            dataset_dict["unique_id_1"].append(str(language))

    ds = datasets.Dataset.from_dict(dataset_dict)
    ds = ds.sort(["unique_id_1", "unique_id_2"])
    ds = ds.cast_column("audio", datasets.Audio(sampling_rate=SAMPLING_RATE))
    ds = ds.add_column("dataset_index", list(range(len(ds))))
    ds = datasets.DatasetDict({"train": ds})
    ds.save_to_disk(DATA_PATH / DATASET_NAME)

    # delete raw data
    for language in URL_LINK.keys():
        language_dir = data_dir / language
        shutil.rmtree(language_dir)


if __name__ == "__main__":
    download_dataset()
