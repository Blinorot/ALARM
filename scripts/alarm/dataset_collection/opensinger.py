import json
import os
import shutil
import subprocess
from collections import defaultdict

import datasets
import torchaudio
from tqdm.auto import tqdm

from utils import DATA_PATH, SAMPLING_RATE, cli_download, get_audio_description

# https://arxiv.org/abs/2112.10358
DATASET_NAME = "opensinger"
URL_LINK = (
    "https://drive.google.com/file/d/1EofoZxvalgMjZqzUEuEdleHIZ6SHtNuK/view?usp=sharing"
)


def download_dataset():
    dataset_dict = defaultdict(list)

    data_dir = DATA_PATH / DATASET_NAME
    data_dir.mkdir(exist_ok=True, parents=True)

    arc_path = data_dir / "OpenSinger.tar.gz"
    cli_download(URL_LINK, output=str(arc_path), method="gdown_file")
    subprocess.run(["tar", "-xzf", str(arc_path), "-C", str(data_dir)], check=True)
    os.remove(arc_path)

    full_data_dir = data_dir / "OpenSinger"

    for dir in ["ManRaw", "WomanRaw"]:
        gender = "Male" if dir == "ManRaw" else "Female"
        wav_dir = full_data_dir / dir
        for song in tqdm(os.listdir(wav_dir), desc=f"Processing {dir}"):
            song_dir = wav_dir / song
            if not song_dir.is_dir():
                continue
            song_split = song.split("_")
            assert len(song_split) == 2, f"Wrong split length. Got {song_split}"
            song_name = song_split[1]
            for utterance in os.listdir(song_dir):
                if not utterance.endswith(".wav"):
                    continue
                audio_path = song_dir / utterance

                text_path = audio_path.with_suffix(".txt")
                text = text_path.read_text()

                audio_frames, audio_sr = torchaudio.load(audio_path)
                duration = audio_frames.shape[-1] / audio_sr
                audio_path = str(audio_path)

                elem_info = {}

                elem_info["duration"] = duration
                elem_info["text"] = text
                elem_info["song name"] = song_name
                elem_info["gender"] = gender

                audio_description = get_audio_description(
                    text_type="speech", **elem_info
                )

                dataset_dict["audio_description"].append(audio_description)
                for k, v in elem_info.items():
                    dataset_dict[k].append(v)
                dataset_dict["audio"].append(audio_path)
                dataset_dict["utterance_id"].append(utterance)
                dataset_dict["unique_id_2"].append(str(utterance))
                # extra word so it does not look like the corresponding metadata field
                # needed for prompt creation
                dataset_dict["unique_id_1"].append(f"gender: {gender}")

    ds = datasets.Dataset.from_dict(dataset_dict)
    ds = ds.sort(["unique_id_1", "unique_id_2"])
    ds = ds.cast_column("audio", datasets.Audio(sampling_rate=SAMPLING_RATE))
    ds = ds.add_column("dataset_index", list(range(len(ds))))
    ds = datasets.DatasetDict({"train": ds})
    ds.save_to_disk(DATA_PATH / DATASET_NAME)

    # delete raw data
    shutil.rmtree(full_data_dir)


if __name__ == "__main__":
    download_dataset()
