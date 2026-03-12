import os
import shutil
from collections import defaultdict

import datasets
import torchaudio
from tqdm.auto import tqdm

from utils import DATA_PATH, SAMPLING_RATE, cli_download, get_audio_description

# https://datashare.ed.ac.uk/handle/10283/2950
# https://datashare.ed.ac.uk/handle/10283/2651 -- this version
DATASET_NAME = "vctk"
URL_LINK = "https://datashare.ed.ac.uk/bitstream/handle/10283/2651/VCTK-Corpus.zip?sequence=2&isAllowed=y"


def download_dataset():
    dataset_dict = defaultdict(list)

    data_dir = DATA_PATH / DATASET_NAME
    data_dir.mkdir(exist_ok=True, parents=True)

    arc_path = data_dir / "vctk.zip"
    cli_download(URL_LINK, output=str(arc_path))
    shutil.unpack_archive(arc_path, data_dir)
    os.remove(str(arc_path))

    full_data_dir = data_dir / "VCTK-Corpus"
    metadata = (full_data_dir / "speaker-info.txt").read_text().split("\n")[1:-1]
    speaker_info = {}
    for line in metadata:
        line_no_comment = line.split("(")[0]
        line_split = line_no_comment.split()
        speaker_id = line_split[0]
        speaker_age = line_split[1]
        speaker_gender = "Female" if line_split[2] == "F" else "Male"
        speaker_accent = line_split[3]
        if len(line_split) > 4:
            speaker_region = " ".join(line_split[4:])
        else:
            speaker_region = "Unknown"
        speaker_info[speaker_id] = {
            "age": speaker_age,
            "accent": speaker_accent,
            "gender": speaker_gender,
            "region": speaker_region,
        }

    for speaker_id in tqdm(speaker_info.keys(), desc="Preparing VCTK..."):
        text_dir = full_data_dir / "txt" / f"p{speaker_id}"
        audio_dir = full_data_dir / "wav48" / f"p{speaker_id}"
        if not text_dir.exists() or not audio_dir.exists():
            continue
        for filename in os.listdir(str(text_dir)):
            text_path = text_dir / filename
            f_id = text_path.stem
            audio_path = audio_dir / f"{f_id}.wav"
            if not text_path.exists() or not audio_path.exists():
                continue

            audio_path = str(audio_path)
            text = text_path.read_text().strip()

            audio_frames, audio_sr = torchaudio.load(audio_path)
            duration = audio_frames.shape[-1] / audio_sr

            elem_info = {}

            elem_info["duration"] = duration
            elem_info["text"] = text
            for k, v in speaker_info[speaker_id].items():
                elem_info[k] = v

            audio_description = get_audio_description(text_type="speech", **elem_info)

            dataset_dict["audio_description"].append(audio_description)
            for k, v in elem_info.items():
                dataset_dict[k].append(v)
            dataset_dict["audio"].append(audio_path)
            dataset_dict["speaker_id"].append(speaker_id)
            dataset_dict["unique_id_1"].append(str(speaker_id))
            dataset_dict["file_id"].append(f_id)
            dataset_dict["unique_id_2"].append(str(f_id))

    ds = datasets.Dataset.from_dict(dataset_dict)
    ds = ds.sort(["unique_id_1", "unique_id_2"])
    ds = ds.cast_column("audio", datasets.Audio(sampling_rate=SAMPLING_RATE))
    ds = ds.add_column("dataset_index", list(range(len(ds))))
    ds = datasets.DatasetDict({"train": ds})
    ds.save_to_disk(DATA_PATH / DATASET_NAME)

    # remove raw data
    shutil.rmtree(full_data_dir)


if __name__ == "__main__":
    download_dataset()
