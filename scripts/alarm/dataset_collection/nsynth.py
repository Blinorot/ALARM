import json
import os
import shutil
import subprocess
from collections import defaultdict

import datasets
import torchaudio
from tqdm.auto import tqdm

from utils import DATA_PATH, SAMPLING_RATE, cli_download, get_audio_description

# https://arxiv.org/abs/1704.01279
DATASET_NAME = "nsynth"
URL_LINK = (
    "http://download.magenta.tensorflow.org/datasets/nsynth/nsynth-train.jsonwav.tar.gz"
)

# see https://magenta.tensorflow.org/datasets/nsynth#note-qualities
QUALITY_DESC = {
    "bright": "A large amount of high frequency content and strong upper harmonics.",
    "dark": "A distinct lack of high frequency content, giving a muted and bassy sound. Also sometimes described as 'Warm'.",
    "distortion": "Waveshaping that produces a distinctive crunchy sound and presence of many harmonics. Sometimes paired with non-harmonic noise.",
    "fast_decay": "Amplitude envelope of all harmonics decays substantially before the 'note-off' point at 3 seconds.",
    "long_release": "Amplitude envelope decays slowly after the 'note-off' point, sometimes still present at the end of the sample 4 seconds.",
    "multiphonic": "Presence of overtone frequencies related to more than one fundamental frequency.",
    "nonlinear_env": "Modulation of the sound with a distinct envelope behavior different than the monotonic decrease of the note. Can also include filter envelopes as well as dynamic envelopes.",
    "percussive": "A loud non-harmonic sound at note onset.",
    "reverb": "Room acoustics that were not able to be removed from the original sample.",
    "tempo-synced": "Rhythmic modulation of the sound to a fixed tempo.",
}

INSTRUMENT_FAMILY = {
    0: "bass",
    1: "brass",
    2: "flute",
    3: "guitar",
    4: "keyboard",
    5: "mallet",
    6: "organ",
    7: "reed",
    8: "string",
    9: "synth lead",
    10: "vocal",
}

INSTRUMENT_SOURCE = {
    0: "acoustic",
    1: "electronic",
    2: "synthetic",
}


def download_dataset():
    dataset_dict = defaultdict(list)

    data_dir = DATA_PATH / DATASET_NAME
    data_dir.mkdir(exist_ok=True, parents=True)

    arc_path = data_dir / "nsynth.tar.gz"
    cli_download(URL_LINK, output=str(arc_path))
    subprocess.run(["tar", "-xzf", str(arc_path), "-C", str(data_dir)], check=True)
    os.remove(arc_path)

    metadata_path = data_dir / "nsynth-train" / "examples.json"
    with metadata_path.open("r") as f:
        metadata = json.load(f)

    for fname in tqdm(metadata.keys()):
        fdata = metadata[fname]

        note_id = fdata["note"]
        audio_path = data_dir / "nsynth-train" / "audio" / f"{fname}.wav"

        text = f"{INSTRUMENT_FAMILY[fdata['instrument_family']]} musical note"

        audio_frames, audio_sr = torchaudio.load(audio_path)
        duration = audio_frames.shape[-1] / audio_sr
        audio_path = str(audio_path)

        elem_info = {}

        elem_info["duration"] = duration
        elem_info["text"] = text
        elem_info["MIDI pitch"] = f"{fdata['pitch']} out of 127"
        elem_info["MIDI velocity"] = f"{fdata['velocity']} out of 127"
        elem_info["instrument source"] = INSTRUMENT_SOURCE[
            fdata["instrument_source"]
        ].title()
        elem_info["instrument family"] = INSTRUMENT_FAMILY[
            fdata["instrument_family"]
        ].title()

        qualities = fdata["qualities_str"]
        if len(qualities) == 0:
            elem_info["special qualitative categories"] = "None"
            elem_info[
                "special qualitative categories description"
            ] = "No special categories for this audio"
        else:
            descriptions = [QUALITY_DESC[elem] for elem in qualities]
            elem_info["special qualitative categories"] = "/".join(qualities)
            elem_info["special qualitative categories description"] = (
                "'" + " ".join(descriptions) + "'"
            )

        audio_description = get_audio_description(text_type="caption", **elem_info)

        dataset_dict["audio_description"].append(audio_description)
        for k, v in elem_info.items():
            dataset_dict[k].append(v)
        dataset_dict["audio"].append(audio_path)
        dataset_dict["note_id"].append(note_id)
        dataset_dict["unique_id_2"].append(str(note_id))
        dataset_dict["fname"].append(fname)
        dataset_dict["unique_id_1"].append(str(fname))

    ds = datasets.Dataset.from_dict(dataset_dict)
    ds = ds.sort(["unique_id_1", "unique_id_2"])
    ds = ds.cast_column("audio", datasets.Audio(sampling_rate=SAMPLING_RATE))
    ds = ds.add_column("dataset_index", list(range(len(ds))))
    ds = datasets.DatasetDict({"train": ds})
    ds.save_to_disk(DATA_PATH / DATASET_NAME)

    # delete raw data
    shutil.rmtree(data_dir / "nsynth-train")


if __name__ == "__main__":
    download_dataset()
