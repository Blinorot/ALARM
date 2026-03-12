import os
import shutil
from collections import defaultdict

import datasets
import torchaudio
from tqdm.auto import tqdm

from utils import DATA_PATH, SAMPLING_RATE, cli_download, get_audio_description

# https://datashare.ed.ac.uk/handle/10283/2791
DATASET_NAME = "noisy_vctk"
URL_LINKS = {
    "metadata": "https://datashare.ed.ac.uk/bitstream/handle/10283/2791/logfiles.zip?sequence=4&isAllowed=y",
    "train28_audio": "https://datashare.ed.ac.uk/bitstream/handle/10283/2791/noisy_trainset_28spk_wav.zip?sequence=6&isAllowed=y",
    "train56_audio": "https://datashare.ed.ac.uk/bitstream/handle/10283/2791/noisy_trainset_56spk_wav.zip?sequence=7&isAllowed=y",
    "clean_train28_audio": "https://datashare.ed.ac.uk/bitstream/handle/10283/2791/clean_trainset_28spk_wav.zip?sequence=2&isAllowed=y",
    "clean_train56_audio": "https://datashare.ed.ac.uk/bitstream/handle/10283/2791/clean_trainset_56spk_wav.zip?sequence=3&isAllowed=y",
    "train28_text": "https://datashare.ed.ac.uk/bitstream/handle/10283/2791/trainset_28spk_txt.zip?sequence=9&isAllowed=y",
    "train56_text": "https://datashare.ed.ac.uk/bitstream/handle/10283/2791/trainset_56spk_txt.zip?sequence=10&isAllowed=y",
}


def download_dataset():
    dataset_dict = defaultdict(list)

    data_dir = DATA_PATH / DATASET_NAME
    data_dir.mkdir(exist_ok=True, parents=True)

    full_data_dir = data_dir / "raw_data"
    full_data_dir.mkdir(exist_ok=True, parents=True)

    for file, link in URL_LINKS.items():
        arc_path = full_data_dir / f"{file}.zip"
        cli_download(link, output=str(arc_path))
        shutil.unpack_archive(arc_path, full_data_dir)
        os.remove(str(arc_path))

    train_sets = ["trainset_28spk", "trainset_56spk"]
    for train_set in train_sets:
        metadata_path = full_data_dir / f"log_{train_set}.txt"
        with metadata_path.open("r") as f:
            metadata = f.readlines()[:-1]
        for line in tqdm(metadata, desc=f"Preparing {train_set}..."):
            line = line.strip()
            f_id, noise, snr = line.split()
            text_path = full_data_dir / f"{train_set}_txt" / f"{f_id}.txt"
            audio_path = full_data_dir / f"noisy_{train_set}_wav" / f"{f_id}.wav"

            audio_path = str(audio_path)
            text = text_path.read_text().strip()

            audio_frames, audio_sr = torchaudio.load(audio_path)
            duration = audio_frames.shape[-1] / audio_sr

            elem_info = {}

            elem_info["duration"] = duration
            elem_info["text"] = text

            # https://www.isca-archive.org/ssw_2016/valentinibotinhao16_ssw.html
            if noise == "ssn":
                noise = "simulated speech-shaped noise (filtered white noise)"

            elem_info["noise description"] = noise
            elem_info["snr"] = snr

            audio_description = get_audio_description(text_type="speech", **elem_info)

            dataset_dict["audio_description"].append(audio_description)
            for k, v in elem_info.items():
                dataset_dict[k].append(v)
            dataset_dict["audio"].append(audio_path)
            dataset_dict["train_set_f_id"].append(f"{train_set}_{f_id}")
            dataset_dict["unique_id_1"].append(str(f"{train_set}_{f_id}"))
            dataset_dict["noise_snr"].append(f"{noise}_{snr}")
            dataset_dict["unique_id_2"].append(str(f"{noise}_{snr}"))

            # clean part

            audio_path = full_data_dir / f"clean_{train_set}_wav" / f"{f_id}.wav"

            audio_path = str(audio_path)
            text = text_path.read_text().strip()

            audio_frames, audio_sr = torchaudio.load(audio_path)
            duration = audio_frames.shape[-1] / audio_sr

            elem_info = {}

            elem_info["duration"] = duration
            elem_info["text"] = text

            elem_info["noise description"] = "there is no noise - the audio is clean"
            elem_info["snr"] = "100"

            audio_description = get_audio_description(text_type="speech", **elem_info)

            dataset_dict["audio_description"].append(audio_description)
            for k, v in elem_info.items():
                dataset_dict[k].append(v)
            dataset_dict["audio"].append(audio_path)
            dataset_dict["train_set_f_id"].append(f"{train_set}_{f_id}")
            dataset_dict["unique_id_1"].append(str(f"{train_set}_{f_id}"))
            dataset_dict["noise_snr"].append(f"clean_{snr}")
            dataset_dict["unique_id_2"].append(str(f"clean_{snr}"))

    ds = datasets.Dataset.from_dict(dataset_dict)
    ds = ds.sort(["unique_id_1", "unique_id_2"])
    ds = ds.cast_column("audio", datasets.Audio(sampling_rate=SAMPLING_RATE))
    ds = ds.add_column("dataset_index", list(range(len(ds))))
    ds = datasets.DatasetDict({"train": ds})
    ds.save_to_disk(DATA_PATH / DATASET_NAME)

    # # remove raw data
    shutil.rmtree(full_data_dir)


if __name__ == "__main__":
    download_dataset()
