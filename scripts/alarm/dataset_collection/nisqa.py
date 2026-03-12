import json
import os
import shutil
from collections import defaultdict

import datasets
import pandas as pd
import torchaudio
from tqdm.auto import tqdm

from utils import DATA_PATH, SAMPLING_RATE, cli_download, get_audio_description

# https://arxiv.org/abs/2104.09494
DATASET_NAME = "nisqa"
URL_LINK = "https://depositonce.tu-berlin.de/bitstream/11303/13012.5/9/NISQA_Corpus.zip"


def download_dataset():
    dataset_dict = defaultdict(list)

    data_dir = DATA_PATH / DATASET_NAME
    data_dir.mkdir(exist_ok=True, parents=True)

    arc_path = data_dir / "nisqa.zip"
    cli_download(URL_LINK, output=str(arc_path), method="wget")
    shutil.unpack_archive(arc_path, data_dir)
    os.remove(arc_path)

    full_data_dir = data_dir / "NISQA_Corpus"
    metadata_path = full_data_dir / "NISQA_corpus_file.csv"
    full_metadata = pd.read_csv(metadata_path)
    for train_folder in ["NISQA_TRAIN_LIVE", "NISQA_TRAIN_SIM"]:
        metadata = full_metadata.loc[full_metadata["db"] == train_folder]
        for _, row in tqdm(
            metadata.iterrows(), total=metadata.shape[0], desc=train_folder
        ):
            f_id = row["filename_deg"]

            audio_path = str(full_data_dir / train_folder / "deg" / f_id)

            audio_frames, audio_sr = torchaudio.load(audio_path)
            duration = audio_frames.shape[-1] / audio_sr

            elem_info = {}

            text = "Speech"

            elem_info["duration"] = duration
            elem_info["text"] = text.strip()
            elem_info["mos"] = round(row["mos"], 3)
            # in the dataset, the more noise / color / distortion we have
            # the closer we are to the 5 value. But this is counterintuitive
            # so we replace it.
            elem_info["noise level"] = f"{round(5 - float(row['noi']), 3)}/5"
            elem_info["coloration level"] = f"{round(5 - float(row['col']), 3)}/5"
            elem_info["discontinuity"] = f"{round(5 - float(row['dis']), 3)}/5"
            elem_info["loudness"] = f"{round(row['loud'], 3)}/5"
            elem_info["degradation type"] = row["con_description"]
            # https://github.com/gabrielmittag/NISQA/issues/21
            elem_info["degradation explanation"] = (
                "[Noisiness (the smaller the level the better): the presence of the background, "
                "circuit, or coding noise; Coloration (the smaller the level the better): "
                "frequency response distortions, low-bitrate codecs, or packet-loss "
                "concealment; Discontinuity (the smaller the level the better): "
                "isolated or non-stationary distortions, e.g. introduced by packet-loss "
                "or clipping; "
                "Loudness (the higher the better): if small, the loudness is not ideal, "
                "either too loud or too quit signals or loudness variation]"
            )

            audio_description = get_audio_description(text_type="caption", **elem_info)

            dataset_dict["audio_description"].append(audio_description)
            for k, v in elem_info.items():
                dataset_dict[k].append(v)
            dataset_dict["audio"].append(audio_path)
            dataset_dict["filename"].append(f_id)
            dataset_dict["unique_id_1"].append(str(f_id))
            dataset_dict["db"].append(train_folder)
            dataset_dict["unique_id_2"].append(str(train_folder))

    ds = datasets.Dataset.from_dict(dataset_dict)
    ds = ds.sort(["unique_id_1", "unique_id_2"])
    ds = ds.cast_column("audio", datasets.Audio(sampling_rate=SAMPLING_RATE))
    ds = ds.add_column("dataset_index", list(range(len(ds))))
    ds = datasets.DatasetDict({"train": ds})
    ds.save_to_disk(DATA_PATH / DATASET_NAME)

    # delete raw data
    shutil.rmtree(data_dir / "NISQA_Corpus")


if __name__ == "__main__":
    download_dataset()
