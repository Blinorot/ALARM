import json
import os
import shutil
import subprocess
from collections import defaultdict

import datasets
import torchaudio
from tqdm.auto import tqdm

from utils import DATA_PATH, SAMPLING_RATE, cli_download, get_audio_description

# https://arxiv.org/abs/2104.00235
# https://www.openslr.org/104
DATASET_NAME = "mucs"
URL_LINKS = {
    "Hindi-English": "https://openslr.trmal.net/resources/104/Hindi-English_train.tar.gz",
    "Bengali-English": "https://openslr.trmal.net/resources/104/Bengali-English_train.tar.gz",
}


def download_dataset():
    dataset_dict = defaultdict(list)

    data_dir = DATA_PATH / DATASET_NAME
    data_dir.mkdir(exist_ok=True, parents=True)

    for language, url_link in URL_LINKS.items():
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

        full_data_dir = language_dir / "train"
        tmp_dir = language_dir / "tmp"
        tmp_dir.mkdir(exist_ok=True, parents=True)

        segments = (
            (full_data_dir / "transcripts" / "segments").read_text().split("\n")[:-1]
        )
        transcripts = (
            (full_data_dir / "transcripts" / "text").read_text().split("\n")[:-1]
        )
        assert len(segments) == len(transcripts), "Wrong length."

        for segment_line, transcript_line in tqdm(
            zip(segments, transcripts), total=len(segments)
        ):
            segment_split = segment_line.strip().split()
            f_id = segment_split[0]
            text_id = transcript_line.strip()[: len(f_id)]
            text = transcript_line.strip()[len(f_id) :].strip()

            assert f_id == text_id, "File id mismatch"

            f_id = segment_split[0]

            audio_name = segment_split[1]
            audio_path = full_data_dir / f"{audio_name}.wav"
            audio, sr = torchaudio.load(audio_path)
            start_time = int(float(segment_split[-2]) * sr)
            end_time = int(float(segment_split[-1]) * sr)
            audio = audio[..., start_time:end_time]
            if sr != SAMPLING_RATE:
                audio = torchaudio.functional.resample(audio, sr, SAMPLING_RATE)
            duration = audio.shape[-1] / SAMPLING_RATE
            audio_path = str(tmp_dir / f"{f_id}.wav")
            torchaudio.save(audio_path, audio, sample_rate=SAMPLING_RATE)

            elem_info = {}

            elem_info["duration"] = duration
            elem_info["text"] = text
            elem_info["language"] = f"{language} code-switching"

            audio_description = get_audio_description(text_type="speech", **elem_info)

            dataset_dict["audio_description"].append(audio_description)
            for k, v in elem_info.items():
                dataset_dict[k].append(v)
            dataset_dict["audio"].append(audio_path)
            dataset_dict["f_id"].append(f_id)
            dataset_dict["unique_id_1"].append(str(f_id))
            dataset_dict["audio_name"].append(audio_name)
            dataset_dict["unique_id_2"].append(str(audio_name))

    ds = datasets.Dataset.from_dict(dataset_dict)
    ds = ds.sort(["unique_id_1", "unique_id_2"])
    ds = ds.cast_column("audio", datasets.Audio(sampling_rate=SAMPLING_RATE))
    ds = ds.add_column("dataset_index", list(range(len(ds))))
    ds = datasets.DatasetDict({"train": ds})
    ds.save_to_disk(DATA_PATH / DATASET_NAME)

    # delete raw data
    for language in URL_LINKS.keys():
        shutil.rmtree(data_dir / language)


if __name__ == "__main__":
    download_dataset()
