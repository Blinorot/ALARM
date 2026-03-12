import json
import math
import os
import shutil
import subprocess
from collections import defaultdict

import datasets
import textgrid
import torch
import torchaudio
from tqdm.auto import tqdm

from utils import DATA_PATH, SAMPLING_RATE, cli_download, get_audio_description

# https://arxiv.org/abs/2110.07393
# https://www.openslr.org/119/
# we use near variant
DATASET_NAME = "alimeeting"
URL_LINK = "https://speech-lab-share-data.oss-cn-shanghai.aliyuncs.com/AliMeeting/openlr/Train_Ali_near.tar.gz"

MAX_DURATION = 40  # 40 seconds

ORDER2TEXT = {
    1: "first",
    2: "second",
    3: "third",
    4: "fourth",
    5: "fifth",
    6: "sixth",
    7: "seventh",
    8: "eighth",
    9: "ninth",
    10: "tenth",
}


def get_meet2spk(data_dir):
    meet2spk = defaultdict(list)
    for filename in os.listdir(data_dir / "textgrid_dir"):
        room_id, meeting_id, _, _ = filename.split("_")
        f_id = filename.split(".")[0]
        meet2spk[f"{room_id}_{meeting_id}"].append(f_id)
    return meet2spk


def get_joint_audio(joint_audio_list, joint_duration_list):
    """
    Combine overlapping audio sequences based on start/end time.
    """
    time_audio_list = []
    total_time = 0
    min_start_time = -1
    for time_info, audio in zip(joint_duration_list, joint_audio_list):
        start_time, end_time, _ = time_info

        if min_start_time == -1:  # the first time
            min_start_time = start_time
        start_time = start_time - min_start_time
        end_time = end_time - min_start_time
        start_time = round(start_time * SAMPLING_RATE)

        # to avoid +-1 issues
        end_time = start_time + audio.shape[-1]

        time_audio_list.append((start_time, end_time, audio))
        total_time = max(total_time, end_time)

    joint_audio = torch.zeros((1, total_time), dtype=torch.float32)
    for start_time, end_time, audio in time_audio_list:
        joint_audio[..., start_time:end_time] = audio[0].clone()
    return joint_audio


def add_elem_to_dataset(
    dataset_dict,
    audio_path,
    meet_id,
    joint_f_id,
    joint_text_list,
    joint_duration_list,
    speaker2gender,
    speaker2order,
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
        elem_info[f"the {order} speaker's gender"] = gender

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

    for k, v in elem_info.items():
        dataset_dict[k].append(v)
    dataset_dict["audio"].append(audio_path)
    dataset_dict["meeting_id"].append(meet_id)
    dataset_dict["unique_id_1"].append(str(meet_id))
    dataset_dict["joint_f_id"].append(joint_f_id)
    dataset_dict["unique_id_2"].append(str(joint_f_id))


def download_dataset():
    dataset_dict = defaultdict(list)

    data_dir = DATA_PATH / DATASET_NAME
    data_dir.mkdir(exist_ok=True, parents=True)

    arc_path = data_dir / "ali.tar.gz"
    cli_download(URL_LINK, output=str(arc_path), method="wget")
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

    full_data_dir = data_dir / "Train_Ali_near"
    meet2spk = get_meet2spk(full_data_dir)

    tmp_dir = data_dir / "tmp"
    tmp_dir.mkdir(exist_ok=True, parents=True)

    for meet_id, f_ids in tqdm(meet2spk.items(), desc="Processing meetings"):
        speaker2gender = {}
        speaker2fid = {}

        time_and_text = []
        for f_id in f_ids:
            _, _, gender, speaker_id = f_id.split("_")
            if gender == "M":
                gender = "Male"
            else:
                gender = "Female"
            assert speaker_id not in speaker2gender.keys(), "Duplicate"
            speaker2gender[speaker_id] = gender
            speaker2fid[speaker_id] = f_id
            f_path = full_data_dir / "textgrid_dir" / f"{f_id}.TextGrid"
            time_text = textgrid.TextGrid.fromFile(f_path)
            for elem in time_text[0]:
                text = elem.mark.strip()
                start_time = elem.minTime
                end_time = elem.maxTime
                time_and_text.append((start_time, end_time, text, speaker_id))

        time_and_text = sorted(time_and_text, key=lambda x: x[0])
        total_duration = 0
        joint_audio_list = []
        joint_text_list = []
        joint_duration_list = []
        joint_speaker_id_list = []
        wav_id = 0
        min_start_time = -1

        # order within the split
        speaker2order = {}
        order = 1
        for start_time, end_time, text, speaker_id in tqdm(time_and_text, leave=False):
            if speaker_id not in speaker2order.keys():
                speaker2order[speaker_id] = ORDER2TEXT[order]
                order += 1
            if min_start_time == -1:  # the first time
                min_start_time = start_time
            f_id = speaker2fid[speaker_id]
            audio_path = full_data_dir / "audio_dir" / f"{f_id}.wav"
            audio, sr = torchaudio.load(audio_path)
            audio_start_time = round(start_time * sr)
            audio_end_time = round(end_time * sr)
            audio = audio[..., audio_start_time:audio_end_time]
            if audio.shape[0] != 1:
                audio = audio.mean(dim=0).unsqueeze(0)
            if sr != SAMPLING_RATE:
                audio = torchaudio.functional.resample(audio, sr, SAMPLING_RATE)
            joint_audio_list.append(audio)
            joint_text_list.append(text)
            joint_duration_list.append((start_time, end_time, speaker_id))
            joint_speaker_id_list.append(speaker_id[len("SPK") :])
            # speech may overlap, so we use end time
            total_duration = end_time - min_start_time

            if total_duration >= MAX_DURATION:
                joint_audio = get_joint_audio(joint_audio_list, joint_duration_list)
                joint_f_id = f"{meet_id}_{wav_id}_" + "_".join(joint_speaker_id_list)
                audio_path = str(tmp_dir / f"{joint_f_id}.wav")
                torchaudio.save(audio_path, joint_audio, sample_rate=SAMPLING_RATE)

                add_elem_to_dataset(
                    dataset_dict,
                    audio_path=audio_path,
                    meet_id=meet_id,
                    joint_f_id=joint_f_id,
                    joint_text_list=joint_text_list,
                    joint_duration_list=joint_duration_list,
                    speaker2gender=speaker2gender,
                    speaker2order=speaker2order,
                )

                total_duration = 0
                min_start_time = -1
                order = 1
                speaker2order = {}
                joint_audio_list = []
                joint_text_list = []
                joint_duration_list = []
                joint_speaker_id_list = []
                wav_id += 1

        # save the last wav
        if total_duration != 0:
            joint_audio = get_joint_audio(joint_audio_list, joint_duration_list)
            joint_f_id = f"{meet_id}_{wav_id}_" + "_".join(joint_speaker_id_list)
            audio_path = str(tmp_dir / f"{joint_f_id}.wav")
            torchaudio.save(audio_path, joint_audio, sample_rate=SAMPLING_RATE)

            add_elem_to_dataset(
                dataset_dict,
                audio_path=audio_path,
                meet_id=meet_id,
                joint_f_id=joint_f_id,
                joint_text_list=joint_text_list,
                joint_duration_list=joint_duration_list,
                speaker2gender=speaker2gender,
                speaker2order=speaker2order,
            )

    ds = datasets.Dataset.from_dict(dataset_dict)
    ds = ds.sort(["unique_id_1", "unique_id_2"])
    ds = ds.cast_column("audio", datasets.Audio(sampling_rate=SAMPLING_RATE))
    ds = ds.add_column("dataset_index", list(range(len(ds))))
    ds = datasets.DatasetDict({"train": ds})
    ds.save_to_disk(DATA_PATH / DATASET_NAME)

    # delete raw data
    shutil.rmtree(full_data_dir)
    shutil.rmtree(tmp_dir)


if __name__ == "__main__":
    download_dataset()
