import os
import shutil
from collections import defaultdict

import datasets
import torch
import torchaudio
from tqdm.auto import tqdm

from utils import DATA_PATH, SAMPLING_RATE, get_audio_description

# https://groups.inf.ed.ac.uk/ami/corpus/
# https://link.springer.com/chapter/10.1007/11677482_3
DATASET_NAME = "amicorpus"

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

    ds = datasets.load_dataset(
        "edinburghcstr/ami",
        split="train",
        revision="8dfb8d51a914b60af751fa9b561878bbba07160d",
    )
    # we want head microphones
    ds = ds.filter(lambda x: x["microphone_id"] != "SDM1")
    ds = ds.sort(["meeting_id", "begin_time", "end_time"])

    tmp_dir = data_dir / "tmp"
    tmp_dir.mkdir(exist_ok=True, parents=True)

    cur_meet_id = -1
    total_duration = 0
    joint_audio_list = []
    joint_text_list = []
    joint_duration_list = []
    joint_speaker_id_list = []

    # order within the split
    speaker2order = {}
    speaker2gender = {}
    order = 1
    min_start_time = -1

    wav_id = 0
    for elem in tqdm(ds, desc="Processing corpus"):
        meet_id = elem["meeting_id"]
        if cur_meet_id != meet_id:
            if total_duration != 0:
                joint_audio = get_joint_audio(joint_audio_list, joint_duration_list)
                joint_f_id = f"{cur_meet_id}_{wav_id}_" + "_".join(
                    joint_speaker_id_list
                )
                audio_path = str(tmp_dir / f"{joint_f_id}.wav")
                torchaudio.save(audio_path, joint_audio, sample_rate=SAMPLING_RATE)

                add_elem_to_dataset(
                    dataset_dict,
                    audio_path=audio_path,
                    meet_id=cur_meet_id,
                    joint_f_id=joint_f_id,
                    joint_text_list=joint_text_list,
                    joint_duration_list=joint_duration_list,
                    speaker2gender=speaker2gender,
                    speaker2order=speaker2order,
                )

                total_duration = 0
                joint_audio_list = []
                joint_text_list = []
                joint_duration_list = []
                joint_speaker_id_list = []

                # order within the split
                speaker2order = {}
                speaker2gender = {}
                order = 1
                min_start_time = -1
                wav_id += 1

            cur_meet_id = meet_id
            # wav_id is within the meeting
            wav_id = 0

        speaker_id = elem["speaker_id"]
        start_time = elem["begin_time"]
        end_time = elem["end_time"]
        text = elem["text"].lower().strip()

        if speaker_id not in speaker2order.keys():
            speaker2order[speaker_id] = ORDER2TEXT[order]
            gender = "Male" if speaker_id[0] == "M" else "Female"
            speaker2gender[speaker_id] = gender
            order += 1
        if min_start_time == -1:  # the first time
            min_start_time = start_time

        audio_data = elem["audio"].get_all_samples()
        audio = audio_data.data
        sr = audio_data.sample_rate
        if audio.shape[0] != 1:
            audio = audio.mean(dim=0).unsqueeze(0)
        if sr != SAMPLING_RATE:
            audio = torchaudio.functional.resample(audio, sr, SAMPLING_RATE)
        joint_audio_list.append(audio)
        joint_text_list.append(text)
        joint_duration_list.append((start_time, end_time, speaker_id))
        joint_speaker_id_list.append(speaker_id[3:])
        # speech may overlap, so we use end time
        total_duration = end_time - min_start_time

        if total_duration >= MAX_DURATION:
            joint_audio = get_joint_audio(joint_audio_list, joint_duration_list)
            joint_f_id = f"{cur_meet_id}_{wav_id}_" + "_".join(joint_speaker_id_list)
            audio_path = str(tmp_dir / f"{joint_f_id}.wav")
            torchaudio.save(audio_path, joint_audio, sample_rate=SAMPLING_RATE)

            add_elem_to_dataset(
                dataset_dict,
                audio_path=audio_path,
                meet_id=cur_meet_id,
                joint_f_id=joint_f_id,
                joint_text_list=joint_text_list,
                joint_duration_list=joint_duration_list,
                speaker2gender=speaker2gender,
                speaker2order=speaker2order,
            )

            total_duration = 0
            joint_audio_list = []
            joint_text_list = []
            joint_duration_list = []
            joint_speaker_id_list = []

            # order within the split
            speaker2order = {}
            speaker2gender = {}
            order = 1
            min_start_time = -1
            wav_id += 1

    # save the last wav
    if total_duration != 0:
        joint_audio = get_joint_audio(joint_audio_list, joint_duration_list)
        joint_f_id = f"{cur_meet_id}_{wav_id}_" + "_".join(joint_speaker_id_list)
        audio_path = str(tmp_dir / f"{joint_f_id}.wav")
        torchaudio.save(audio_path, joint_audio, sample_rate=SAMPLING_RATE)

        add_elem_to_dataset(
            dataset_dict,
            audio_path=audio_path,
            meet_id=cur_meet_id,
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
    shutil.rmtree(tmp_dir)


if __name__ == "__main__":
    download_dataset()
