import json
import os
import shutil
from collections import defaultdict

import datasets
import torch
import torchaudio
from tqdm.auto import tqdm

from utils import DATA_PATH, SAMPLING_RATE, cli_download, get_audio_description

# https://arxiv.org/abs/2207.01063
DATASET_NAME = "dailytalk"
URL_LINK = "https://drive.google.com/drive/folders/1WRt-EprWs-2rmYxoWYT9_13omlhDHcaL"
TRAIN_URL_LINK = "https://raw.githubusercontent.com/keonlee9420/DailyTalk/881eb3c6c66ef0ed466895a6a5ef4d9c511bc2af/preprocessed_data/DailyTalk/train_frame.txt"

MAX_DURATION = 25  # 25 seconds

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


def add_elem_to_dataset(
    dataset_dict,
    audio_path,
    dialogue_id,
    joint_f_id,
    joint_text_list,
    joint_duration_list,
    joint_act_list,
    joint_emotion_list,
    speaker2order,
):
    elem_info = {}

    processed_joint_duration_list = []
    min_start_time = -1
    all_speakers = set()
    for index in range(len(joint_duration_list)):
        start_time, end_time, speaker_id = joint_duration_list[index]
        emotion = joint_emotion_list[index]
        speech_act = joint_act_list[index]
        if min_start_time == -1:  # the first time
            min_start_time = start_time
        start_time -= min_start_time
        end_time -= min_start_time
        order = speaker2order[speaker_id]
        speaker_info = (
            f"the {order} speaker, emotion: {emotion}, speech act: {speech_act}"
        )
        processed_joint_duration_list.append((start_time, end_time, speaker_info))
        all_speakers.add(speaker_id)

    elem_info["the number of speakers"] = len(all_speakers)

    elem_info["duration"] = processed_joint_duration_list
    elem_info["text"] = joint_text_list

    audio_description = get_audio_description(text_type="speech", **elem_info)

    dataset_dict["audio_description"].append(audio_description)

    elem_info["duration"] = elem_info["duration"][-1][1]
    elem_info["text"] = ";".join(elem_info["text"])

    for k, v in elem_info.items():
        dataset_dict[k].append(v)
    dataset_dict["audio"].append(audio_path)
    dataset_dict["dialogue id"].append(dialogue_id)
    dataset_dict["unique_id_1"].append(str(dialogue_id))
    dataset_dict["joint_f_id"].append(joint_f_id)
    dataset_dict["unique_id_2"].append(str(joint_f_id))


def download_dataset():
    dataset_dict = defaultdict(list)

    data_dir = DATA_PATH / DATASET_NAME
    data_dir.mkdir(exist_ok=True, parents=True)

    arc_path = data_dir / "dailytalk.zip"
    cli_download(URL_LINK, output=str(arc_path.parent), method="gdown_folder")
    shutil.unpack_archive(arc_path, data_dir)
    os.remove(arc_path)

    train_txt_path = data_dir / "dailytalk" / "train.txt"
    cli_download(TRAIN_URL_LINK, output=str(train_txt_path))

    train_dialogues = set()
    train_file_id_to_clean_text = {}
    with train_txt_path.open("r") as f:
        for line in f.readlines():
            line_split = line.split("|")
            file_id = line_split[0]
            dialogue_id = file_id.split("d")[1]
            train_dialogues.add(dialogue_id)

            clean_text = line_split[-2]
            train_file_id_to_clean_text[file_id] = clean_text

    metadata_path = data_dir / "dailytalk" / "metadata.json"
    full_data_path = data_dir / "dailytalk" / "data"
    with metadata_path.open("r") as meta:
        metadata = json.load(meta)

    tmp_dir = full_data_path / "tmp"
    tmp_dir.mkdir(exist_ok=True, parents=True)

    for dialogue_id in tqdm(metadata.keys()):
        if dialogue_id not in train_dialogues:
            continue

        utterance_ids = sorted(metadata[dialogue_id].keys())

        total_duration = 0
        joint_audio_list = []
        joint_text_list = []
        joint_duration_list = []
        joint_utterance_id_list = []
        joint_emotion_list = []
        joint_act_list = []
        wav_id = 0

        # order within the split
        speaker2order = {}
        order = 1

        for utterance_id in utterance_ids:
            info = metadata[dialogue_id][utterance_id]

            speaker_id = info["speaker"]
            file_id = f"{utterance_id}_{speaker_id}_d{dialogue_id}"
            audio_path = str(full_data_path / dialogue_id / f"{file_id}.wav")

            if speaker_id not in speaker2order.keys():
                speaker2order[speaker_id] = ORDER2TEXT[order]
                order += 1

            text = train_file_id_to_clean_text[file_id]

            audio, sr = torchaudio.load(audio_path)
            if audio.shape[0] != 1:
                audio = audio.mean(dim=0).unsqueeze(0)
            if sr != SAMPLING_RATE:
                audio = torchaudio.functional.resample(audio, sr, SAMPLING_RATE)
            joint_audio_list.append(audio)

            speech_act = info["act"]
            emotion = info["emotion"]
            if emotion == "no emotion":
                emotion = "neutral"

            start_time = total_duration
            end_time = total_duration + audio.shape[-1] / SAMPLING_RATE

            joint_duration_list.append((start_time, end_time, speaker_id))
            joint_utterance_id_list.append(str(utterance_id))
            total_duration = end_time

            joint_emotion_list.append(emotion)
            joint_act_list.append(speech_act)
            joint_text_list.append(text.strip())

            if total_duration >= MAX_DURATION:
                joint_audio = torch.concat(joint_audio_list, dim=-1)
                joint_f_id = f"{dialogue_id}_{wav_id}_" + "_".join(
                    joint_utterance_id_list
                )
                audio_path = str(tmp_dir / f"{joint_f_id}.wav")
                torchaudio.save(audio_path, joint_audio, sample_rate=SAMPLING_RATE)

                add_elem_to_dataset(
                    dataset_dict,
                    audio_path=audio_path,
                    dialogue_id=dialogue_id,
                    joint_f_id=joint_f_id,
                    joint_text_list=joint_text_list,
                    joint_duration_list=joint_duration_list,
                    joint_act_list=joint_act_list,
                    joint_emotion_list=joint_emotion_list,
                    speaker2order=speaker2order,
                )

                total_duration = 0
                joint_audio_list = []
                joint_text_list = []
                joint_duration_list = []
                joint_utterance_id_list = []
                joint_emotion_list = []
                joint_act_list = []
                wav_id += 1

                # order within the split
                speaker2order = {}
                order = 1
        if total_duration != 0:
            joint_audio = torch.concat(joint_audio_list, dim=-1)
            joint_f_id = f"{dialogue_id}_{wav_id}_" + "_".join(joint_utterance_id_list)
            audio_path = str(tmp_dir / f"{joint_f_id}.wav")
            torchaudio.save(audio_path, joint_audio, sample_rate=SAMPLING_RATE)

            add_elem_to_dataset(
                dataset_dict,
                audio_path=audio_path,
                dialogue_id=dialogue_id,
                joint_f_id=joint_f_id,
                joint_text_list=joint_text_list,
                joint_duration_list=joint_duration_list,
                joint_act_list=joint_act_list,
                joint_emotion_list=joint_emotion_list,
                speaker2order=speaker2order,
            )

    ds = datasets.Dataset.from_dict(dataset_dict)
    ds = ds.sort(["unique_id_1", "unique_id_2"])
    ds = ds.cast_column("audio", datasets.Audio(sampling_rate=SAMPLING_RATE))
    ds = ds.add_column("dataset_index", list(range(len(ds))))
    ds = datasets.DatasetDict({"train": ds})
    ds.save_to_disk(DATA_PATH / DATASET_NAME)

    # delete raw data
    shutil.rmtree(data_dir / "dailytalk")


if __name__ == "__main__":
    download_dataset()
