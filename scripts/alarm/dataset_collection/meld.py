import os
import shutil
from collections import defaultdict

import datasets
import pandas as pd
import torch
import torchaudio
from torchcodec.decoders import AudioDecoder
from tqdm.auto import tqdm

from utils import DATA_PATH, SAMPLING_RATE, cli_download, get_audio_description

# https://arxiv.org/abs/1810.02508
DATASET_NAME = "meld"
URL_LINK = "https://huggingface.co/datasets/declare-lab/MELD/resolve/9abc51ee7903424ffb971297608aa6d3d0de3bfa/MELD.Raw.tar.gz?download=true"

MAX_DURATION = 20  # 20 seconds

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
    joint_sentiment_list,
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
        sentiment = joint_sentiment_list[index]
        if min_start_time == -1:  # the first time
            min_start_time = start_time
        start_time -= min_start_time
        end_time -= min_start_time
        order = speaker2order[speaker_id]
        speaker_info = (
            f"the {order} speaker, emotion: {emotion}, sentiment: {sentiment}"
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

    arc_path = data_dir / "MELD.Raw.tar.gz"
    cli_download(URL_LINK, output=str(arc_path))
    shutil.unpack_archive(arc_path, data_dir)
    os.remove(str(arc_path))
    train_dir = data_dir / "MELD.Raw"
    train_arc_path = data_dir / "MELD.Raw" / "train.tar.gz"
    shutil.unpack_archive(train_arc_path, train_dir)

    tmp_dir = train_dir / "tmp"
    tmp_dir.mkdir(exist_ok=True, parents=True)

    metadata = pd.read_csv(train_dir / "train_sent_emo.csv")
    dialogue_ids = metadata["Dialogue_ID"].unique()
    for dialogue_id in tqdm(dialogue_ids):
        dialogue_meta = metadata.loc[metadata["Dialogue_ID"] == dialogue_id]
        dialogue_meta = dialogue_meta.sort_values(by="Utterance_ID", ascending=True)

        total_duration = 0
        joint_audio_list = []
        joint_text_list = []
        joint_duration_list = []
        joint_utterance_id_list = []
        joint_emotion_list = []
        joint_sentiment_list = []
        wav_id = 0

        # order within the split
        speaker2order = {}
        order = 1

        for _, row in tqdm(
            dialogue_meta.iterrows(), total=dialogue_meta.shape[0], leave=False
        ):
            utterance_id = row["Utterance_ID"]
            file_id = f"dia{dialogue_id}_utt{utterance_id}.mp4"
            video_path = str(train_dir / "train_splits" / file_id)

            try:
                audio_data = AudioDecoder(video_path).get_all_samples()
            except RuntimeError:
                print(f"Broken file {file_id}, skipping")
                if len(joint_audio_list) != 0:
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
                        joint_sentiment_list=joint_sentiment_list,
                        joint_emotion_list=joint_emotion_list,
                        speaker2order=speaker2order,
                    )

                    total_duration = 0
                    joint_audio_list = []
                    joint_text_list = []
                    joint_duration_list = []
                    joint_utterance_id_list = []
                    joint_emotion_list = []
                    joint_sentiment_list = []
                    wav_id += 1

                    # order within the split
                    speaker2order = {}
                    order = 1
                continue

            speaker_id = row["Speaker"]
            if speaker_id not in speaker2order.keys():
                speaker2order[speaker_id] = ORDER2TEXT[order]
                order += 1

            audio = audio_data.data
            sr = audio_data.sample_rate
            if audio.shape[0] != 1:
                audio = audio.mean(dim=0).unsqueeze(0)
            if sr != SAMPLING_RATE:
                audio = torchaudio.functional.resample(audio, sr, SAMPLING_RATE)
            joint_audio_list.append(audio)

            start_time = total_duration
            end_time = total_duration + audio.shape[-1] / SAMPLING_RATE

            joint_duration_list.append((start_time, end_time, speaker_id))
            joint_utterance_id_list.append(str(utterance_id))
            total_duration = end_time

            joint_emotion_list.append(row["Emotion"])
            joint_sentiment_list.append(row["Sentiment"])

            text = row["Utterance"]
            # ' looks like \x92 in string
            text = text.replace("\x92", "'")
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
                    joint_sentiment_list=joint_sentiment_list,
                    joint_emotion_list=joint_emotion_list,
                    speaker2order=speaker2order,
                )

                total_duration = 0
                joint_audio_list = []
                joint_text_list = []
                joint_duration_list = []
                joint_utterance_id_list = []
                joint_emotion_list = []
                joint_sentiment_list = []
                wav_id += 1

                # order within the split
                speaker2order = {}
                order = 1

        # save the last wav
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
                joint_sentiment_list=joint_sentiment_list,
                joint_emotion_list=joint_emotion_list,
                speaker2order=speaker2order,
            )

    ds = datasets.Dataset.from_dict(dataset_dict)
    ds = ds.sort(["unique_id_1", "unique_id_2"])
    ds = ds.cast_column("audio", datasets.Audio(sampling_rate=SAMPLING_RATE))
    ds = ds.add_column("dataset_index", list(range(len(ds))))
    ds = datasets.DatasetDict({"train": ds})
    ds.save_to_disk(DATA_PATH / DATASET_NAME)

    # remove raw data
    shutil.rmtree(data_dir / "MELD.Raw")


if __name__ == "__main__":
    download_dataset()
