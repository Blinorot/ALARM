import argparse
import os
import shutil
import subprocess
from pathlib import Path

import datasets
from huggingface_hub import list_repo_files, snapshot_download
from tqdm import tqdm

from utils import DATA_PATH, SAMPLING_RATE

# https://arxiv.org/abs/2409.06666
DATASET_NAME = "instructs2s"


def download_dataset(repo_id):
    snapshot_path = snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        resume_download=True,
        revision="004320105d5ca5246e8eacc4cbf53f7f100278e8",
    )
    local_root = DATA_PATH / DATASET_NAME
    local_root.mkdir(exist_ok=True, parents=True)

    parts = [Path(snapshot_path) / f"en_part_{i:02d}" for i in range(33)]
    tar_path = local_root / "en_combined.tar.gz"
    print("Combining archives...")
    if not tar_path.exists():
        with (local_root / "en_combined.tar.gz").open("wb") as fcomb:
            for part in parts:
                with part.open("rb") as fpart:
                    fcomb.write(fpart.read())
    print("Extracting tar.gz...")
    subprocess.run(
        [
            "tar",
            "--use-compress-program=pigz -p 32",
            "-xf",
            str(tar_path),
            "-C",
            str(local_root),
        ],
        check=True,
    )
    os.remove(str(local_root / "en_combined.tar.gz"))


def convert_dataset(args):
    DATA_PATH.mkdir(exist_ok=True, parents=True)

    repo_id = "ICTNLP/InstructS2S-200K"
    local_root = DATA_PATH / DATASET_NAME
    wav_path = local_root / "en" / "wav"
    if not wav_path.exists():
        download_dataset(repo_id)

    dataset = datasets.load_dataset(
        repo_id, revision="004320105d5ca5246e8eacc4cbf53f7f100278e8"
    )["train"]
    if args.limit > 0:
        dataset = dataset.select(range(args.limit))
    new_dataset_list = []
    for elem in tqdm(dataset):
        conversation = elem["conversation"]
        id = elem["id"]
        utterance_id = -1
        context = ""
        for utterance in conversation:
            utterance_id += 1
            if utterance["from"] != "human":
                context += f"Assistant: {utterance['text']}\n"
                continue
            question = utterance["text"]
            speech_path = utterance["speech"].split("/")
            speech_fname = speech_path[-1]
            speech_dir = speech_path[-2]
            speech_path = str(wav_path / speech_dir / speech_fname)

            new_dataset_list.append(
                {
                    "question": question,
                    "context": context,
                    "audio": speech_path,
                    "id": id,
                    "utterance_id": utterance_id,
                    "unique_id_1": str(id),
                    "unique_id_2": str(utterance_id),
                }
            )

            context += f"User: {utterance['text']}\n"

    new_dataset = datasets.Dataset.from_list(new_dataset_list)
    new_dataset = new_dataset.sort(["unique_id_1", "unique_id_2"])
    new_dataset = new_dataset.cast_column(
        "audio", datasets.Audio(sampling_rate=SAMPLING_RATE)
    )
    new_dataset = new_dataset.add_column("dataset_index", list(range(len(new_dataset))))

    new_dataset = datasets.DatasetDict({"train": new_dataset})
    new_dataset.save_to_disk(DATA_PATH / DATASET_NAME)

    shutil.rmtree(DATA_PATH / DATASET_NAME / "en")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Convert InstructS2S-200K to the QA format")
    parser.add_argument(
        "--limit",
        default=-1,
        type=int,
        help="Limit dataset to this number of samples (Default: -1)",
    )
    args = parser.parse_args()
    convert_dataset(args)
