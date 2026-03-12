import argparse
import json

import datasets
from tqdm.auto import tqdm

from utils import DATA_PATH, RESPONSE_PATH, get_response_name, load_merged_dataset


def create_answer_json(args):
    actual_model_name = args.model_name.split("/")[-1]
    checkpoint_name = args.checkpoint_name
    if "," in checkpoint_name:
        checkpoint_name = checkpoint_name.split(",")
    ds = load_merged_dataset(
        args.dataset_name,
        actual_model_name,
        checkpoint_name,
        args.max_tokens,
        args.max_thinking_tokens,
        args.seed,
    )
    ds = ds.select_columns(["dataset_index", "llm_answer"])
    # to be sure
    ds = ds.sort("dataset_index")

    response_name = get_response_name(
        args.dataset_name,
        checkpoint_name,
        args.max_tokens,
        args.max_thinking_tokens,
        args.seed,
    )
    response_path = RESPONSE_PATH / actual_model_name / response_name
    response_metadata_dir = response_path / "metadata"
    response_metadata_dir.mkdir(exist_ok=True, parents=True)

    metadata_dir = DATA_PATH / args.dataset_name / "metadata"
    for split in ds:
        ds_split = ds[split]
        metadata_path = metadata_dir / f"{split}_metadata.json"
        if not metadata_path.exists():
            print(f"Could not found {metadata_path}, skipping...")
            continue
        with metadata_path.open("r") as f:
            metadata = json.load(f)
        new_metadata = []
        for index, elem in tqdm(enumerate(metadata), desc=f"Adding {split}"):
            elem["model_prediction"] = ds_split[index]["llm_answer"]
            new_metadata.append(elem)
        with (response_metadata_dir / f"{split}_metadata.json").open("w") as f:
            json.dump(new_metadata, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Add LLM responses to the instruction dataset")
    parser.add_argument(
        "--dataset-name",
        default="mmar",
        type=str,
        help="Dataset name (Default: mmar)",
    )
    parser.add_argument(
        "--model-name",
        default="Qwen/Qwen3-4B-Thinking-2507",
        type=str,
        help="LLM name (Default: Qwen/Qwen3-4B-Thinking-2507)",
    )
    parser.add_argument(
        "--checkpoint-name",
        default="saved/Qwen3-4B-Thinking-2507/bigdataset_whisper_downsample_conformer/checkpoint-30000",
        type=str,
        help="Checkpoint name. To pass a list, separate checkpoints by comma. (Default: saved/Qwen3-4B-Thinking-2507/bigdataset_whisper_downsample_conformer/checkpoint-30000)",
    )
    parser.add_argument(
        "--max-tokens",
        default=8192,
        type=int,
        help="Max tokens for generation (Default: 8192)",
    )
    parser.add_argument(
        "--max-thinking-tokens",
        default=1024,
        type=int,
        help="Max tokens for internal thinking (Default: 1024)",
    )
    parser.add_argument(
        "--seed",
        default=None,
        type=int,
        help="Random seed used for generation.",
    )
    args = parser.parse_args()
    create_answer_json(args)
