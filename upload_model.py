import argparse

from transformers import AutoTokenizer

from src.model.wrapped_llms.qwen3 import (
    Qwen3AudioWrappedConfig,
    Qwen3AudioWrappedForCausalLM,
)
from src.utils import ROOT_PATH


def upload_model(args):
    tokenizer = AutoTokenizer.from_pretrained(args.llm)

    model = Qwen3AudioWrappedForCausalLM.from_pretrained(args.checkpoint)
    model.push_to_hub(f"{args.username}/{args.repo_name}", private=not args.public)
    tokenizer.push_to_hub(
        f"{args.username}/{args.repo_name}",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Upload model to HuggingFace")

    parser.add_argument(
        "--llm",
        default="Qwen/Qwen3-4B-Thinking-2507",
        type=str,
        help="Backbone LLM name (Default: Qwen/Qwen3-4B-Thinking-2507)",
    )
    parser.add_argument(
        "--checkpoint",
        default="saved/Qwen3-4B-Thinking-2507/content_whisper_batch/checkpoint-34960",
        type=str,
        help="Checkpoint weights (Default: saved/Qwen3-4B-Thinking-2507/content_whisper_batch/checkpoint-34960)",
    )

    parser.add_argument(
        "--username",
        default="Blinorot",
        type=str,
        help="HF Username",
    )
    parser.add_argument(
        "--repo-name",
        default="AL-Whisper-Instruct-R",
        type=str,
        help="HF repo name",
    )
    parser.add_argument(
        "--public",
        dest="public",
        action="store_true",
        help="Make the model public",
    )
    parser.set_defaults(public=False)

    args = parser.parse_args()
    upload_model(args)
