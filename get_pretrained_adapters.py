import argparse
import os

import torch

from src.model.wrapped_llms.qwen3 import Qwen3AudioWrappedForCausalLM
from src.utils import ROOT_PATH


def extract_weights(checkpoints, save_dir):
    save_dir = (ROOT_PATH / save_dir).resolve()
    save_dir.mkdir(exist_ok=True, parents=True)

    for encoder_name, checkpoint_path in checkpoints.items():
        if not os.path.exists(checkpoint_path):
            print(f"{checkpoint_path} not found for {encoder_name}. Skipping...")
            continue
        print(f"Using weights from {checkpoint_path} for {encoder_name}")
        model = Qwen3AudioWrappedForCausalLM.from_pretrained(checkpoint_path)
        adapter_state_dict = (
            model.get_audio_adapter().adapters[encoder_name].state_dict()
        )
        weights_save_path = save_dir / f"{encoder_name}.pth"
        torch.save(adapter_state_dict, weights_save_path)
        print(f"{encoder_name} weights saved at {weights_save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Extract weights from single-encoder pretrained models"
    )
    parser.add_argument(
        "--whisper",
        default="saved/Qwen3-4B-Thinking-2507/content_whisper_batch/checkpoint-34960",
        # default="saved/Qwen3-4B-Thinking-2507/alldataset_whisper_batch/checkpoint-172380",
        type=str,
        help="Path to Whisper weights.",
    )
    parser.add_argument(
        "--w2vbert",
        default="saved/Qwen3-4B-Thinking-2507/speech_w2vbert_layer/checkpoint-31376",
        type=str,
        help="Path to W2VBERT2 weights.",
    )
    parser.add_argument(
        "--sslam",
        default="saved/Qwen3-4B-Thinking-2507/audio_sslam_layer/checkpoint-27010",
        type=str,
        help="Path to SSLAM weights.",
    )
    parser.add_argument(
        "--muq",
        default="saved/Qwen3-4B-Thinking-2507/music_muq/checkpoint-25522",
        type=str,
        help="Path to MuQ weights.",
    )
    parser.add_argument(
        "--save-dir",
        default="data/pretrained_weights",
        type=str,
        help="Path where to save weights.",
    )
    args = parser.parse_args()

    checkpoints = {
        "whisper": args.whisper,
        "w2vbert": args.w2vbert,
        "sslam": args.sslam,
        "muq": args.muq,
    }
    extract_weights(checkpoints, args.save_dir)
