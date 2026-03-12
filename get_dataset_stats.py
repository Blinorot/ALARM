import argparse

from hydra.utils import instantiate
from omegaconf import OmegaConf
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from src.dataset import HFDataset
from src.utils import ROOT_PATH


def get_dataset_stats(data_dir):
    datasets = ["content_4b", "speech_4b", "audio_4b", "music_4b"]
    # placeholder
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B-Thinking-2507")
    for dataset_name in datasets:
        config_path = ROOT_PATH / "src" / "configs" / "dataset" / f"{dataset_name}.yaml"
        config = OmegaConf.load(config_path)
        dataset = instantiate(
            config.train,
            tokenizer=tokenizer,
            feature_extractor=None,
            data_dir=data_dir,
        )
        length = len(dataset.dataset)
        audio_length = 0
        i = 0
        for elem in tqdm(dataset.dataset):
            audio_data = elem["audio"].get_all_samples()
            duration = audio_data.data.shape[-1] / audio_data.sample_rate
            audio_length += duration
            i += 1

        audio_hours = round(audio_length / 60 / 60, 3)
        print(f"Dataset: {dataset_name}, {length} elems, {audio_hours} hours")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Calculate statistics for the dataset")
    parser.add_argument(
        "--data-dir",
        default="data",
        type=str,
        help="Name of the data dir.",
    )
    args = parser.parse_args()
    get_dataset_stats(args.data_dir)
