import argparse
import re
from pathlib import Path

import datasets
import torch
from torch.multiprocessing import set_start_method
from tqdm import tqdm

from utils import DATA_PATH, RESPONSE_PATH, load_merged_dataset

PROCESS_OBJECTS = {"asr_model": None, "asr_processor": None, "wer": None, "cer": None}

set_start_method("spawn")


def normalize_text(text: str):
    text = text.lower()
    text = re.sub(r"[^a-z ]", "", text)
    return text


def get_asr_model(rank):
    if PROCESS_OBJECTS["asr_model"] is not None:
        return (
            PROCESS_OBJECTS["asr_model"],
            PROCESS_OBJECTS["asr_processor"],
            PROCESS_OBJECTS["wer"],
            PROCESS_OBJECTS["cer"],
        )

    # lazy import because these models trigger cuda
    import evaluate
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    device = f"cuda:{rank}" if rank is not None else "cuda"

    whisper_name = "openai/whisper-large-v3"
    asr_processor = WhisperProcessor.from_pretrained(whisper_name)
    asr_model = WhisperForConditionalGeneration.from_pretrained(whisper_name)
    asr_model.to(device)
    asr_model.eval()
    PROCESS_OBJECTS["asr_model"] = asr_model
    PROCESS_OBJECTS["asr_processor"] = asr_processor

    wer = evaluate.load("wer")
    PROCESS_OBJECTS["wer"] = wer
    cer = evaluate.load("cer")
    PROCESS_OBJECTS["cer"] = cer

    return asr_model, asr_processor, wer, cer


@torch.no_grad()
def add_flag_to_the_dataset(batch, rank, threshold, threshold_type):
    asr_model, asr_processor, wer, cer = get_asr_model(rank)
    batch_size = len(batch["context"])
    device = f"cuda:{rank}" if rank is not None else "cuda"

    audio_list = []
    for j in range(batch_size):
        audio_data = batch["audio"][j].get_all_samples()
        audio = audio_data.data
        sr = audio_data.sample_rate
        audio_list.append(audio.mean(dim=0).numpy())

    forced_decoder_ids = asr_processor.get_decoder_prompt_ids(
        language="english", task="transcribe"
    )

    inputs = asr_processor(
        audio_list, sampling_rate=sr, return_tensors="pt", return_attention_mask=True
    )
    input_features = inputs.input_features.to(device)
    attention_masks = inputs.attention_mask.to(device)
    # generate token ids
    predicted_ids = asr_model.generate(
        input_features=input_features,
        attention_mask=attention_masks,
        forced_decoder_ids=forced_decoder_ids,
    )
    # decode token ids to text
    asr_questions = asr_processor.batch_decode(predicted_ids, skip_special_tokens=True)

    # chosen_question = batch["question"][j]
    # # fix to make it a question
    # if chosen_question[-1] == "?" and asr_question[-1] != "?":
    #     asr_question[-1] = "?"
    asr_questions = [elem.strip() for elem in asr_questions]

    score_metric = cer if threshold_type == "cer" else wer
    asr_match = []
    for j in range(batch_size):
        asr_question = [normalize_text(asr_questions[j])]
        ds_question = [normalize_text(batch["question"][j])]

        score = score_metric.compute(predictions=asr_question, references=ds_question)
        if score > threshold:
            asr_match.append(0)
        else:
            asr_match.append(1)

    return {"asr_match": asr_match, "asr_question": asr_questions}


def filter_dataset_with_asr(args):
    model_name = args.model_name.split("/")[-1]
    base_path = DATA_PATH / f"{args.dataset_name}_filtered"

    response_name = f"{args.dataset_name}_filtered_{model_name}_{args.max_tokens}"
    response_path = RESPONSE_PATH / model_name / response_name

    dataset = load_merged_dataset(args.dataset_name, model_name, args.max_tokens)
    if base_path.exists():
        print("Found the filtered version of the dataset. Filtering responses")
        # dataset was already filtered, we just need to take dataset_index
        filtered_dataset = datasets.load_from_disk(base_path)
        dataset_splits = {}
        for split in filtered_dataset:
            filtered_split = filtered_dataset[split]
            dataset_indexes = set(filtered_split["dataset_index"])
            dataset_split = dataset[split]
            dataset_split = dataset_split.filter(
                lambda x: x["dataset_index"] in dataset_indexes
            )
            dataset_splits[split] = dataset_split

        dataset = datasets.DatasetDict(dataset_splits)

        response_dataset = dataset.select_columns(
            [
                "dataset_index",
                "question",
                "context",
                "llm_answer_with_context",
                "llm_answer_no_context",
            ]
        )
        response_dataset.save_to_disk(response_path)
        return None

    if args.limit >= 0:
        dataset = datasets.DatasetDict(
            {
                split: dataset[split].select(range(args.limit))
                for split in dataset.keys()
            }
        )

    original_lengths = {k: len(dataset[k]) for k in dataset.keys()}
    fn_kwargs = {"threshold": args.threshold, "threshold_type": args.threshold_type}
    dataset = dataset.map(
        add_flag_to_the_dataset,
        batched=True,
        batch_size=args.batch_size,
        with_rank=True,
        num_proc=args.num_gpus,
        fn_kwargs=fn_kwargs,
        desc="Adding filter tag to the dataset...",
    )

    bad_dataset = dataset.filter(lambda x: x["asr_match"] == 0)
    for split in bad_dataset:
        for elem in bad_dataset[split]:
            print("Bad example:", split, elem["question"], elem["asr_question"])

    dataset = dataset.filter(lambda x: x["asr_match"] == 1)
    dataset = dataset.remove_columns(["asr_match", "asr_question"])
    print("Dataset original size:", original_lengths)
    print("Dataset filtered size:", {k: len(dataset[k]) for k in dataset.keys()})

    dataset = dataset.sort(["dataset_index"])

    response_dataset = dataset.select_columns(
        [
            "dataset_index",
            "question",
            "context",
            "llm_answer_with_context",
            "llm_answer_no_context",
        ]
    )
    base_dataset = dataset.remove_columns(
        ["llm_answer_with_context", "llm_answer_no_context"]
    )

    base_dataset.save_to_disk(base_path)
    response_dataset.save_to_disk(response_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Dataset Converter")
    parser.add_argument(
        "--model-name",
        default="Qwen/Qwen3-0.6B",
        type=str,
        help="LLM name (Default: Qwen/Qwen3-0.6B)",
    )
    parser.add_argument(
        "--dataset-name",
        default="heysquad_human",
        type=str,
        help="Dataset name (Default: heysquad_human)",
    )
    parser.add_argument(
        "--batch-size",
        default=64,
        type=int,
        help="Batch size for the conversion (Default: 64)",
    )
    parser.add_argument(
        "--max-tokens",
        default=512,
        type=int,
        help="Max tokens for generation (Default: 512)",
    )
    parser.add_argument(
        "--limit",
        default=-1,
        type=int,
        help="Limit dataset to this number of samples (Default: -1)",
    )
    parser.add_argument(
        "--threshold",
        default=0.5,
        type=float,
        help="Threshold for CER/WER (Default: 0.5)",
    )
    parser.add_argument(
        "--threshold_type",
        default="wer",
        type=str,
        help="cer or wer (Default: wer)",
    )
    parser.add_argument(
        "--num-gpus",
        default=1,
        type=int,
        help="Number of GPUs (Default: 1)",
    )
    args = parser.parse_args()
    filter_dataset_with_asr(args)
