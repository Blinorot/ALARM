import argparse
import gc
import os
import time
from pathlib import Path

import datasets
import torch
from combined_dataset import ALL_DATASETS, COLUMNS_TO_TAKE, DATA_PATH, ROOT_PATH
from create_prompts import PROMPT_PATH
from torch.multiprocessing import set_start_method
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.distributed import cleanup_dist_env_and_memory

set_start_method("spawn")

RESPONSE_PATH = ROOT_PATH / "data" / "datasets" / "responses"

PROCESS_OBJECTS = {
    "llm": None,
    "sample": None,
    "tokenizer": None,
}


def free_memory():
    if PROCESS_OBJECTS["llm"] is not None:
        PROCESS_OBJECTS["llm"].llm_engine.engine_core.shutdown()
        del PROCESS_OBJECTS["llm"]
        del PROCESS_OBJECTS["sample"]
        del PROCESS_OBJECTS["tokenizer"]
    PROCESS_OBJECTS["llm"] = None
    PROCESS_OBJECTS["sample"] = None
    PROCESS_OBJECTS["tokenizer"] = None
    cleanup_dist_env_and_memory()
    gc.collect()
    torch.cuda.empty_cache()
    time.sleep(5)


def get_model(model_name, cuda_devices, max_tokens, rank):
    if PROCESS_OBJECTS["llm"] is not None:
        return (
            PROCESS_OBJECTS["llm"],
            PROCESS_OBJECTS["sample"],
            PROCESS_OBJECTS["tokenizer"],
        )

    # cuda_devices == list of available devices,
    # e.g. cuda_devices = "[0, 1, 2, 3]"
    if rank is None:  # Only 1 proc
        rank = 0
    cuda_device = cuda_devices[rank]
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_device

    llm = LLM(
        model_name,
        enable_prefix_caching=True,
        max_model_len=8192,
        max_num_seqs=2048,
        max_num_batched_tokens=8192,
        gpu_memory_utilization=0.95,
    )
    sample = llm.get_default_sampling_params()
    sample.max_tokens = max_tokens
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    PROCESS_OBJECTS["llm"] = llm
    PROCESS_OBJECTS["sample"] = sample
    PROCESS_OBJECTS["tokenizer"] = tokenizer

    return llm, sample, tokenizer


def add_llm_answer(batch, rank, only_context, max_tokens, cuda_devices, model_name):
    llm, sample, tokenizer = get_model(model_name, cuda_devices, max_tokens, rank)
    batch_size = len(batch["context"])
    questions = []
    full_questions = []
    for j in range(batch_size):
        full_question = batch["context"][j] + "\n" + batch["question"][j]
        full_question = [{"role": "user", "content": full_question.strip()}]
        question = [{"role": "user", "content": batch["question"][j]}]
        full_question, question = tokenizer.apply_chat_template(
            [full_question, question],
            add_generation_prompt=True,
            tokenize=False,
        )
        full_questions.append(full_question)
        questions.append(question)

    response = llm.generate(full_questions, sample)
    answers_with_context = [elem.outputs[0].text for elem in response]
    if only_context:
        answers_no_context = [""] * len(answers_with_context)
    else:
        response = llm.generate(questions, sample)
        answers_no_context = [elem.outputs[0].text for elem in response]

    return {
        "llm_answer_with_context": answers_with_context,
        "llm_answer_no_context": answers_no_context,
    }


def generate_responses_with_llm(args):
    cuda_devices = args.cuda_devices.split(",")
    if len(cuda_devices) > 1:
        os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    dataset = datasets.load_from_disk(DATA_PATH / args.dataset_name)

    used_columns = ["dataset_index", "question", "context"]
    new_dataset = dataset.select_columns(used_columns)
    if args.limit >= 0:
        new_dataset = datasets.DatasetDict(
            {
                split: new_dataset[split].select(range(args.limit))
                for split in new_dataset.keys()
            }
        )
    fn_kwargs = {
        "only_context": args.only_context,
        "max_tokens": args.max_tokens,
        "model_name": args.model_name,
        "cuda_devices": cuda_devices,
    }
    new_dataset = new_dataset.map(
        add_llm_answer,
        batched=True,
        batch_size=args.batch_size,
        with_rank=True,
        num_proc=len(cuda_devices),
        desc="Adding llm_answer...",
        fn_kwargs=fn_kwargs,
    )
    model_name = args.model_name.split("/")[-1]
    dataset_name = f"{args.dataset_name}_{model_name}_{args.max_tokens}"
    save_path = RESPONSE_PATH / model_name / dataset_name
    save_path.mkdir(exist_ok=True, parents=True)
    new_dataset.save_to_disk(save_path)

    free_memory()


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Add LLM responses to the instruction dataset")
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
        "--only-context",
        action="store_true",
        help="Generate only the context version (default: False)",
    )
    parser.add_argument(
        "--max-tokens",
        default=512,
        type=int,
        help="Max tokens for generation (Default: 512)",
    )
    parser.add_argument(
        "--batch-size",
        default=2048,
        type=int,
        help="Batch size for the conversion (Default: 2048)",
    )
    parser.add_argument(
        "--limit",
        default=-1,
        type=int,
        help="Limit dataset to this number of samples (Default: -1)",
    )
    parser.add_argument(
        "--cuda-devices",
        default="0",
        type=str,
        help="String of cuda device ids separated with comma (Default: '0')",
    )
    args = parser.parse_args()
    generate_responses_with_llm(args)
