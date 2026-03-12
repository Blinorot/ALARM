import argparse
import gc
import os
import random
import time

import datasets
import torch
from combined_dataset import ALL_DATASETS, COLUMNS_TO_TAKE
from create_prompts import (
    DATA_PATH,
    PROMPT_PATH,
    SEED,
    choose_prompt,
    get_checker_main_prompt,
    get_checker_system_prompt,
    get_dataset_system_prompt,
    get_main_prompt,
)
from torch.multiprocessing import set_start_method
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.distributed import cleanup_dist_env_and_memory

set_start_method("spawn")


PROCESS_OBJECTS = {
    "llm": None,
    "sample": None,
    "tokenizer": None,
}


def get_model(model_name, cuda_devices, rank):
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
        max_model_len=8704,
        max_num_seqs=2048,
        max_num_batched_tokens=8704,
        gpu_memory_utilization=0.95,
    )
    sample = llm.get_default_sampling_params()
    sample.max_tokens = 4000
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    PROCESS_OBJECTS["llm"] = llm
    PROCESS_OBJECTS["sample"] = sample
    PROCESS_OBJECTS["tokenizer"] = tokenizer

    return llm, sample, tokenizer


def get_prompt_list(
    elems,
    prompt_client,
    prompt_sample,
    prompt_tokenizer,
    prompt_system_prompt,
    debug=False,
):
    prompts = []
    for elem in elems:
        main_prompt = get_main_prompt(elem)
        messages = [
            {"role": "system", "content": prompt_system_prompt},
            {"role": "user", "content": main_prompt},
        ]
        prompt = prompt_tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )
        prompts.append(prompt)
    responses = prompt_client.generate(prompts, prompt_sample)
    prompt_lists = []
    for response in responses:
        response = response.outputs[0].text
        if debug:
            print("===FULL LLM RESPOSNE===")
            print(response)
        response = response.split("<answer>")[-1]
        prompt_list = response.split("</answer>")[0].strip()
        prompt_lists.append(prompt_list)

    return prompt_lists


def get_checker_response(
    prompt_lists,
    audio_descriptions,
    checker_model_client,
    checker_sample,
    checker_tokenizer,
    checker_system_prompt,
):
    prompts = []
    for prompt_list, desc in zip(prompt_lists, audio_descriptions):
        checker_main_prompt = get_checker_main_prompt(prompt_list, desc)
        messages = [
            {"role": "system", "content": checker_system_prompt},
            {"role": "user", "content": checker_main_prompt},
        ]
        prompt = checker_tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        prompts.append(prompt)
    check_responses = checker_model_client.generate(prompts, checker_sample)
    is_good_lists = []
    for check_response in check_responses:
        check_response = check_response.outputs[0].text
        check_response = check_response.split("<answer>")[-1]
        is_good_list = check_response.split("</answer>")[0].strip()
        is_good_lists.append(is_good_list)
    return is_good_lists


def get_prompt(prompt_lists, is_good_lists):
    prompts = []
    for i in range(len(prompt_lists)):
        prompt = choose_prompt(prompt_lists[i], is_good_lists[i])
        prompts.append(prompt)
    return prompts


def add_prompts_to_batch(
    batch,
    rank,
    prompt_model_name,
    prompt_system_prompt,
    checker_model_name,
    checker_system_prompt,
    stage,
    cuda_devices,
):
    if stage == "generation":
        llm, sample, tokenizer = get_model(prompt_model_name, cuda_devices, rank)
        descriptions = batch["audio_description"]

        prompt_lists = get_prompt_list(
            descriptions,
            prompt_client=llm,
            prompt_sample=sample,
            prompt_tokenizer=tokenizer,
            prompt_system_prompt=prompt_system_prompt,
        )
        return {"prompt_list": prompt_lists}
    elif stage == "filter":
        llm, sample, tokenizer = get_model(checker_model_name, cuda_devices, rank)
        descriptions = batch["audio_description"]
        prompt_lists = batch["prompt_list"]

        is_good_lists = get_checker_response(
            prompt_lists=prompt_lists,
            audio_descriptions=descriptions,
            checker_model_client=llm,
            checker_sample=sample,
            checker_tokenizer=tokenizer,
            checker_system_prompt=checker_system_prompt,
        )
        prompts = get_prompt(prompt_lists, is_good_lists)
        return {"prompt": prompts}
    else:
        raise NotImplementedError()


def add_prompts_to_dataset(
    dataset_name,
    batch_size,
    prompt_model_name,
    checker_model_name,
    limit,
    num_proc,
    cuda_devices,
):
    ds = datasets.load_from_disk(DATA_PATH / dataset_name)["train"]
    # we will save only columns subset to avoid audio re-saving
    ds = ds.select_columns(["dataset_index", "audio_description"])

    if limit > 0:
        ds = ds.select(range(limit))

    random.seed(SEED)
    prompt_system_prompt = get_dataset_system_prompt(ds)
    checker_system_prompt = get_checker_system_prompt()

    fn_kwargs = {
        "prompt_model_name": prompt_model_name,
        "prompt_system_prompt": prompt_system_prompt,
        "checker_model_name": checker_model_name,
        "checker_system_prompt": checker_system_prompt,
        "cuda_devices": cuda_devices,
    }
    # STAGE=GENERATION
    stage_kwargs = {"stage": "generation"}
    stage_kwargs.update(**fn_kwargs)
    ds = ds.map(
        add_prompts_to_batch,
        batched=True,
        batch_size=batch_size,
        num_proc=num_proc,
        with_rank=True,
        desc=f"{dataset_name}: Adding prompt lists...",
        fn_kwargs=stage_kwargs,
    )
    # clear GPUs
    free_memory()
    # STAGE=FILTER
    stage_kwargs = {"stage": "filter"}
    stage_kwargs.update(**fn_kwargs)
    ds = ds.map(
        add_prompts_to_batch,
        batched=True,
        batch_size=batch_size,
        num_proc=num_proc,
        with_rank=True,
        desc=f"{dataset_name}: Filtering prompt lists...",
        fn_kwargs=stage_kwargs,
    )
    ds = ds.remove_columns("prompt_list")
    ds = ds.sort(["dataset_index"])
    ds = datasets.DatasetDict({"train": ds})
    ds.save_to_disk(PROMPT_PATH / f"{dataset_name}_with_prompts")

    # clear GPUs
    free_memory()


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


def debug(
    prompt_model_name, checker_model_name, cuda_devices, debug_ind, debug_dataset
):
    llm, sample, tokenizer = get_model(prompt_model_name, cuda_devices, rank=0)

    print("===PROMPT SAMPLE ARGS===")
    print(sample)

    ds = datasets.load_from_disk(DATA_PATH / debug_dataset)["train"]
    elem = ds[debug_ind]
    audio_description = elem["audio_description"]
    system_prompt = get_dataset_system_prompt(ds)
    main_prompt = get_main_prompt(audio_description)
    prompt_list = get_prompt_list(
        [audio_description], llm, sample, tokenizer, system_prompt, debug=True
    )[0]

    system_tokens, main_tokens, prompt_list_tokens = tokenizer(
        [system_prompt, main_prompt, prompt_list],
        padding=False,
        truncation=False,
        add_special_tokens=False,
    ).input_ids
    llm.llm_engine.engine_core.shutdown()
    del llm
    del sample
    del tokenizer
    free_memory()
    llm, sample, tokenizer = get_model(checker_model_name, cuda_devices, rank=0)
    print("===CHECKER SAMPLE ARGS===")
    print(sample)
    checker_system_prompt = get_checker_system_prompt()
    checker_main_prompt = get_checker_main_prompt(prompt_list, audio_description)
    is_good_list = get_checker_response(
        [prompt_list],
        [audio_description],
        llm,
        sample,
        tokenizer,
        checker_system_prompt,
    )[0]
    checker_system_tokens, checker_main_tokens, is_good_list_tokens = tokenizer(
        [checker_system_prompt, checker_main_prompt, is_good_list],
        padding=False,
        truncation=False,
        add_special_tokens=False,
    ).input_ids

    llm.llm_engine.engine_core.shutdown()
    del llm
    del sample
    del tokenizer
    free_memory()

    print("===SYSTEM===")
    print(system_prompt)
    print("===MAIN===")
    print(main_prompt)
    print("===LLM===")
    print(prompt_list)

    print("===CHECKER SYSTEM===")
    print(checker_system_prompt)
    print("===CHECKER MAIN===")
    print(checker_main_prompt)
    print("===CHECKER RESPONSE===")
    print(is_good_list)

    prompt = choose_prompt(prompt_list, is_good_list)
    print("===CHOSEN PROMPT===")
    print(prompt)

    print(
        (
            f"Prompt tokens lengths, system: {len(system_tokens)}, "
            f"main: {len(main_tokens)}, prompt_list: {len(prompt_list_tokens)}"
        )
    )
    print(
        (
            f"Checker tokens lengths, system: {len(checker_system_tokens)}, "
            f"main: {len(checker_main_tokens)}, is_good_list: {len(is_good_list_tokens)}"
        )
    )


def create_prompts(batch_size, limit, cuda_devices, debug_ind, debug_dataset):
    if len(cuda_devices) > 1:
        os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # vllm serve MODEL_NAME --port PORT --enable-prefix-caching

    # prompt_model_name = "neuralmagic/Meta-Llama-3.1-8B-Instruct-FP8-dynamic"
    # prompt_model_name = "RedHatAI/Llama-3.3-70B-Instruct-FP8-dynamic"
    # prompt_model_name = "openai/gpt-oss-120b"
    # prompt_model_name = "openai/gpt-oss-20b"
    # prompt_model_name = "RedHatAI/Qwen3-32B-FP8-dynamic"
    # prompt_model_name = "Qwen/Qwen3-Next-80B-A3B-Instruct-FP8"
    prompt_model_name = "Qwen/Qwen3-30B-A3B-Instruct-2507-FP8"

    # checker_model_name = "neuralmagic/Meta-Llama-3.1-8B-Instruct-FP8-dynamic"
    # checker_model_name = "RedHatAI/Llama-3.3-70B-Instruct-FP8-dynamic"
    # checker_model_name = "openai/gpt-oss-20b"
    # checker_model_name = "openai/gpt-oss-120b"
    # checker_model_name = "RedHatAI/Qwen3-32B-FP8-dynamic"
    # checker_model_name = "Qwen/Qwen3-Next-80B-A3B-Instruct-FP8"
    checker_model_name = "Qwen/Qwen3-30B-A3B-Instruct-2507-FP8"

    if debug_ind >= 0:
        assert len(cuda_devices) == 1, "For debug, select only 1 cuda device"
        debug(
            prompt_model_name,
            checker_model_name,
            cuda_devices,
            debug_ind,
            debug_dataset,
        )
        return None

    for dataset_name in ALL_DATASETS:
        if (PROMPT_PATH / f"{dataset_name}_with_prompts").exists():
            print(f"{dataset_name} is already processed, skipping ...")
            continue
        print(f"Processing {dataset_name}")
        add_prompts_to_dataset(
            dataset_name=dataset_name,
            batch_size=batch_size,
            prompt_model_name=prompt_model_name,
            checker_model_name=checker_model_name,
            limit=limit,
            num_proc=len(cuda_devices),
            cuda_devices=cuda_devices,
        )


if __name__ == "__main__":
    PROMPT_PATH.mkdir(exist_ok=True, parents=True)

    parser = argparse.ArgumentParser("Create prompts")
    parser.add_argument(
        "--batch-size",
        default=32,
        type=int,
        help="Batch size for the conversion (Default: 32)",
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
    parser.add_argument(
        "--debug-ind",
        default=-1,
        type=int,
        help="If debug-ind >= 0, choose debug-ind elem from debug dataset (Default: -1)",
    )
    parser.add_argument(
        "--debug-dataset",
        default="cameo",
        type=str,
        help="Dataset used for debugging (Default: cameo)",
    )

    args = parser.parse_args()
    create_prompts(
        batch_size=args.batch_size,
        limit=args.limit,
        cuda_devices=args.cuda_devices.split(","),
        debug_ind=args.debug_ind,
        debug_dataset=args.debug_dataset,
    )
