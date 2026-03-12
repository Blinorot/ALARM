import argparse
import gc
import os
import time
from copy import deepcopy
from pathlib import Path

import datasets
import hydra
import torch
from omegaconf import ListConfig
from torch.multiprocessing import set_start_method
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.distributed import cleanup_dist_env_and_memory

from src.model.wrapped_llms.qwen3 import Qwen3AudioWrappedFeatureExtractor
from src.utils import ROOT_PATH

set_start_method("spawn")

PROCESS_OBJECTS = {
    "llm": None,
    "sample": None,
    "tokenizer": None,
    "feature_extractor": None,
}

RESPONSE_PATH = ROOT_PATH / "data" / "datasets" / "generated"
DATA_PATH = ROOT_PATH / "data" / "datasets" / "raw"


def free_memory():
    if PROCESS_OBJECTS["llm"] is not None:
        PROCESS_OBJECTS["llm"].llm_engine.engine_core.shutdown()
        del PROCESS_OBJECTS["llm"]
        del PROCESS_OBJECTS["sample"]
        del PROCESS_OBJECTS["tokenizer"]
        del PROCESS_OBJECTS["feature_extractor"]
    PROCESS_OBJECTS["llm"] = None
    PROCESS_OBJECTS["sample"] = None
    PROCESS_OBJECTS["tokenizer"] = None
    PROCESS_OBJECTS["feature_extractor"] = None
    cleanup_dist_env_and_memory()
    gc.collect()
    torch.cuda.empty_cache()
    time.sleep(5)


def get_model(model_config, checkpoint_name, cuda_devices, seed, rank):
    if PROCESS_OBJECTS["llm"] is not None:
        return (
            PROCESS_OBJECTS["llm"],
            PROCESS_OBJECTS["sample"],
            PROCESS_OBJECTS["tokenizer"],
            PROCESS_OBJECTS["feature_extractor"],
        )

    # cuda_devices == list of available devices,
    # e.g. cuda_devices = "[0, 1, 2, 3]"
    if rank is None:  # Only 1 proc
        rank = 0
    cuda_device = cuda_devices[rank]
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_device

    tokenizer = AutoTokenizer.from_pretrained(model_config.llm)
    if isinstance(checkpoint_name, list) or isinstance(checkpoint_name, ListConfig):
        feature_extractor_list = []
        for name in checkpoint_name:
            feature_extractor = Qwen3AudioWrappedFeatureExtractor(
                model_config=model_config,
                checkpoint_name=str(Path(name).resolve()),
                tokenizer=tokenizer,
            )
            print("FINDING META...")
            for name, p in feature_extractor.named_parameters():
                if p.is_meta:
                    print("meta", name)
            feature_extractor.to("cuda")
            feature_extractor_list.append(feature_extractor)
        feature_extractor = feature_extractor_list
    else:
        feature_extractor = Qwen3AudioWrappedFeatureExtractor(
            model_config=model_config,
            checkpoint_name=str(Path(checkpoint_name).resolve()),
            tokenizer=tokenizer,
        )
        print("FINDING META...")
        for name, p in feature_extractor.named_parameters():
            if p.is_meta:
                print("meta", name)
        feature_extractor.to("cuda")

    print(model_config)

    llm = LLM(
        model_config.llm,
        enable_prefix_caching=True,
        max_model_len=model_config.max_model_len,
        max_num_seqs=model_config.max_num_seq,
        max_num_batched_tokens=model_config.max_num_batched_tokens,
        gpu_memory_utilization=model_config.gpu_memory_utilization,
        enable_prompt_embeds=True,
    )
    sample = llm.get_default_sampling_params()
    print("SAMPLE SEED", sample.seed)
    sample.seed = seed
    sample.max_tokens = model_config.max_tokens
    PROCESS_OBJECTS["llm"] = llm
    PROCESS_OBJECTS["sample"] = sample
    PROCESS_OBJECTS["tokenizer"] = tokenizer
    PROCESS_OBJECTS["feature_extractor"] = feature_extractor

    return llm, sample, tokenizer, feature_extractor


def get_system_prompt():
    system_prompt = (
        "You are an audio-understanding model. "
        "You will be given an audio signal, a question about it, and "
        "a list of possible answer options in the following format: "
        "'(Option_{option_ID}) {option_value}'.\n\n"
        "Analyze the audio and choose the correct option that answers the question. "
        "Put your final answer between <answer> and </answer> tags. "
        "Your answer must be a single number: the {option_ID} that "
        "corresponds to the correct option. You must always return a valid {option_ID}, "
        "even if you are not sure about the correct choice.\n\n"
        "Good response format example:\n"
        "<answer>{option_id}</answer>\n\n"
        "Bad response format example:\n"
        "{some text}<answer>{option_id}<answer>\n\n"
        "The bad response is bad because it returns additional text apart from the answer. "
        "Generate only good responses."
    )
    return system_prompt


def get_main_prompt(question, choices):
    options_list = ""
    for i, option in enumerate(choices):
        option = f"(Option_{i}) {option}\n"
        options_list += option
    main_prompt = (
        "Instruction: Answer the question defined in the <question> tag. "
        "The options are given inside <options> tag. You must return only your "
        "final answer and put it between <answer> and </answer> tags. "
        "Your answer must be a single number corresponding to "
        "the correct {option_ID}.\n\n"
        f"<question>\n{question}\n</question>\n\n"
        f"<options>\n{options_list}</options>\n\n"
    )
    return main_prompt


def convert_single_text_to_batch_ids(text, tokenizer, batch_size):
    text = [text] * batch_size
    inputs_text = tokenizer(
        text, padding=False, truncation=False, add_special_tokens=False
    ).input_ids
    inputs_text = [torch.tensor(inputs_text[i]) for i in range(batch_size)]
    return inputs_text


def concat_batch_input_ids(inputs_list, batch_size):
    concat_inputs_list = []
    for i in range(batch_size):
        concat_inputs_list.append(torch.cat([elem[i] for elem in inputs_list], dim=-1))
    return concat_inputs_list


def get_response(
    questions,
    choices_list,
    audio_list,
    llm,
    feature_extractor,
    sample,
    tokenizer,
    system_prompt,
    max_thinking_tokens=-1,
    debug=False,
):
    if isinstance(feature_extractor, list):
        model_config = feature_extractor[0].model.get_config()
    else:
        model_config = feature_extractor.model.get_config()
    use_explicit_audio_tokens = model_config.use_explicit_audio_tokens
    inputs_system = []
    inputs_start = []
    inputs_main = []
    for question, options in zip(questions, choices_list):
        full_question = get_main_prompt(question, options)
        full_question_system = [
            {"role": "system", "content": system_prompt},
        ]
        full_question_system = tokenizer.apply_chat_template(
            full_question_system,
            add_generation_prompt=False,
            tokenize=False,
        )
        full_question_user = [
            {"role": "user", "content": full_question},
        ]
        full_question_user = tokenizer.apply_chat_template(
            full_question_user,
            add_generation_prompt=True,
            tokenize=False,
        )
        full_question_start = full_question_user.split("\n")[0] + "\n"
        full_question_main = full_question_user[len(full_question_start) :]
        if use_explicit_audio_tokens:
            full_question_start = full_question_start + "<start_audio>"
            full_question_main = "<end_audio>" + full_question_main
        inputs_system.append(full_question_system)
        inputs_start.append(full_question_start)
        inputs_main.append(full_question_main)

    inputs_system = tokenizer(
        inputs_system, padding=False, truncation=False, add_special_tokens=False
    ).input_ids
    inputs_start = tokenizer(
        inputs_start, padding=False, truncation=False, add_special_tokens=False
    ).input_ids
    inputs_main = tokenizer(
        inputs_main, padding=False, truncation=False, add_special_tokens=False
    ).input_ids
    batch_size = len(inputs_start)
    inputs_system = [torch.tensor(inputs_system[i]) for i in range(batch_size)]
    inputs_start = [torch.tensor(inputs_start[i]) for i in range(batch_size)]
    inputs_main = [torch.tensor(inputs_main[i]) for i in range(batch_size)]

    if isinstance(feature_extractor, list):
        full_prompt_embeds_list = []
        full_inputs_system = inputs_system
        full_inputs_start = inputs_start
        full_inputs_main = inputs_main
        for i in range(len(feature_extractor)):
            inputs_system = None
            input_start_str = (
                "This is what you hear in the audio "
                f"for the ({i+1}/{len(feature_extractor)}) time:\n"
            )
            inputs_main_str = "\n"
            inputs_start = convert_single_text_to_batch_ids(
                input_start_str, tokenizer, batch_size
            )
            inputs_main = convert_single_text_to_batch_ids(
                inputs_main_str, tokenizer, batch_size
            )
            if i == 0:
                extra_input_str = (
                    f"You will hear the same audio {len(feature_extractor)} times and perceive "
                    "different aspects of it, e.g., content description, environment description, "
                    "speech transcription, sound quality, etc.. "
                    "Depending on the audio the aspects may differ."
                    "Process all of the information to make a decision.\n\n"
                )
                extra_inputs_start = convert_single_text_to_batch_ids(
                    extra_input_str, tokenizer, batch_size
                )
                inputs_start = concat_batch_input_ids(
                    [full_inputs_start, extra_inputs_start, inputs_start], batch_size
                )
                inputs_system = full_inputs_system

            if i == (len(feature_extractor) - 1):
                inputs_main = concat_batch_input_ids(
                    [inputs_main, full_inputs_main], batch_size
                )

            prompt_embeds_list = feature_extractor[i](
                inputs_system=inputs_system,
                inputs_start=inputs_start,
                inputs_main=inputs_main,
                audio=audio_list,
            )
            prompt_embeds_list = [elem.detach().cpu() for elem in prompt_embeds_list]
            full_prompt_embeds_list.append(prompt_embeds_list)

        prompt_embeds_list = []
        for b in range(batch_size):
            prompt_embeds_list.append(
                torch.cat(
                    [elem[b] for elem in full_prompt_embeds_list], dim=-2
                )  # time-wise
            )

        prompt_embeds_dict = [{"prompt_embeds": elem} for elem in prompt_embeds_list]
        feature_extractor = feature_extractor[0]  # for text embeddings below
    else:
        prompt_embeds_list = feature_extractor(
            inputs_system=inputs_system,
            inputs_start=inputs_start,
            inputs_main=inputs_main,
            audio=audio_list,
        )
        prompt_embeds_list = [elem.detach().cpu() for elem in prompt_embeds_list]
        prompt_embeds_dict = [{"prompt_embeds": elem} for elem in prompt_embeds_list]

    if max_thinking_tokens < 0:
        responses = llm.generate(prompt_embeds_dict, sample)
        final_responses = [elem.outputs[0].text for elem in responses]
    else:
        thinking_sample = deepcopy(sample)
        thinking_sample.max_tokens = max_thinking_tokens
        responses = llm.generate(prompt_embeds_dict, thinking_sample)
        inter_responses = []
        reasoning_responses = []
        prompt_embeds_list_with_reasoning = []
        for prompt_embeds, response in zip(prompt_embeds_list, responses):
            inside_reasoning = "</think>" not in response.outputs[0].text
            if response.outputs[0].finish_reason == "length":
                # mark as unfinished
                inter_responses.append("")

                reasoning_response = response.outputs[0].text
                if inside_reasoning:
                    reasoning_response += (
                        "\n\nConsidering the limited time by the user, I have to give the "
                        "solution based on the thinking directly now.\n</think>\n\n"
                    )
                reasoning_responses.append(reasoning_response)

                reasoning_response = torch.tensor(
                    tokenizer(
                        reasoning_response,
                        padding=False,
                        truncation=False,
                        add_special_tokens=False,
                    ).input_ids
                )
                reasoning_response = feature_extractor.get_text_embeds(
                    [reasoning_response]
                )[0]
                reasoning_response = reasoning_response.detach().cpu()
                prompt_embeds_with_reasoning = torch.cat(
                    [prompt_embeds, reasoning_response], dim=0
                )
                prompt_embeds_list_with_reasoning.append(prompt_embeds_with_reasoning)
            else:
                inter_responses.append(response.outputs[0].text)
        # rerun only for unfinished
        prompt_embeds_list_with_reasoning_dict = [
            {"prompt_embeds": elem} for elem in prompt_embeds_list_with_reasoning
        ]
        responses = llm.generate(prompt_embeds_list_with_reasoning_dict, sample)
        response_index = 0
        final_responses = []
        # combine finished with unfinished into final responses
        for elem in inter_responses:
            if elem == "":
                reasoning_response = reasoning_responses[response_index]
                response = responses[response_index].outputs[0].text
                final_responses.append(reasoning_response + response)
                response_index += 1
            else:
                final_responses.append(elem)

    answers = []
    for response in final_responses:
        if debug:
            tokens = tokenizer(
                response,
                padding=False,
                truncation=False,
                add_special_tokens=False,
            ).input_ids
            print(f"===FULL LLM RESPONSE: {len(tokens)} tokens===")
            print(response)
        print(response)
        response = response.split("<answer>")[-1]
        response = response.split("</answer>")[0].strip()
        answers.append(response)

    if debug:
        return answers, tokens

    return answers


def get_choice_from_id(choices_list, llm_answers):
    answers = []
    for choices, llm_answer in zip(choices_list, llm_answers):
        try:
            llm_answer_id = int(llm_answer)
        except ValueError:
            llm_answer_id = 0
        try:
            choice = choices[llm_answer_id]
        except:
            print(llm_answer_id, llm_answer, len(choices), choices)
            # if the option is number, sometimes llm can return number instead of id
            choice = str(llm_answer_id)
        answers.append(choice)
    return answers


def add_responses_to_batch(
    batch,
    rank,
    model_config,
    checkpoint_name,
    system_prompt,
    cuda_devices,
    seed,
):
    llm, sample, tokenizer, feature_extractor = get_model(
        model_config, checkpoint_name, cuda_devices, seed, rank
    )
    questions = batch["question"]
    choices_list = batch["choices"]
    audio_list = batch["audio"]

    llm_answers = get_response(
        questions,
        choices_list,
        audio_list,
        llm=llm,
        feature_extractor=feature_extractor,
        sample=sample,
        tokenizer=tokenizer,
        system_prompt=system_prompt,
        max_thinking_tokens=model_config.max_thinking_tokens,
    )
    final_answers = get_choice_from_id(choices_list, llm_answers)
    return {
        "llm_answer": final_answers,
    }


def add_responses_to_dataset(
    dataset_name,
    dataset_split,
    limit,
    batch_size,
    model_config,
    checkpoint_name,
    num_proc,
    cuda_devices,
    seed,
):
    dataset_splits = {}

    full_ds = datasets.load_from_disk(DATA_PATH / dataset_name)

    for split in full_ds.keys():
        if dataset_split is not None and dataset_split != "" and dataset_split != split:
            print(f"Desired split is {dataset_split}. Skipping {split}...")
            continue
        ds = full_ds[split]
        if limit > 0:
            ds = ds.select(range(limit))

        system_prompt = get_system_prompt()

        fn_kwargs = {
            "model_config": model_config,
            "checkpoint_name": checkpoint_name,
            "system_prompt": system_prompt,
            "cuda_devices": cuda_devices,
            "seed": seed,
        }

        ds = ds.map(
            add_responses_to_batch,
            batched=True,
            batch_size=batch_size,
            num_proc=num_proc,
            with_rank=True,
            desc=f"{dataset_name}, {split}: Adding responses...",
            fn_kwargs=fn_kwargs,
        )

        # drop audio when saving responses
        ds = ds.select_columns(["dataset_index", "question", "choices", "llm_answer"])

        ds = ds.sort(["dataset_index"])
        dataset_splits[split] = ds

    ds = datasets.DatasetDict(dataset_splits)

    model_name = model_config.llm.split("/")[-1]
    response_name = get_response_name(
        dataset_name,
        checkpoint_name,
        model_config.max_tokens,
        model_config.max_thinking_tokens,
        seed,
    )
    ds.save_to_disk(RESPONSE_PATH / model_name / response_name)

    free_memory()


def debug(dataset_name, debug_ind, model_config, checkpoint_name, cuda_devices, seed):
    llm, sample, tokenizer, feature_extractor = get_model(
        model_config, checkpoint_name, cuda_devices, seed, rank=0
    )

    print("===PROMPT SAMPLE ARGS===")
    print(sample)

    ds = datasets.load_from_disk(DATA_PATH / dataset_name)["test"]
    elem = ds[debug_ind]
    question = elem["question"]
    options = elem["choices"]
    audio = elem["audio"]

    system_prompt = get_system_prompt()
    main_prompt = get_main_prompt(question, options)
    response, full_response_tokens = get_response(
        [question],
        [options],
        [audio],
        llm=llm,
        feature_extractor=feature_extractor,
        sample=sample,
        tokenizer=tokenizer,
        system_prompt=system_prompt,
        max_thinking_tokens=model_config.max_thinking_tokens,
        debug=True,
    )
    response = response[0]

    system_tokens, main_tokens, response_tokens = tokenizer(
        [system_prompt, main_prompt, response],
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
    print(response)

    print(
        (
            f"Prompt tokens lengths, system: {len(system_tokens)}, "
            f"Main: {len(main_tokens)}, "
            f"Response: {len(response_tokens)}, "
            f"Full response {len(full_response_tokens)}, "
        )
    )


def get_response_name(
    dataset_name, checkpoint_name, max_tokens, max_thinking_tokens, seed
):
    response_name = f"{dataset_name}"
    if isinstance(checkpoint_name, list) or isinstance(checkpoint_name, ListConfig):
        save_checkpoint_name = "_".join(
            [name.replace("/", "_") for name in checkpoint_name]
        )
    else:
        save_checkpoint_name = checkpoint_name.replace("/", "_")
    if seed is not None:
        save_checkpoint_name += f"_seed_{seed}"
    response_name += f"_{save_checkpoint_name}_{max_tokens}_{max_thinking_tokens}"
    return response_name


@hydra.main(
    version_base=None,
    config_path=str(ROOT_PATH / "src" / "configs"),
    config_name="generate",
)
def generate_responses_with_llm(config):
    cuda_devices = str(config.cuda_devices)
    cuda_devices = cuda_devices.split(",")

    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    print(f"Running with seed {config.seed}")

    if config.dataset.debug_ind > 0:
        debug(
            dataset_name=config.dataset.name,
            debug_ind=config.dataset.debug_ind,
            model_config=config.model,
            checkpoint_name=config.checkpoint_name,
            cuda_devices=cuda_devices,
            seed=config.seed,
        )
        return

    add_responses_to_dataset(
        dataset_name=config.dataset.name,
        dataset_split=config.dataset.split,
        limit=config.dataset.limit,
        batch_size=config.batch_size,
        model_config=config.model,
        checkpoint_name=config.checkpoint_name,
        num_proc=len(cuda_devices),
        cuda_devices=cuda_devices,
        seed=config.seed,
    )


if __name__ == "__main__":
    generate_responses_with_llm()
