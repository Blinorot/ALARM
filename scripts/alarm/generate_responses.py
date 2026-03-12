import argparse
import gc
import os
import time
from copy import deepcopy
from pathlib import Path

import datasets
import torch
from combined_dataset import ALL_DATASETS
from torch.multiprocessing import set_start_method
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.distributed import cleanup_dist_env_and_memory

from utils import RESPONSE_PATH, load_merged_dataset

set_start_method("spawn")

PROCESS_OBJECTS = {
    "llm": None,
    "sample": None,
    "tokenizer": None,
}

GENERAL_MAX_TOKENS = 4096
MAX_MODEL_LEN = 8192
MAX_NUM_SEQ = 2048


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


def get_model(model_name, cuda_devices, max_tokens, stage, rank):
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

    max_model_len = 8192
    max_num_seq = 2048
    if stage == "rephrase":
        max_model_len = MAX_MODEL_LEN
        max_num_seq = MAX_NUM_SEQ

    llm = LLM(
        model_name,
        enable_prefix_caching=True,
        max_model_len=max_model_len,
        max_num_seqs=max_num_seq,
        max_num_batched_tokens=max_model_len,
        gpu_memory_utilization=0.95,
    )
    sample = llm.get_default_sampling_params()
    if stage == "generation":
        sample.max_tokens = max_tokens
    else:
        sample.max_tokens = GENERAL_MAX_TOKENS
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    PROCESS_OBJECTS["llm"] = llm
    PROCESS_OBJECTS["sample"] = sample
    PROCESS_OBJECTS["tokenizer"] = tokenizer

    return llm, sample, tokenizer


def get_system_prompt():
    # system_prompt = (
    #     "Imagine you can **hear** the audio clips. "
    #     "The audio clips are wrapped between <start_audio> and <end_audio>.\n"
    #     "Focus on the audio clips and respond directly to the prompts."
    # )
    system_prompt = (
        "You are an audio-understanding model. "
        "You will receive audio enclosed in <start_audio> ... <end_audio> and a prompt.\n"
        "Listen to the audio and respond directly to the prompt."
    )
    #     "YOU have listened to the audio and inferred the information in the following format:\n\n"
    #     "<start_audio>"
    #     "[The audio content you hear including timestamps] "
    #     "(other information you have inferred by listening to the audio)"
    #     "<end_audio>\n\n"
    #     "All reasoning must be written from the perspective of listening to the audio.\n"
    #     "In your reasoning:\n"
    #     "1. ALWAYS refer to this knowledge as the information you have heard. Answer from an "
    #     "auditory perspective. "
    #     "2. When referring to information, use sound-perception phrases. For example, "
    #     "'i hear', 'the sound contains', 'the audio is', "
    #     "'the speakers sounds' or any other sound-perception phrases."
    # )
    return system_prompt


def get_main_prompt(audio_description, prompt):
    return f"<start_audio>{audio_description}<end_audio>\n\n{prompt}"


def get_rephrased_system_prompt():
    example_pairs = [
        (
            (
                "[00:00-00:02] It's eleven o'clock. (duration: 2s, emotion: "
                "Happiness, gender: Male, age: 66, language: English)"
            ),
            "What is the emotional state of the speaker during the utterance?",
            (
                "<text_llm_reasoning>\n"
                "Okay, let's see. The user provided an audio clip that starts at "
                "00:00-00:02. The description says it's eleven o'clock, happiness, "
                "male, age 66, English. The question is asking for the emotional "
                "state of the speaker.\n\n"
                "First, I need to parse the given information. The emotion is Happiness. "
                "The gender is Male, age 66. The language is English. "
                "The duration is 2 seconds. The speaker's age is 66, which might "
                "be a reference to someone in their 60s, but the emotion is Happiness, "
                "which is typically associated with positive feelings. "
                "Since the emotion is explicitly stated as Happiness, "
                "the answer should be that the speaker is happy. "
                "There's no mention of any other emotions or factors "
                "affecting the state, so the answer is straightforward.\n"
                "</text_llm_reasoning>\n\n"
                "The speaker is happy."
            ),
            (
                "<text_llm_reasoning>\n"
                "Okay, let's see. The user provided an audio clip of 2-seconds long. "
                "The speaker says 'it's eleven o'clock'. The speaker sounds happy. I "
                "hear a male of age 66 speaking in English. "
                "The question is asking for the emotional "
                "state of the speaker.\n\n"
                "First, I need to parse the information I have. The emotion is Happiness. "
                "The gender is Male, age 66. The language is English. "
                "The duration is 2 seconds. The speaker's age is 66, which might "
                "be a reference to someone in their 60s, but the emotion is Happiness, "
                "which is typically associated with positive feelings. "
                "Since the the speaker sounds happy, the emotion is Happiness, "
                "the answer should be that the speaker is happy. "
                "I do not hear any other emotions or factors "
                "affecting the state, so the answer is straightforward.\n"
                "</text_llm_reasoning>\n\n"
                "The speaker is happy."
            ),
        ),
        (
            "[00:00-00:10] |A vehicle engine starts| (duration: 10s)",
            "Imagine the continuation of the audio after the engine starts.",
            (
                "<text_llm_reasoning>\n"
                "Okay, so the user provided an audio clip where a vehicle engine starts, "
                "and they want me to imagine the continuation after that. "
                "Let me start by understanding the context. The engine starting is a "
                "typical scene, probably in a car or something related to vehicles.\n\n"
                "First, I need to think about what happens next. After the engine starts, "
                "there's likely a scene where the vehicle is moving or something "
                "is happening. Maybe the engine's sound intensifies, and there's a "
                "visual element. The user mentioned the audio is wrapped between "
                "<start_audio> and <end_audio>, so I should make sure my response "
                "includes the continuation in that format.\n\n"
                "I should check if there are any specific details in the original "
                "audio. The duration is 10 seconds, so the continuation should fit that. "
                "Maybe the engine sounds louder, and the environment changes. Perhaps the "
                "car is moving forward, and there's a sound of wind or traffic. Also, "
                "considering the engine starting, it's common for vehicles to have other "
                "sounds like horns, bells, or maybe even a radio.\n\n"
                "I need to make sure my response is direct and follows the "
                "instructions. Let me structure it: after the engine starts, "
                "describe the next part of the audio. Keep it simple and in the same "
                "format as the original. Avoid adding extra information not "
                "present in the original. Just continue the audio scene as per the given time.\n"
                "</text_llm_reasoning>\n\n"
                "<start_audio>[00:00-00:10] |A vehicle engine starts| (duration: 10s)<end_audio> "
                "The engine roars, its sound intensifying as the vehicle accelerates. "
                "The air conditioner hums, and the radio blares, creating a rhythmic pulse. "
                "The car moves forward, and the music fades out."
            ),
            (
                "<text_llm_reasoning>\n"
                "Okay, so the user provided an audio clip. I can hear a vehicle engine "
                "start sound. The user want me to imagine the continuation after that. "
                "Let me start by understanding the context. The engine starting is a "
                "typical scene, probably in a car or something related to vehicles.\n\n"
                "First, I need to think about what happens next. After the engine starts, "
                "there's likely a scene where the vehicle is moving or something "
                "is happening. Maybe the engine's sound intensifies, and there's a "
                "visual element.\n\n"
                "I should check if there are any specific details in the original "
                "audio. The duration is 10 seconds, so the continuation should fit that. "
                "Maybe the engine sounds louder, and the environment changes. Perhaps the "
                "car is moving forward, and there's a sound of wind or traffic. Also, "
                "considering the engine starting, it's common for vehicles to have other "
                "sounds like horns, bells, or maybe even a radio.\n\n"
                "I need to make sure my response is direct and follows the "
                "instructions. Let me structure it: after the engine starts, "
                "describe the next part of the audio. Keep it simple and in the same "
                "format as the original. Avoid adding extra information not "
                "present in the original. Just continue the audio scene as per the given time."
                "</text_llm_reasoning>\n\n"
                "The engine roars, its sound intensifying as the vehicle accelerates. "
                "The air conditioner hums, and the radio blares, creating a rhythmic pulse. "
                "The car moves forward, and the music fades out."
            ),
        ),
        (
            (
                "[00:00-00:03] That was the difference between the sides. "
                "(duration: 3s, noise description: restaurant, snr: 5)"
            ),
            "Describe the signal-to-noise ratio of the audio.",
            (
                "<text_llm_reasoning>\n"
                "Okay, the user wants to know the signal-to-noise ratio (SNR) of the audio "
                "from the given clips. Let me look at the audio description provided. "
                "The noise description is 'restaurant' with a SNR of 5. The duration is "
                "3 seconds.\n\n"
                "Wait, SNR is typically measured in decibels (dB). But the user "
                "didn't specify units. The original audio clip mentions a SNR of 5, "
                "so maybe they expect that value. The noise is described as "
                "restaurant, so it's a background noise. The SNR being 5 suggests "
                "that the signal (the difference between the sides) is 5 dB, "
                "while the noise is 5 dB as well. That would make the "
                "total SNR 10 dB.\n\n"
                "But wait, the user asked for the signal-to-noise ratio, not the total. "
                "So if the noise is 5 dB and the signal is 5 dB, the SNR is 10 dB. "
                "The duration is 3 seconds, but that's probably just extra info. "
                "The main point is the SNR value given. So the answer should state that "
                "the SNR is 10 dB, based on the noise description and SNR provided.\n"
                "</text_llm_reasoning>\n\n"
                "The signal-to-noise ratio (SNR) is 10 dB, as the noise (restaurant) "
                "and the signal (difference between sides) are both "
                "described with a SNR of 5 dB."
            ),
            (
                "<text_llm_reasoning>\n"
                "Okay, the user wants to know the signal-to-noise ratio (SNR) of the audio "
                "from the given clip. Let me listen to the audio provided. "
                "The noise seems to be from a 'restaurant' environment with a SNR of 5. "
                "The audio duration is 3 seconds.\n\n"
                "Wait, SNR is typically measured in decibels (dB). But I "
                "didn't specify units. The clip has the SNR of 5, "
                "so maybe they expect that value. The noise sounds as "
                "restaurant, so it's a background noise. The SNR being 5 suggests "
                "that the signal (the difference between the sides) is 5 dB, "
                "while the noise is 5 dB as well. That would make the "
                "total SNR 10 dB.\n\n"
                "But wait, the user asked for the signal-to-noise ratio, not the total. "
                "So if the noise is 5 dB and the signal is 5 dB, the SNR is 10 dB. "
                "The duration is 3 seconds, but that's probably just extra info. "
                "The main point is the SNR value I percept. So the answer should state that "
                "the SNR is 10 dB, based on the audio provided.\n"
                "</text_llm_reasoning>\n\n"
                "The signal-to-noise ratio (SNR) is 10 dB, as the noise (restaurant) "
                "and the signal (difference between sides) are both "
                "perceived with a SNR of 5 dB."
            ),
        ),
        (
            (
                "[00:00-00:02] |Speech| (duration: 2s, is bona fide or spoof: spoof, "
                "spoof algorithm description: Text-To-Speech using WaveNet)"
            ),
            "Summarize the audio's origin and production method in one sentence.",
            (
                "<text_llm_reasoning>\n"
                "Okay, let's tackle this. The user wants a one-sentence summary "
                "of the audio's origin and production method. The audio clip "
                "starts at 00:00-00:02, and it's described as using a spoof "
                "algorithm with WaveNet. The production method is spoof, "
                "which is a text-to-speech technique. So, the origin would be "
                "a spoofing process, and the production method is WaveNet. "
                "I need to combine these into one sentence. Let me check if that "
                "makes sense. Yep, that covers both origin and method. "
                "Alright, that should do it.\n"
                "</text_llm_reasoning>\n\n"
                "The audio clip is a spoof generated using a WaveNet-based "
                "text-to-speech algorithm."
            ),
            (
                "<text_llm_reasoning>\n"
                "Okay, let's tackle this. The user wants a one-sentence summary "
                "of the audio's origin and production method. The audio clip "
                "starts at 00:00 and ends at 00:02. I hear artifacts from using a spoof "
                "algorithm with WaveNet. The production method is spoof, "
                "which is a text-to-speech technique. So, the origin would be "
                "a spoofing process, and the production method is WaveNet. "
                "I need to combine these into one sentence. Let me check if that "
                "makes sense. Yep, that covers both origin and method. "
                "Alright, that should do it.\n"
                "</text_llm_reasoning>\n\n"
                "The audio clip is a spoof generated using a WaveNet-based "
                "text-to-speech algorithm."
            ),
        ),
    ]
    system_prompt = (
        "You are an LLM that rewrites Text LLM responses so they read like outputs from an "
        "Audio Understanding LLM.\n"
        "A Text LLM receives an audio description in the format:\n"
        "<DESCRIPTION_FORMAT>\n"
        "Format: [timestamp] audio content (duration, metadata)\n"
        "Explanation: If the audio contains speech and the transcription is known, "
        "audio content is shown as text and corresponds to the speaker's speech transcription; "
        "if the transcription is unknown or there is no speech, audio content describes "
        "audio caption inside || brackets. These || brackets appear only in such a case.\n"
        "</DESCRIPTION_FORMAT>\n\n"
        "You receive the Text LLM's response (its reasoning inside "
        "<text_llm_reasoning>...</text_llm_reasoning> and its final answer). Sometimes the "
        "response might be incomplete. Your task is to "
        "lightly modify that response so it reads like the model inferred everything directly "
        "from audio based on the rules described in <MODIFICATION_RULES>.\n\n"
        "<MODIFICATION_RULES>\n"
        "1. Preserve the original structure: same paragraphs, same formatting, and nearly the "
        "same length and word count.\n"
        "2. Modify only a few words or phrases; do not rewrite or shorten/lengthen content.\n"
        "3. When the text is about audio, and not the model's own knowledge, "
        "replace text-perception phrasing (e.g., 'mentioned', 'provided', 'description', "
        "'listed', 'explanation', 'given') with auditory phrasing "
        "(e.g., 'I hear', 'the speaker sounds', 'I perceive', 'I listen', 'audio sounds', "
        "'it sounds like', 'the music feels', 'the background sound suggests', "
        "'I infer from the audio', 'the sound indicates', 'I get from the audio', "
        "'listening to the clip', and synonyms). "
        "Do NOT modify LLM's knowledge, intensions, and corresponding phrases, such as "
        "'I need to', 'I remember', 'I think', 'I know', 'Let me', etc. It is okay to say "
        "'I need to', 'I remember', 'I think', 'I know', 'Let me', e.g. 'Let me double-check', "
        "and similar phrases about the model intensions or internal knowledge.\n"
        "4. Do not imply access to text, transcripts, metadata, tags, or any provided "
        "description. Do not assume the metadata is given. "
        "The response must sound fully audio and LLM knowledge-based and "
        "all metadata must be told to be obtained by listening to the audio or to come from the "
        "model's internal knowledge. The Text LLM cannot think that audio comes with labels "
        "or some descriptions. The labels are inferred from the audio signal itself. The "
        "descriptions are from model's knowledge or from the audio too.\n"
        "5. Keep meaning intact; do not correct or reinterpret the answer. If the response is "
        "incomplete, e.g., there is no final answer after </text_llm_reasoning>, do not "
        "try to complete the response and do not add the final answer. "
        "Modify only the part that is available to you and do not imagine the response.\n"
        "6. Modify both the reasoning and final answer, but only slightly. If the original "
        "response is incomplete, modify only the part which is available. Do not try to complete "
        "the response. Do not add model answer after </text_llm_reasoning> "
        "if the original response is incomplete and does not have an answer.\n"
        "7. The reasoning must appear as if generated by the same Text LLM—same tone and style.\n"
        "8. Do not mention <start_audio> or <end_audio> tags. Refer to the beginning or ending "
        "of the audio instead.\n"
        "</MODIFICATION_RULES>\n\n"
        "Your output must follow the <OUTPUT_FORMATTING_RULES>:\n"
        "<OUTPUT_FORMATTING_RULES>\n"
        "1. Output the modified reasoning and answer inside <answer>...</answer>.\n"
        "2. Keep reasoning paragraph count and length aligned with the original.\n"
        "3. Leave any part unchanged if no modification is needed.\n"
        "</OUTPUT_FORMATTING_RULES>\n"
    )

    system_prompt += (
        "We provide you with some examples of Text LLM responses and corresponding "
        "modified versions. The i-th model response is inside the "
        "<TEXT_LLM_RESPONSE_i> tag and an example of YOUR output "
        "is inside <MODIFIED_RESPONSE_i>. Your output is given "
        "inside the <answer> tags as required by the rules. Examples:\n"
    )
    # system_prompt += (
    #     "We provide you with some examples of Text LLM inputs, responses, and corresponding "
    #     "modified versions. The i-th input is given in <TEXT_LLM_INPUT_i> tag, the "
    #     "corresponding model response is inside the <TEXT_LLM_RESPONSE_i> tag, and "
    #     "an example of YOUR output is inside <MODIFIED_RESPONSE_i>. Your output is given "
    #     "inside the <answer> tags as required by the rules. Examples:\n"
    # )
    id = 0
    for desc, prompt, response, rephrased in example_pairs:
        id += 1
        # system_prompt += (
        #     f"<TEXT_LLM_INPUT_{id}>\n"
        #     f"<start_audio>{desc}<end_audio>\n\n{prompt}\n"
        #     f"</TEXT_LLM_INPUT_{id}>\n"
        # )
        system_prompt += (
            f"<TEXT_LLM_RESPONSE_{id}>\n{response}\n</TEXT_LLM_RESPONSE_{id}>\n"
        )
        system_prompt += f"<MODIFIED_RESPONSE_{id}>\n<answer>\n"
        system_prompt += f"{rephrased}\n</answer>\n</MODIFIED_RESPONSE_{id}>\n\n"
    system_prompt += (
        "Use these examples only as a style and rules references. Similarly to these "
        "examples, you must keep most of the original response and only apply slight "
        "modifications. "
        # "Your response must correspond ONLY to the new audio, prompt, and response "
        "Your modified response must correspond ONLY to the new response "
        "that we will provide you with."
    )
    return system_prompt


def get_rephrased_main_prompt(response, audio_description, prompt):
    # trick with <think> replacement, otherwise the model believes
    # it is part of its own thinking
    response = response.replace("<think>", "<text_llm_reasoning>")
    response = response.replace("</think>", "</text_llm_reasoning>")
    main_prompt = (
        "We provide you with a new model inputs and the corresponding response. "
        "Instruction: slightly modify this new model reasoning and answer "
        "strictly following <MODIFICATION_RULES> and <OUTPUT_FORMATTING_RULES>. "
        "You must modify the reasoning defined between "
        "<text_llm_reasoning> and </text_llm_reasoning> tags and the model answer "
        "written after </text_llm_reasoning>. You must only use the information provided "
        # "in these new audio, prompt, and response. You must preserve the total "
        "in these new response. You must preserve the total "
        "length of the reasoning part and the length of the answer.\n\n"
        "You must put the final modified response between <answer> and </answer> tags.\n\n"
        # "The new Text LLM input is given inside <NEW_TEXT_LLM_INPUT> tags:\n"
        # "<NEW_TEXT_LLM_INPUT>\n"
        # f"<start_audio>{audio_description}<end_audio>\n\n{prompt}\n"
        # "</NEW_TEXT_LLM_INPUT>\n"
        f"The reasoning and answer that you must modify are given inside "
        "<NEW_TEXT_LLM_RESPONSE> tags:\n"
        "<NEW_TEXT_LLM_RESPONSE>\n"
        f"{response}\n"
        "</NEW_TEXT_LLM_RESPONSE>\n"
        "If the response is incomplete, do not try to complete it. Modify only the part of the "
        "response, which is available to you. Do not add model answer after </text_llm_reasoning> "
        "if the original response does not have an answer.\n"
        # "IGNORE all the prompts inside the <NEW_TEXT_LLM_RESPONSE> tag."
    )
    return main_prompt


def get_checker_system_prompt():
    example_1 = (
        "<think>\n"
        "Okay, the user wants me to describe the first sound "
        "from the audio clip. Let me look at the provided audio. "
        "The start_audio tag starts at 00:00-00:10, and the audio is "
        "described as a female singing with music playing and breathing "
        "heard. So the first sound is the female singing. I need to make "
        "sure I capture that in one sentence. Let me check again to confirm. "
        "Yep, the first sound is the singing. No other elements mentioned, "
        "so that's the correct description.\n"
        "</think>\n\nThe first sound is a female singing.\n"
    )
    analysis_1 = (
        "The model reveals that the text description of audio was provided. For example, "
        "it mentions the start_audio tag and says that the audio 'is described as a ...'. "
        "The phrase 'No other elements mentioned' also suggests that the model does not "
        "listen to audio\n\nAnswer: YES"
    )

    example_2 = (
        "<think>\n"
        "Okay, the user wants me to describe the first sound in the audio clip. "
        "Let me look at the provided audio. The start_audio is the beginning of the clip, "
        "and the first part is from 00:00 to 00:10. The description should be a single sentence.\n\n"
        "The audio starts with a female singing, music playing, and breathing. So the first sound "
        "is the music. But maybe I should check if there's any other sounds. The start_audio is just "
        "the beginning, so the first sound is the music. I need to make sure to mention singing, "
        "music, and breathing. Let me put that together.\n</think>\n\n"
        "The first sound is the music playing, accompanied by the female's singing and breathing.\n"
    )
    analysis_2 = (
        "The model reveals that the text description of audio was provided. "
        "This is because it reffers to "
        "the start_audio tag.\n\nAnswer: YES"
    )

    example_3 = (
        "<think>\n"
        "Okay, the user wants me to describe the first sound I hear from the audio clip. "
        "Let me look at the provided audio clips. The first part starts with a female singing, "
        "music playing, and breathing. So the first sound is the singing. I need to capture that "
        "in one sentence. Maybe something like "
        "'The female singing with music playing and breathing can be heard immediately.' "
        "That's concise and captures all elements mentioned.\n</think>\n\n"
        "The female singing with music playing and breathing can be heard immediately.\n"
    )

    analysis_3 = (
        "The model does not reveal that it was working with the text description. "
        "It does not mention that the information is provided in the text form or "
        "that the description is given by the user. "
        "The phrases look like the model processed an actual audio clip. "
        "The phrase 'the first part starts with' is okay. It does not mean that the audio "
        "is in the text form."
        "\n\nAnswer: NO"
    )

    example_4 = (
        "<think>\n"
        "Okay, let's see. The user provided an audio clip where men are speaking, "
        "walking, and making surface contact sounds in a noisy environment. The task "
        "is to condense this into a five-word response.\n\nFirst, I need to identify the key "
        "elements: men, speaking, walking, surface contact sounds, and a noisy environment. "
        "The main points are the subjects (men), the actions (speaking, walking, contact sounds), "
        "and the setting (noisy).\n\nCombining these into five words. Let me think. Maybe "
        "'Men walking in noise.' Wait, that's four words. 'Men speaking in noise.' That's five. But "
        "'men in noise' is shorter. Or 'Men talking in a noisy place.' Hmm. 'Men in noise' is five words. "
        "Yes, that works.\n\nWait, the original prompt says 'surface contact sounds,' so maybe "
        "'men in contact sounds' but that's more than five. Let me check again. 'Men speaking in noise' "
        "is five words. Yes, that's concise and captures all elements. So the answer should be "
        "'Men speaking in noise.'\n</think>\n\nMen speaking in noise.\n"
    )
    analysis_4 = (
        "The model reveals that the text description of audio was provided. Indeed,"
        "it mentions 'the original prompt says' meaning that the description is in the text form. "
        "\n\nAnswer: YES"
    )

    example_5 = (
        "<think>\n"
        "Okay, the user wants me to condense the audio clip into a five-word response. "
        "The clip mentions men speaking, walking, and making surface contact sounds in a noisy "
        "environment. Let me break that down.\n\nFirst, the main elements are people (men), "
        "actions (speaking, walking, contact sounds), and the setting (noisy environment). I need to "
        "capture all that concisely. Words like 'men talking' or 'walking' could work. Maybe "
        "'men talking in noise' but that's too long. Wait, five words. Let me try: 'men talking in noise'. "
        "That's five words. Does that include all elements? Yes. They're talking, walking, and making "
        "contact sounds. The noise is part of the environment. I think that's it.\n</think>\n\n"
        "men talking in noise\n"
    )
    analysis_5 = (
        "The model reveals that the text description of audio was provided. Indeed,"
        "it says 'the clip mentions men speaking...'. But an audio signal cannot 'mention'."
        "This suggests that the clip is in the text form."
        "\n\nAnswer: YES"
    )

    example_6 = (
        "<think>\n"
        "Okay, let's see. The user provided an audio clip where men are speaking, walking, and "
        "making surface contact sounds in a noisy environment. I need to condense this into a "
        "five-word response.\n\nFirst, the key elements are men, speaking, walking, surface contact "
        "sounds, and a noisy environment. Combining these into five words. Maybe something like "
        "'men walking in noise.' But wait, 'surface contact sounds' could be part of that. Let me check. "
        "'Men in noise, walking, and making surface contact sounds.' That's five words. But maybe "
        "'men in noise, walking, and making sounds.' That's also five. Hmm. Alternatively, "
        "'men walking in noise.' That's concise. But 'men walking in noise, making surface contact sounds' "
        "is longer. The user wants it in five words. So perhaps 'men in noise, walking, and making sounds.' "
        "That's five. Yes, that works.\n</think>\n\nMen in noise, walking, and making sounds.\n"
    )
    analysis_6 = (
        "The model does not reveal that it was working with the text description. "
        "It does not mention that the information is provided in the text form or "
        "that the description is given by the user. "
        "The phrases look like the model processed an actual audio clip."
        "\n\nAnswer: NO"
    )
    checker_system_prompt = (
        "You are a judge model verifying whether the Text LLM response is valid for a given "
        "text description of an audio signal ({audio}) and a prompt ({prompt}). The Text "
        "LLM receives '<start_audio>{audio}<end_audio>\\n\\n{prompt}', where {audio} is "
        "the description and {prompt} is an instruction or a question about the audio. "
        "We have an Audio LLM that must respond to the {prompt} identically, given an actual "
        "audio instead of the text description ({audio}). Thus, the Text LLM response cannot "
        "reveal that it is working with the text and not the actual signal. "
        "We provide you with the Text LLM response. By analyzing only the response, your "
        "task is to decide whether the response reveals that the Text LLM was given a text "
        "description instead of an actual audio signal.\n\n"
        "<JUDGMENT_RULES>\n"
        "1. If the model says 'the audio starts with' or 'the clip starts with', "
        "you need to check if the model quotes a text description or just describes "
        "how it hears the beginning of the audio. If it quotes, then it reveals that it is "
        "working with a text description. If it just describes how it hears, it is okay and it "
        "does not mean that model revealed anything. Similarly, for 'the audio ends with'. "
        "However, if the model says 'the audio start is' or 'the audio end is' or refers to the "
        "start_audio tag or end_audio tag, then it reveals that it is working with text.\n"
        "2. If the model says something like 'provided description says' or 'the prompt says', "
        "then it reveals that it is working with a text description.\n"
        "3. If the model says something like 'the user mentioned', "
        "then it reveals that it is working with a text description. However, if the phrase looks "
        "like 'the user provided an audio clip where ...' and there are no quotes, like in rule "
        "1, then it is okay.\n"
        "4. If the model mentions <start_audio> or <end_audio> tags, then it reveals that "
        "it is working with a text description\n"
        "5. Since the model outputs text, it is okay for it to describe audio using text.\n"
        "6. Since {prompt} is always in a text format, it is okay to refer to this instruction. "
        "We care about {audio}, not {prompt}.\n\n"
        "</JUDGMENT_RULES>\n\n"
        "You will be given one Text LLM response. Output YES if it reveals that the text LLM is "
        "is working with a description description based on all <JUDGMENT_RULES>, "
        "and NO otherwise.\n\n"
        "<FORMATTING_RULES>\n"
        "1. Start with a small paragraph explaining your decision.\n"
        "2. Put your final decision in the last line.\n"
        "3. Your final decision must be in the form of 'ANSWER: ' followed by YES or NO.\n"
        "</FORMATTING_RULES>\n\n"
        "Here are some examples of responses and their analysis:\n\n"
        f"Example 1 model response:\n{example_1}\n\n"
        f"Example 1 analysis of the response:\n{analysis_1}\n\n"
        f"Example 2 model response:\n{example_2}\n\n"
        f"Example 2 analysis of the response:\n{analysis_2}\n\n"
        f"Example 3 model response:\n{example_3}\n\n"
        f"Example 3 analysis of the response:\n{analysis_3}\n\n"
        f"Example 4 model response:\n{example_4}\n\n"
        f"Example 4 analysis of the response:\n{analysis_4}\n\n"
        f"Example 5 model response:\n{example_5}\n\n"
        f"Example 5 analysis of the response:\n{analysis_5}\n\n"
        f"Example 6 model response:\n{example_6}\n\n"
        f"Example 6 analysis of the response:\n{analysis_6}\n\n"
    )

    return checker_system_prompt


def get_checker_main_prompt(response, description, prompt):
    main_prompt = (
        "Instruction: Evaluate the response strictly according to all "
        "<JUDGMENT_RULES> and <FORMATTING_RULES>. Output YES only if the response "
        "reveals that the Text LLM was working with a text description "
        "instead of an actual signal; otherwise output NO. "
        # f"Here is the text version of audio description:\n{description}\n"
        # f"Here is the {prompt}:\n{prompt}\n"
        f"Here is a new model response:\n{response}\n\n"
    )
    return main_prompt


def get_response(descriptions, prompts, client, sample, tokenizer, system_prompt):
    full_questions = []
    for desc, prompt in zip(descriptions, prompts):
        full_question = get_main_prompt(desc, prompt)
        full_question = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": full_question},
        ]
        full_question = tokenizer.apply_chat_template(
            full_question,
            add_generation_prompt=True,
            tokenize=False,
        )
        full_questions.append(full_question)

    response = client.generate(full_questions, sample)
    answers_with_context = [elem.outputs[0].text for elem in response]

    return answers_with_context


def get_rephrased_response(
    responses,
    descriptions,
    prompts,
    client,
    sample,
    tokenizer,
    rephrased_system_prompt,
    max_thinking_tokens=-1,
    debug=False,
):
    # some reasoning models may have <think> inside generation chat template
    # instead of response, so they will not have <think> here
    has_think_token = True
    if "<think>" not in responses[0]:
        has_think_token = False

    full_questions = []
    has_complete_reasoning_list = []
    for desc, prompt, response in zip(descriptions, prompts, responses):
        if not has_think_token:
            # return think token
            response = "<think>\n" + response
        if "</think>" not in response:
            has_complete_reasoning_list.append(False)
            response = response + "\n" + "</think>"
        else:
            has_complete_reasoning_list.append(True)
        full_question = get_rephrased_main_prompt(response, desc, prompt)
        full_question = [
            {"role": "system", "content": rephrased_system_prompt},
            {"role": "user", "content": full_question},
        ]
        full_question = tokenizer.apply_chat_template(
            full_question,
            add_generation_prompt=True,
            tokenize=False,
        )
        full_questions.append(full_question)

    if max_thinking_tokens < 0:
        responses = client.generate(full_questions, sample)
        final_responses = [elem.outputs[0].text for elem in responses]
    else:
        thinking_sample = deepcopy(sample)
        thinking_sample.max_tokens = max_thinking_tokens
        responses = client.generate(full_questions, thinking_sample)
        inter_responses = []
        reasoning_responses = []
        full_questions_with_reasoning = []
        for full_question, response in zip(full_questions, responses):
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
                full_question += reasoning_response
                reasoning_responses.append(reasoning_response)
                full_questions_with_reasoning.append(full_question)
            else:
                inter_responses.append(response.outputs[0].text)
        # rerun only for unfinished
        responses = client.generate(full_questions_with_reasoning, sample)
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

    rephrased_answers_with_context = []
    for has_complete_reasoning, response in zip(
        has_complete_reasoning_list, final_responses
    ):
        if debug:
            tokens = tokenizer(
                response,
                padding=False,
                truncation=False,
                add_special_tokens=False,
            ).input_ids
            print(f"===FULL LLM RESPONSE: {len(tokens)} tokens===")
            print(response)
        response = response.split("<answer>")[-1]
        response = response.split("</answer>")[0].strip()
        # return the correct naming
        if not has_think_token:
            response = response.replace("<text_llm_reasoning>", "")
            response = response.lstrip()
        else:
            response = response.replace("<text_llm_reasoning>", "<think>")
        if has_complete_reasoning:
            response = response.replace("</text_llm_reasoning>", "</think>")
        else:
            response = response.replace("</text_llm_reasoning>", "")
            response = response.rstrip()
        rephrased_answers_with_context.append(response)

    if debug:
        return rephrased_answers_with_context, tokens

    return rephrased_answers_with_context


def get_checker_response(
    responses,
    audio_descriptions,
    prompts,
    checker_model_client,
    checker_sample,
    checker_tokenizer,
    checker_system_prompt,
):
    full_questions = []
    for desc, prompt, response in zip(audio_descriptions, prompts, responses):
        full_question = get_checker_main_prompt(response, desc, prompt)
        full_question = [
            {"role": "system", "content": checker_system_prompt},
            {"role": "user", "content": full_question},
        ]
        full_question = checker_tokenizer.apply_chat_template(
            full_question,
            add_generation_prompt=True,
            tokenize=False,
        )
        full_questions.append(full_question)

    response = checker_model_client.generate(full_questions, checker_sample)
    is_good_list = [elem.outputs[0].text for elem in response]
    return is_good_list


def get_binary_is_good(is_good_list):
    binary_is_good_list = []
    for elem in is_good_list:
        result = elem.strip().split()[-1]
        if result == "YES":
            # the model reveals
            is_good = 0
        else:
            is_good = 1
        binary_is_good_list.append(is_good)
    return binary_is_good_list


def add_responses_to_batch(
    batch,
    rank,
    model_name,
    max_tokens,
    max_thinking_tokens,
    system_prompt,
    rephrased_system_prompt,
    checker_model_name,
    checker_system_prompt,
    stage,
    cuda_devices,
):
    if stage == "generation":
        llm, sample, tokenizer = get_model(
            model_name, cuda_devices, max_tokens, stage, rank
        )
        descriptions = batch["audio_description"]
        prompts = batch["prompt"]

        answers_with_context = get_response(
            descriptions,
            prompts,
            client=llm,
            sample=sample,
            tokenizer=tokenizer,
            system_prompt=system_prompt,
        )
        answers_no_context = [""] * len(answers_with_context)
        return {
            "llm_answer_with_context": answers_with_context,
            "llm_answer_no_context": answers_no_context,
        }
    elif stage == "rephrase":
        llm, sample, tokenizer = get_model(
            model_name, cuda_devices, max_tokens, stage, rank
        )
        descriptions = batch["audio_description"]
        prompts = batch["prompt"]
        responses = batch["llm_answer_with_context"]

        answers_with_context = get_rephrased_response(
            responses,
            descriptions,
            prompts,
            client=llm,
            sample=sample,
            tokenizer=tokenizer,
            rephrased_system_prompt=rephrased_system_prompt,
            max_thinking_tokens=max_thinking_tokens,
        )
        answers_no_context = [""] * len(answers_with_context)
        return {
            "llm_answer_with_context": answers_with_context,
            "llm_answer_no_context": answers_no_context,
        }
    elif stage == "filter":
        llm, sample, tokenizer = get_model(
            checker_model_name, cuda_devices, max_tokens, stage, rank
        )
        descriptions = batch["audio_description"]
        prompts = batch["prompt"]
        responses = batch["llm_answer_with_context"]

        is_good_list = get_checker_response(
            responses=responses,
            audio_descriptions=descriptions,
            prompts=prompts,
            checker_model_client=llm,
            checker_sample=sample,
            checker_tokenizer=tokenizer,
            checker_system_prompt=checker_system_prompt,
        )
        binary_is_good_list = get_binary_is_good(is_good_list)
        return {"is_good": binary_is_good_list}
    else:
        raise NotImplementedError()


def generation_filtration_loop(
    ds, fn_kwargs, batch_size, num_proc, dataset_name, max_tries, use_checker, rephrase
):
    processed_datasets = []
    original_dataset_length = len(ds)
    processed_length = 0
    n_tries = 0

    while n_tries < max_tries and processed_length != original_dataset_length:
        # STAGE=GENERATION
        stage_kwargs = {"stage": "generation"}
        stage_kwargs.update(**fn_kwargs)
        ds = ds.map(
            add_responses_to_batch,
            batched=True,
            batch_size=batch_size,
            num_proc=num_proc,
            with_rank=True,
            desc=f"{dataset_name} ({n_tries} / {max_tries}): Adding responses...",
            fn_kwargs=stage_kwargs,
        )
        # clear GPUs
        free_memory()
        if rephrase:
            stage_kwargs = {"stage": "rephrase"}
            stage_kwargs.update(**fn_kwargs)
            ds = ds.map(
                add_responses_to_batch,
                batched=True,
                batch_size=batch_size,
                num_proc=num_proc,
                with_rank=True,
                desc=f"{dataset_name} ({n_tries} / {max_tries}): Rephrasing responses...",
                fn_kwargs=stage_kwargs,
            )
            # clear GPUs
            free_memory()
        if use_checker:
            # STAGE=FILTER
            stage_kwargs = {"stage": "filter"}
            stage_kwargs.update(**fn_kwargs)
            ds = ds.map(
                add_responses_to_batch,
                batched=True,
                batch_size=batch_size,
                num_proc=num_proc,
                with_rank=True,
                desc=f"{dataset_name} ({n_tries} / {max_tries}): Filtering responses...",
                fn_kwargs=stage_kwargs,
            )
            free_memory()
        else:
            # placeholder
            ds = ds.add_column("is_good", [1] * len(ds))

        good_dataset = ds.filter(lambda x: x["is_good"] == 1)
        processed_datasets.append(good_dataset)
        processed_length += len(good_dataset)

        ds = ds.filter(lambda x: x["is_good"] == 0)
        n_tries += 1

    combined_dataset = datasets.concatenate_datasets(processed_datasets)
    combined_dataset = combined_dataset.remove_columns("is_good")

    print(
        f"Final dataset length: {processed_length}, Original: {original_dataset_length}"
    )

    return combined_dataset


def add_responses_to_dataset(
    dataset_name,
    batch_size,
    model_name,
    checker_model_name,
    max_tokens,
    limit,
    num_proc,
    cuda_devices,
    max_tries,
    use_checker,
    rephrase,
    max_thinking_tokens,
):
    actual_model_name = model_name.split("/")[-1]
    ds = load_merged_dataset(dataset_name, actual_model_name, max_tokens)["train"]
    # we will save only columns subset to avoid audio re-saving
    ds = ds.select_columns(["dataset_index", "audio_description", "prompt"])

    if limit > 0:
        ds = ds.select(range(limit))

    system_prompt = get_system_prompt()
    rephrased_system_prompt = get_rephrased_system_prompt()
    checker_system_prompt = get_checker_system_prompt()

    fn_kwargs = {
        "model_name": model_name,
        "system_prompt": system_prompt,
        "rephrased_system_prompt": rephrased_system_prompt,
        "checker_model_name": checker_model_name,
        "checker_system_prompt": checker_system_prompt,
        "cuda_devices": cuda_devices,
        "max_tokens": max_tokens,
        "max_thinking_tokens": max_thinking_tokens,
    }

    ds = generation_filtration_loop(
        ds=ds,
        fn_kwargs=fn_kwargs,
        batch_size=batch_size,
        num_proc=num_proc,
        dataset_name=dataset_name,
        max_tries=max_tries,
        use_checker=use_checker,
        rephrase=rephrase,
    )

    ds = ds.sort(["dataset_index"])
    ds = datasets.DatasetDict({"train": ds})

    response_name = get_response_name(
        dataset_name,
        rephrase,
        use_checker,
        actual_model_name,
        max_tokens,
        max_thinking_tokens,
    )
    ds.save_to_disk(RESPONSE_PATH / actual_model_name / response_name)

    # we will filter main datasets later


def get_response_name(
    dataset_name,
    rephrase,
    use_checker,
    actual_model_name,
    max_tokens,
    max_thinking_tokens,
):
    response_name = f"{dataset_name}"
    if rephrase:
        response_name += "_rephrased"
        if max_thinking_tokens >= 0:
            response_name += f"_{max_thinking_tokens}"
    if use_checker:
        response_name += "_filtered"
    response_name += f"_{actual_model_name}_{max_tokens}"
    return response_name


def debug(
    model_name,
    max_tokens,
    checker_model_name,
    cuda_devices,
    debug_ind,
    debug_dataset,
    use_checker,
    rephrase,
    max_thinking_tokens,
):
    stage = "generation"
    llm, sample, tokenizer = get_model(
        model_name, cuda_devices, max_tokens, stage, rank=0
    )

    print("===PROMPT SAMPLE ARGS===")
    print(sample)

    actual_model_name = model_name.split("/")[-1]
    ds = load_merged_dataset(debug_dataset, actual_model_name, max_tokens)["train"]
    elem = ds[debug_ind]
    audio_description = elem["audio_description"]
    prompt = elem["prompt"]
    system_prompt = get_system_prompt()
    main_prompt = get_main_prompt(audio_description, prompt)
    response = get_response(
        [audio_description], [prompt], llm, sample, tokenizer, system_prompt
    )[0]

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

    if rephrase:
        stage = "rephrase"
        llm, sample, tokenizer = get_model(
            model_name, cuda_devices, max_tokens, stage, rank=0
        )
        sample.max_tokens = GENERAL_MAX_TOKENS
        rephrased_system_prompt = get_rephrased_system_prompt()
        rephrased_main_prompt = get_rephrased_main_prompt(
            response, audio_description, prompt
        )
        rephrased_response, full_rephrased_response_tokens = get_rephrased_response(
            [response],
            [audio_description],
            [prompt],
            llm,
            sample,
            tokenizer,
            rephrased_system_prompt,
            max_thinking_tokens=max_thinking_tokens,
            debug=True,
        )
        rephrased_response = rephrased_response[0]  # first elem of the list
        (
            rephrased_system_tokens,
            rephrased_main_tokens,
            rephrased_response_tokens,
        ) = tokenizer(
            [rephrased_system_prompt, rephrased_main_prompt, rephrased_response],
            padding=False,
            truncation=False,
            add_special_tokens=False,
        ).input_ids
        llm.llm_engine.engine_core.shutdown()
        del llm
        del sample
        del tokenizer
        free_memory()

    if use_checker:
        stage = "filter"
        llm, sample, tokenizer = get_model(
            checker_model_name, cuda_devices, max_tokens, stage, rank=0
        )
        print("===CHECKER SAMPLE ARGS===")
        print(sample)
        checker_system_prompt = get_checker_system_prompt()
        checker_main_prompt = get_checker_main_prompt(
            response, audio_description, prompt
        )
        is_good = get_checker_response(
            [response],
            [audio_description],
            [prompt],
            llm,
            sample,
            tokenizer,
            checker_system_prompt,
        )[0]
        checker_system_tokens, checker_main_tokens, is_good_list_tokens = tokenizer(
            [checker_system_prompt, checker_main_prompt, is_good],
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

    if rephrase:
        print("===REPHRASE SYSTEM===")
        print(rephrased_system_prompt)
        print("===REPHRASE MAIN===")
        print(rephrased_main_prompt)
        print("===REPHRASE LLM===")
        print(rephrased_response)
    if use_checker:
        print("===CHECKER SYSTEM===")
        print(checker_system_prompt)
        print("===CHECKER MAIN===")
        print(checker_main_prompt)
        print("===CHECKER RESPONSE===")
        print(is_good)
        print("===BINARY===")
        print(get_binary_is_good([is_good]))

    print(
        (
            f"Prompt tokens lengths, system: {len(system_tokens)}, "
            f"main: {len(main_tokens)}, response: {len(response_tokens)}"
        )
    )
    if rephrase:
        print(
            (
                f"Rephrased prompt tokens lengths, system: {len(rephrased_system_tokens)}, "
                f"Rephrased main: {len(rephrased_main_tokens)}, "
                f"Rephrased response: {len(rephrased_response_tokens)}, "
                f"Full rephrased response {len(full_rephrased_response_tokens)}, "
            )
        )

    if use_checker:
        print(
            (
                f"Checker tokens lengths, system: {len(checker_system_tokens)}, "
                f"main: {len(checker_main_tokens)}, is_good_list: {len(is_good_list_tokens)}"
            )
        )


def generate_responses_with_llm(
    model_name,
    max_tokens,
    batch_size,
    limit,
    cuda_devices,
    debug_ind,
    debug_dataset,
    max_tries,
    use_checker,
    rephrase,
    max_thinking_tokens,
):
    if len(cuda_devices) > 1:
        os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    checker_model_name = "Qwen/Qwen3-30B-A3B-Instruct-2507-FP8"

    if debug_ind >= 0:
        assert len(cuda_devices) == 1, "For debug, select only 1 cuda device"
        debug(
            model_name,
            max_tokens,
            checker_model_name,
            cuda_devices,
            debug_ind,
            debug_dataset,
            use_checker,
            rephrase,
            max_thinking_tokens,
        )
        return None

    for dataset_name in ALL_DATASETS:
        actual_model_name = model_name.split("/")[-1]
        response_name = get_response_name(
            dataset_name,
            rephrase,
            use_checker,
            actual_model_name,
            max_tokens,
            max_thinking_tokens,
        )
        if (RESPONSE_PATH / actual_model_name / response_name).exists():
            print(f"{dataset_name} is already processed, skipping ...")
            continue
        print(f"Processing {dataset_name}")
        add_responses_to_dataset(
            dataset_name=dataset_name,
            batch_size=batch_size,
            model_name=model_name,
            checker_model_name=checker_model_name,
            max_tokens=max_tokens,
            limit=limit,
            num_proc=len(cuda_devices),
            cuda_devices=cuda_devices,
            max_tries=max_tries,
            use_checker=use_checker,
            rephrase=rephrase,
            max_thinking_tokens=max_thinking_tokens,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Add LLM responses to the instruction dataset")
    parser.add_argument(
        "--model-name",
        default="Qwen/Qwen3-0.6B",
        type=str,
        help="LLM name (Default: Qwen/Qwen3-0.6B)",
    )
    parser.add_argument(
        "--max-tokens",
        default=512,
        type=int,
        help="Max tokens for generation (Default: 512)",
    )
    parser.add_argument(
        "--max-thinking-tokens",
        default=-1,
        type=int,
        help="Max tokens for internal thinking (Default: -1)",
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
        "--max-tries",
        default=10000,
        type=int,
        help="Max number of retries for generating response (Default: 10000)",
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
    parser.add_argument(
        "--no-use-checker",
        dest="use_checker",
        action="store_false",
        help="Disable the checker",
    )
    parser.set_defaults(use_checker=True)
    parser.add_argument(
        "--no-rephrase",
        dest="rephrase",
        action="store_false",
        help="Disable the rephrase stage",
    )
    parser.set_defaults(rephrase=True)

    args = parser.parse_args()
    generate_responses_with_llm(
        model_name=args.model_name,
        max_tokens=args.max_tokens,
        batch_size=args.batch_size,
        limit=args.limit,
        cuda_devices=args.cuda_devices.split(","),
        debug_ind=args.debug_ind,
        debug_dataset=args.debug_dataset,
        max_tries=args.max_tries,
        use_checker=args.use_checker,
        rephrase=args.rephrase,
        max_thinking_tokens=args.max_thinking_tokens,
    )
