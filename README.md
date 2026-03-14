<h1 align="center">
ALARM:<br>
Audio–Language Alignment for Reasoning Models
  <br>
  [Under Review in Interspeech2026]
</h1>

<p align="center">
  <a href="#about">About</a> •
  <a href="#installation">Installation</a> •
  <a href="#inference">Inference</a> •
  <a href="#training">Training</a> •
  <a href="#evaluation">Evaluation</a> •
  <a href="#dataset">Dataset</a> •
  <a href="#credits">Credits</a> •
  <a href="#license">License</a>
</p>

<p align="center">
<a href="https://arxiv.org/abs/2603.09556">
  <img src="https://img.shields.io/badge/Paper-arXiv%3A2603.09556-b31b1b.svg?logo=arxiv&logoColor=white">
</a>
<a href="https://hf.co/collections/Blinorot/alarm">
  <img src="https://img.shields.io/badge/HuggingFace-Collection-yellow.svg?logo=huggingface&logoColor=white">
</a>
</a>
<a href="https://huggingface.co/datasets/Blinorot/ALARM-E-Demo">
  <img src="https://img.shields.io/badge/ALARM-Demo%20Samples-orange?logoColor=white">
</a>
</p>

## About

This is the official implementation of [ALARM: Audio–Language Alignment for Reasoning Models](https://arxiv.org/abs/2603.09556), an audio reasoning language model trained in a self-generation setup that achieves state-of-the-art performance on Speech Understanding benchmarks with a 4B backbone.

> **Abstract:** Large audio language models (ALMs) extend LLMs with auditory understanding. A common approach freezes the LLM and trains only an adapter on self-generated targets. However, this fails for reasoning LLMs (RLMs) whose built-in chain-of-thought traces expose the textual surrogate input, yielding unnatural responses. We propose self-rephrasing, converting self-generated responses into audio-understanding variants compatible with RLMs while preserving distributional alignment. We further fuse and compress multiple audio encoders for stronger representations. For training, we construct a 6M-instance multi-task corpus (2.5M unique prompts) spanning 19K hours of speech, music, and sound. Our 4B-parameter ALM outperforms similarly sized models and surpasses most larger ALMs on related audio-reasoning benchmarks, while preserving textual capabilities with a low training cost. Notably, we achieve the best open-source result on the MMAU-speech and MMSU benchmarks and rank third among all the models.

**Authors:** [Petr Grinberg](https://www.linkedin.com/in/petr-grinberg/) and [Hassan Shahmohammadi](https://www.linkedin.com/in/hassan-shahmohamadi/).

## Installation

Set up the environment and dependencies

```bash
# we use uv==0.10.4
uv venv --python 3.11.7 ./environments/my_new_env
source ~/environments/my_new_env/bin/activate

uv pip install -r requirements.txt --torch-backend=cu128
```

## Inference

All model checkpoints are available on [HuggingFace](https://hf.co/collections/Blinorot/alarm). We also provide [vLLM](https://github.com/vllm-project/vllm) support using [vLLM Prompt Embedding API](https://docs.vllm.ai/en/stable/features/prompt_embeds/). Since ALARM uses the frozen Qwen3 model as the backbone, `vllm` just runs the original Qwen3 checkpoint, and the ALARM checkpoint is used for extracting LLM input embeddings. After you cloned the repo and installed the dependencies, you can run the pretrained model as follows:

```python
# Import libraries
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0" #optional

# run before importing torch because generate_vllm sets the multiprocessing method
from generate_vllm import get_response
from src.model.wrapped_llms.qwen3 import Qwen3AudioWrappedFeatureExtractor

from omegaconf import OmegaConf
from torchaudio.utils import _download_asset
from torchcodec.decoders import AudioDecoder
from transformers import AutoTokenizer
from vllm import LLM


# The model configuration config.
# Handles vllm-related configuration and defines feature extractors,
# i.e., audio -> encoder input embedding conversion.
# All other configuration, including model architecture, will be
# loaded from the checkpoint.
default_model_config_name = "src/configs/model/default_inference.yaml"
model_config = OmegaConf.load(default_model_config_name)

# checkpoint_name = which model to run
#   Single model version (no inference-time ensemble):
#   checkpoint_name='Blinorot/AL-Whisper-Instruct-R'
#   ALARM-E embedding fusion-type version (inference-time ensemble):
#   checkpoint_name=["Blinorot/ALARM-CA","Blinorot/AL-Whisper-Instruct-R"]
checkpoint_name = ["Blinorot/ALARM-CA","Blinorot/AL-Whisper-Instruct-R"]

device = "cuda"

# Load Tokenizer for Text Processing
tokenizer = AutoTokenizer.from_pretrained(model_config.llm)

# Load ALARM/AL-*-R checkpoints for extraction of LLM input embeddings
if isinstance(checkpoint_name, list): # ALARM-E-style embedding fusion (inference-time ensemble)
    feature_extractor_list = []
    for name in checkpoint_name:
        # Load weights into the (audio,text)->LLM embeddings converter
        feature_extractor = Qwen3AudioWrappedFeatureExtractor(
            model_config=model_config,
            checkpoint_name=name,
            tokenizer=tokenizer,
        )
        feature_extractor.to(device)
        feature_extractor_list.append(feature_extractor)
    feature_extractor = feature_extractor_list
else: # Single Model version (no inference-time ensemble)
    # Load weights into the (audio,text)->LLM embeddings converter
    feature_extractor = Qwen3AudioWrappedFeatureExtractor(
        model_config=model_config,
        checkpoint_name=checkpoint_name,
        tokenizer=tokenizer,
    )
    feature_extractor.to(device)

# Start the offline vLLM instance of original Qwen3 RLM
# Model will be loaded to CUDA_VISIBLE_DEVICES id
llm = LLM(
    model_config.llm,
    enable_prefix_caching=True,
    max_model_len=model_config.max_model_len,
    max_num_seqs=model_config.max_num_seq,
    max_num_batched_tokens=model_config.max_num_batched_tokens,
    gpu_memory_utilization=model_config.gpu_memory_utilization,
    enable_prompt_embeds=True,
)

# Set sampling arguments for the RLM
sample = llm.get_default_sampling_params()
sample.seed = model_config.seed
sample.max_tokens = model_config.max_tokens

# Define audio and prompt
# Audio must come from torchcodec.AudioDecoder
audio_example_path = _download_asset("tutorial-assets/ctc-decoding/1688-142285-0007.wav")
audio = AudioDecoder(audio_example_path)
prompt = "Describe the audio content."

# Define a system prompt
system_prompt = "You are an audio-understanding model."

# Obtain response from Audio RLM
response = get_response(
    prompts=[prompt], # list of all the prompts
    audio_list=[audio], # list of corresponding audio
    llm=llm,
    feature_extractor=feature_extractor,
    sample=sample,
    tokenizer=tokenizer,
    system_prompt=system_prompt,
    max_thinking_tokens=model_config.max_thinking_tokens, # controls thinking budget for the RLM
    debug=False,
)

# Response is a list of responses, one per each (prompt, audio) input pair
# We have only one input pair, so the final response is at index 0
response = response[0]

print(f"Model response:\n\n{response}")
```

One can also run the `vLLM` version for the whole dataset using our `generale_vllm.py` script. The dataset is required to have the following fields:

- `audio`: audio to process.
- `content` or `prompt`: text context or text prompt, respectively.

Datasets with `prompts` generated via our [Dataset Creation](#dataset-creation) scripts will be merged with their prompts automatically. We provide an example below. The script will run the model over the whole dataset and save responses locally or remotely (on HuggingFace):

```bash
### ARGUMENTS DEFINITION
# dataset.name = local or remote dataset name
# dataset.limit = optionally limit to N samples
# dataset.split = optionally limit to a specific split
# dataset.shuffle = optionally shuffle the dataset
# cuda_devices = GPU IDs
#   Multi-GPU inference is supported in DataParallel style
# batch_size = batch size
# checkpoint_name = which model to run
#   Single model version:
#   checkpoint_name='Blinorot/AL-Whisper-Instruct-R'
#   ALARM-E embedding fusion-type version:
#   checkpoint_name='["Blinorot/ALARM-CA","Blinorot/AL-Whisper-Instruct-R"]'
# model.max_thinking_tokens = thinking budget for the model
# model.max_tokens = max output length
# push_to_hub = optionally push the results to hub
# hf_public = optionally make HF repo public
# hf_repo_name = optionally provide HF repo name
# save_audio = optionally save audio in HF repo too

### COMMAND EXAMPLE
python3 generate_vllm.py \
    dataset.name="yijingwu/HeySQuAD_human" \
    dataset.limit=100 \
    dataset.split="validation" \
    dataset.shuffle=True \
    cuda_devices="'$CUDA_VISIBLE_DEVICES'" \
    batch_size=8 \
    checkpoint_name='["Blinorot/ALARM-CA","Blinorot/AL-Whisper-Instruct-R"]' \
    model.max_thinking_tokens=8192 \
    model.max_tokens=16384 \
    push_to_hub=True \
    hf_public=False \
    hf_repo_name="ALARM-E-HeySQuAD_human" \
    save_audio=True
```

## Training

To train your own model, you need to configure training by modifying `Hydra` arguments in CLI or changing/adding `.yaml` configs under `src/configs` directory (see [Tutorials](https://github.com/Blinorot/pytorch_project_template#tutorials) to get familiar with `Hydra` if needed). The basic command is:

```bash
accelerate launch \
    --config-file src/configs/accelerate/multigpu.yaml \
    --num-processes NUM_GPUS \
    train.py HYDRA_OPTIONS
```

Where `HYDRA_OPTIONS` are `Hydra`-related modifications of configuration. See config files in `src/configs` for possible options. We provide full sets of training scripts needed to train `ALARM-CA` and `ALARM-P` in `train_alarm_ca.sh` and `train_alarm_p.sh`, respectively, as examples. The single-encoder variants from the paper are also included in these scripts.

## Evaluation

We evaluate our models on Multi-Choice Question Answering (MCQA) benchmarks: [MMSU](https://arxiv.org/abs/2506.04779), [MMAU](https://arxiv.org/abs/2410.19168), [MMAR](https://arxiv.org/abs/2505.13032), and [AIR-Bench](https://arxiv.org/abs/2402.07729).

First, we prepare the data with scripts from `scripts/benchmarks/dataset_collection` that will download all the benchmarks and convert them to a desirable `HuggingFace` format.

```bash
cd scripts/benchmarks/dataset_collection

python3 mmsu.py
python3 mmau.py
python3 mmar.py
python3 airbench.py
```

As MMSU and AIR-Bench have lots of small files, you may need to restart downloading many times. We provide `.sh` scripts for that next to the scripts.

Once the data is ready, the next step is to obtain responses from our model. We call the `generate_vllm_mcqa.py` script for that. Its CLI interface is the same as for the `generate_vllm.py` discussed in the [Inference](#inference) section. Alternatively, you can simply run the `run_eval.sh` wrapper. For example:

```bash
run_eval.sh mmsu alarm_ca "Blinorot/ALARM-CA" "test" 8
```

See `run_eval.sh` for arguments definition. The responses will be saved in `data/datasets/generated/LLM_MODEL_NAME/{benchmark_name}_{checkpoint_name}_{Optional:seed_{seed}}_{max_tokens}_{max_thinking_tokens}`.

Finally, we need to get `.json` from responses and calculate the scores using the official benchmark scripts. To do so, we call:

```bash
cd scripts/benchmarks

# python3 create_answer_json.py --dataset-name $dataset_name \
#             --checkpoint-name $checkpoint_name \
#             --max-tokens $max_tokens --max-thinking-tokens $max_thinking_tokens

# example:
python3 create_answer_json.py --dataset-name mmsu \
    --checkpoint-name "Blinorot/ALARM-CA" \
    --max-tokens 16384 --max-thinking-tokens 8192

cd evaluation

# python3 dataset_name.py --input=PATH_TO_JSON

# example
python3 mmsu.py --input="PATH_TO_ROOT/data/datasets/generated/Qwen3-4B-Thinking-2507/mmsu_Blinorot_ALARM-CA_16384_8192/metadata/test_metadata.json"
```

In general, the `generate_vllm.py` script can be used to generate responses for any dataset and some subsequent scripts can be added for evaluating these responses.

## Dataset

Our models are trained in a self-generation manner: (i) Text Large Language Model (LLM) backbone generates responses given a prompt and a text description of an audio and (ii) Audio Language Model (ALM) is trained on actual audio with these responses as targets.

We propose a dataset collection framework with two features:

1. Obtaining diverse and aligned prompts for self-generation based training of Audio Language Models.
2. A self-rephrasing mechanism enabling self-generation based training of Audio Reasoning Models.

All information regarding dataset creation and usage are provided in [DATASET](./DATASET.md) file. Our prompts and generated responses are available in [ALARM Corpora](https://huggingface.co/datasets/Blinorot/ALARM-Corpora). See [DATASET](./DATASET.md) for instructions on downloading corresponding audio files. ALARM-Corpora statistics are provided below:

| Audio Type  | # Elements (M)     | # Hours (K)          | # Unique Prompts (M) |
| ----------- | ------------------ | -------------------- | -------------------- |
| Speech      | 2.91 / 2.60 / 0.29 | 9.88 / 8.83 / 0.98   | 1.40 / 1.27 / 0.16   |
| Sound       | 2.01 / 1.80 / 0.20 | 5.54 / 4.98 / 0.55   | 0.36 / 0.33 / 0.06   |
| Music       | 0.59 / 0.53 / 0.06 | 2.45 / 2.21 / 0.24   | 0.16 / 0.14 / 0.03   |
| Instruction | 0.57 / 0.56 / 0.01 | 1.02 / 1.01 / 0.01   | 0.57 / 0.56 / 0.01   |
| **Total**   | 6.08 / 5.49 / 0.56 | 18.89 / 17.03 / 1.78 | 2.49 / 2.30 / 0.26   |

## Citation

If you use this work, please cite:

```bibtex
@article{grinberg2026alarm,
  title={ALARM: Audio-Language Alignment for Reasoning Models},
  author={Grinberg, Petr and Shahmohammadi, Hassan},
  journal={arXiv preprint arXiv:2603.09556},
  year={2026}
}
```

## Credits

Some parts of the code are based on/inspired by/use code from these repositories:

- [LLaMA-Omni](https://github.com/ictnlp/LLaMA-Omni): Modality Wrapper Code Design
- [Perceiver](https://github.com/krasserm/perceiver-io): Perceiver Implementation
- [MMSU Benchmark](https://github.com/dingdongwang/MMSU): Evaluation Script
- [MMAU Benchmark](https://github.com/Sakshi113/MMAU): Evaluation Script
- [MMAR Benchmark](https://github.com/ddlBoJack/MMAR): Evaluation Script
- [AIR-Bench Benchmark](https://github.com/OFA-Sys/AIR-Bench): Evaluation Script

We thank the authors for sharing their work.

## License

Code in this repository is licensed under the [MIT License](/LICENSE_CODE).

Model checkpoints located on [HuggingFace](https://hf.co/collections/Blinorot/alarm) are licensed under
[Creative Commons Attribution-NonCommercial 4.0 (CC BY-NC 4.0)](/LICENSE_CHECKPOINTS).
They may only be used for non-commercial research purposes.

The collected ALARM corpora preserves the licenses of its constituent datasets apart from our model responses, which are licensed under the same [Creative Commons Attribution-NonCommercial 4.0 (CC BY-NC 4.0)](/LICENSE_CHECKPOINTS).
