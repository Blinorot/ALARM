import numpy as np
import torch
import torchaudio
from datasets import concatenate_datasets, load_from_disk
from torch import nn
from torch.utils.data import Dataset

from src.dataset.data_utils import load_merged_dataset
from src.utils import ROOT_PATH

DATA_PATH = ROOT_PATH / "data" / "datasets"


class HFDataset(Dataset):
    def __init__(
        self,
        dataset_names,
        split,
        tokenizer,
        feature_extractor,
        limit=None,
        return_audio=True,
        name="",
        use_explicit_audio_tokens=False,
        data_dir="data",
    ):
        """
        Args:
            dataset_names (list[list[str|float]]): list of dataset names. Can be combined.
                Includes a list with 3 elements [DatasetName, 'WithContext' or 'NoContext',
                percentage of the dataset to use]. Percentage must be in [0, 1] scale.
            split (str): dataset split.
            tokenizer (HF.Tokenizer): tokenizer object.
            feature_extractor (src.model.audio.AudioFeatureExtractor): converter from
                audio to audio encoder features.
            limit (None | int): limit dataset to this humber.
            return_audio (bool): if False, return only text.
            name (str): dataset name.
            use_explicit_audio_tokens (bool): if True, add <start_audio> and <end_audio> tags.
            data_dir (str): name of the dir with all data, default ("data").
        """
        super().__init__()
        self.set_dataset(dataset_names, split, data_dir)
        if limit is not None:
            limit = min(limit, len(self.dataset))
            self.dataset = self.dataset.select(range(limit))
        self.tokenizer = tokenizer
        if not hasattr(self.tokenizer, "pad_token"):
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.feature_extractor = feature_extractor
        self.return_audio = return_audio
        self.name = name
        self.use_explicit_audio_tokens = use_explicit_audio_tokens

    def set_dataset(self, dataset_names, split, data_dir):
        datasets = []
        with_context_datasets = []
        no_context_datasets = []
        for name, use_context, percentage, use_random in dataset_names:
            dataset = load_merged_dataset(name, split, data_dir=data_dir)
            limit = int(len(dataset) * percentage)
            indexes = range(limit)
            if use_random:
                np.random.seed(1)
                indexes = np.random.permutation(len(dataset))
                indexes = indexes[:limit]
                indexes = np.sort(indexes)
            if use_context == "WithContext":
                dataset = dataset.rename_column("llm_answer_with_context", "llm_answer")
                dataset = dataset.select(indexes)
                with_context_datasets.append(dataset)
            elif use_context == "NoContext":
                dataset = dataset.remove_columns("context")
                dataset = dataset.add_column("context", [""] * len(dataset))
                dataset = dataset.rename_column("llm_answer_no_context", "llm_answer")
                dataset = dataset.select(indexes)
                no_context_datasets.append(dataset)
            else:
                raise NotImplementedError()
            datasets.append(dataset)

        self.dataset = concatenate_datasets(datasets)
        if len(with_context_datasets) == 0:
            self.with_context_dataset = concatenate_datasets(no_context_datasets)
        else:
            self.with_context_dataset = concatenate_datasets(with_context_datasets)
        if len(no_context_datasets) == 0:
            self.no_context_dataset = concatenate_datasets(with_context_datasets)
        else:
            self.no_context_dataset = concatenate_datasets(no_context_datasets)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        data_dict = self.dataset[index]

        instance_data = self.extract_data_from_dict(
            data_dict, return_audio=self.return_audio
        )
        instance_data = self.prepare_data(instance_data, return_audio=self.return_audio)
        return instance_data

    def extract_data_from_dict(self, data_dict, return_audio):
        context = data_dict["context"]
        llm_answer = data_dict["llm_answer"]
        audio_description = data_dict.get("audio_description", "")
        reference = data_dict.get("reference", "")

        if not return_audio:
            if self.use_explicit_audio_tokens:
                audio_description_with_tags = (
                    f"<start_audio>{audio_description}<end_audio>"
                )
            else:
                audio_description_with_tags = audio_description
            if context == "":
                context = audio_description_with_tags
            else:
                context = context + "\n" + audio_description_with_tags

        audio = data_dict.get("audio", None)

        return {
            "llm_answer": llm_answer,
            "context": context,
            "audio_description": audio_description,
            "reference": reference,
            "audio": audio,
        }

    def prepare_data(self, instance_data, return_audio):
        audio = instance_data["audio"]
        context = instance_data["context"]
        llm_answer = instance_data["llm_answer"]
        audio_description = instance_data["audio_description"]
        reference = instance_data["reference"]

        inputs = [
            {"role": "user", "content": context},
        ]
        # outputs = [{"role": "assistant", "content": llm_answer}]
        inputs = self.tokenizer.apply_chat_template(
            inputs, add_generation_prompt=True, tokenize=False
        )
        # outputs = self.tokenizer.apply_chat_template(
        #     outputs, add_generation_prompt=False, tokenize=False
        # )
        inputs_start = inputs.split("\n")[0] + "\n"
        inputs_main = inputs[len(inputs_start) :]
        # outputs_start = outputs.split("\n")[0] + "\n"
        # outputs = outputs[len(outputs_start) :]
        # we manually construct output, otherwise <think> is removed
        # we also avoid adding eos_token because the answer might be
        # truncated and we do not want the model to learn truncation
        # it will still output eos when needed because it is frozen LLM
        outputs = llm_answer
        # outputs = llm_answer + self.tokenizer.eos_token + "\n"

        if self.use_explicit_audio_tokens and return_audio:
            inputs_start = inputs_start + "<start_audio>"
            inputs_main = "<end_audio>" + inputs_main

        # inputs start -- <bos>user
        # inputs main -- content <eos>assistant
        # outputs -- assistant_content<eos>

        instance_dict = {
            "inputs_start": inputs_start,
            "inputs_main": inputs_main,
            "outputs": outputs,
            "llm_answer": llm_answer,
            "context": context,
            "audio_description": audio_description,
            "reference": reference,
        }

        if return_audio:
            processed_audio = self.feature_extractor(audio)
            instance_dict.update(**processed_audio)

        return instance_dict
