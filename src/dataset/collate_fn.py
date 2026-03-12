from collections import defaultdict
from dataclasses import dataclass

import torch
from transformers import PreTrainedTokenizer


@dataclass
class DataCollator:
    tokenizer: PreTrainedTokenizer

    def __call__(self, data_items: list[dict]):
        inputs_start = []
        inputs_main = []
        outputs = []
        rejected_outputs = []
        audio_lengths = defaultdict(list)
        audio = defaultdict(list)
        audio_attention = defaultdict(list)
        audio_description = []
        reference = []
        context = []
        has_audio = "audio" in data_items[0].keys()
        has_outputs = "outputs" in data_items[0].keys()
        has_rejected_outputs = "rejected_outputs" in data_items[0].keys()
        for elem in data_items:
            if has_audio:
                for encoder_name in elem["audio"].keys():
                    audio[encoder_name].append(elem["audio"][encoder_name])
                    audio_lengths[encoder_name].append(
                        elem["audio_lengths"][encoder_name]
                    )
                    audio_attention[encoder_name].append(
                        elem["audio_attention"][encoder_name]
                    )

            inputs_start.append(elem["inputs_start"])
            inputs_main.append(elem["inputs_main"])
            if has_outputs:
                outputs.append(elem["outputs"])
            if has_rejected_outputs:
                rejected_outputs.append(elem["rejected_outputs"])

            audio_description.append(elem.get("audio_description", ""))
            reference.append(elem.get("reference", ""))
            context.append(elem.get("context", ""))

        inputs_start = self.tokenizer(
            inputs_start, padding=False, truncation=False, add_special_tokens=False
        ).input_ids
        inputs_main = self.tokenizer(
            inputs_main, padding=False, truncation=False, add_special_tokens=False
        ).input_ids

        batch_size = len(inputs_start)
        inputs_start = [
            torch.tensor(inputs_start[i], dtype=torch.long) for i in range(batch_size)
        ]
        inputs_main = [
            torch.tensor(inputs_main[i], dtype=torch.long) for i in range(batch_size)
        ]

        if has_outputs:
            outputs = self.tokenizer(
                outputs, padding=False, truncation=False, add_special_tokens=False
            ).input_ids
            outputs = [
                torch.tensor(outputs[i], dtype=torch.long) for i in range(batch_size)
            ]
        else:
            outputs = None

        if has_rejected_outputs:
            rejected_outputs = self.tokenizer(
                rejected_outputs,
                padding=False,
                truncation=False,
                add_special_tokens=False,
            ).input_ids
            rejected_outputs = [
                torch.tensor(rejected_outputs[i], dtype=torch.long)
                for i in range(batch_size)
            ]
        else:
            rejected_outputs = None

        # if no audio
        # if audio -- used as placeholder
        # labels will be overwritten by model forward
        labels = []
        for i in range(batch_size):
            len_input = inputs_start[i].shape[0] + inputs_main[i].shape[0]
            input_label = torch.tensor([-100] * len_input, dtype=torch.long)
            if has_outputs:
                output_label = outputs[i].clone()
                labels.append(torch.cat([input_label, output_label], dim=0))
            else:
                labels.append(input_label)

        collate_dict = {
            "inputs_start": inputs_start,
            "inputs_main": inputs_main,
            "outputs": outputs,
            "rejected_outputs": rejected_outputs,
            "labels": labels,  # placeholder,
            "audio_description": audio_description,
            "context": context,
            "reference": reference,
        }

        if has_audio:
            processed_audio = {}
            processed_attention = {}
            processed_lengths = {}

            for encoder_name in audio.keys():
                processed_lengths[encoder_name] = torch.tensor(
                    audio_lengths[encoder_name], dtype=torch.long
                )
                processed_audio[encoder_name] = torch.nn.utils.rnn.pad_sequence(
                    audio[encoder_name], batch_first=True
                )
                processed_attention[encoder_name] = torch.nn.utils.rnn.pad_sequence(
                    audio_attention[encoder_name], batch_first=True
                )

            collate_dict.update(
                **{
                    "audio_attention": processed_attention,
                    "audio": processed_audio,
                    "audio_lengths": processed_lengths,
                }
            )

        return collate_dict
