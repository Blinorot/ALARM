import torch
from torch import nn

from src.model.audio.fusion_implementations import fusion_model_dict


class AudioFusion(nn.Module):
    def __init__(self, fusion_config):
        super().__init__()
        self.fusion_type = fusion_config["fusion_type"]
        if self.fusion_type != "sum":
            fusion_model_cls = fusion_model_dict[self.fusion_type]
            self.fusion_model = fusion_model_cls(**fusion_config)
        else:
            self.fusion_model = None

    def forward(self, adapter_outputs, lengths):
        adapter_outputs = self.pad_outputs(adapter_outputs)
        if self.fusion_type == "sum":
            result = 0
            for _, adapter_output in adapter_outputs.items():
                result += adapter_output
        else:
            result = self.fusion_model(adapter_outputs, lengths)
        return result

    def calc_length(self, lengths):
        """
        Calculated the unpadded lengths after fusion.
        Current implementation assumes that all audio encoders have the same
        frame rate after adapters. So lengths are the same for all encoders.
        But due to slightly different padding, rarely some encoders may have
        length which is less by 1, so we take max length.

        Args:
            lengths (dict[tensor]): (encoder_name, adapter lengths) dict.
        Returns:
            processed_lengths (tensor): lengths after fusion.
        """
        if self.fusion_model is not None and hasattr(self.fusion_model, "calc_length"):
            return self.fusion_model.calc_length(lengths)
        concat_list = []
        for _, adapter_length in lengths.items():
            concat_list.append(adapter_length)
        # N = number of encoders
        # N = len(concat_list)
        concat_tensor = torch.stack(concat_list, dim=-1)  # B x N
        lengths = concat_tensor.max(dim=-1).values
        return lengths

    def pad_outputs(self, outputs):
        outputs_dict = {}
        max_length = 0
        for k, output in outputs.items():
            # output (B x T_i x D)
            max_length = max(output.shape[-2], max_length)
        for k, output in outputs.items():
            pad_length = max_length - output.shape[-2]
            padded_output = nn.functional.pad(output, pad=(0, 0, 0, pad_length))
            outputs_dict[k] = padded_output
        return outputs_dict

    def freeze_fusion(self):
        if self.fusion_model is not None:
            self.fusion_model.freeze_fusion()
