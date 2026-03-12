import torch
from torch import nn

from src.model.audio.adapter_implementations import adapter_model_dict


class AudioAdapter(nn.Module):
    def __init__(
        self,
        adapter_configs: list[dict],
    ):
        super().__init__()
        adapters_dict = {}
        for adapter_config in adapter_configs:
            encoder_name = adapter_config["encoder_name"]
            adapter_name = adapter_config["adapter_name"]
            adapter_cls = adapter_model_dict[adapter_name]
            adapter = adapter_cls(**adapter_config)
            adapters_dict[encoder_name] = adapter

        # sort to ensure that the order is fixed
        sorted_encoder_names = sorted(adapters_dict.keys())
        self.adapters = nn.ModuleDict()
        for name in sorted_encoder_names:
            self.adapters[name] = adapters_dict[name]

        self.frozen = {name: False for name in sorted_encoder_names}

    def calc_length(self, lengths):
        """
        Args:
            lengths (dict[tensor]): (encoder_name, encoder lengths) dict.
        Returns:
            processed_lengths (dict[tensor]): lengths after adapter.
        """
        processed_lengths = {}
        for encoder_name, adapter in self.adapters.items():
            processed_lengths[encoder_name] = adapter.calc_length(lengths[encoder_name])
        return processed_lengths

    def forward(self, encoded_layers, lengths):
        """
        Args:
            encoded_layers (dict[tensor]): (encoder_name, encoder features) dict.
            text_cond_emb (Tensor): BxLxD text conditional embedding.
            lengths (dict[tensor]): lengths of encoded tensors. Used to trim extra padding.
        Returns:
            outputs (dict[tensor]): features after adapter.
        """
        outputs = {}
        for encoder_name, adapter in self.adapters.items():
            if self.frozen[encoder_name]:
                with torch.no_grad():
                    adapter_output = adapter(
                        encoded_layers=encoded_layers[encoder_name],
                        lengths=lengths[encoder_name],
                    )
            else:
                adapter_output = adapter(
                    encoded_layers=encoded_layers[encoder_name],
                    lengths=lengths[encoder_name],
                )
            outputs[encoder_name] = adapter_output
        return outputs

    def freeze_adapter(self, frozen_adapters_names):
        for encoder_name, adapter in self.adapters.items():
            if frozen_adapters_names is None or encoder_name in frozen_adapters_names:
                adapter.freeze_adapter()
                print(f"{encoder_name} adapter is frozen")
                self.frozen[encoder_name] = True
