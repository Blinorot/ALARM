import torch
from torch import nn

from src.model.audio.adapter_implementations.layer_fusion import layer_fusion_model_dict


class BaseAudioAdapter(nn.Module):
    def __init__(
        self,
        audio_encoder_layers,
        adapter_embedding_dim,
        llm_embedding_dim,
        pre_average=True,
        trim_extra_padding=True,
        layer_fusion_config=None,
        use_llm_proj=True,
    ):
        super().__init__()
        self.trim_extra_padding = trim_extra_padding
        self.pre_average = pre_average
        self.audio_encoder_layers = audio_encoder_layers
        if len(self.audio_encoder_layers) > 1:
            assert layer_fusion_config is not None, "Provide layer fusion config"
            layer_fusion_type = layer_fusion_config["layer_fusion_type"]
            layer_fusion_model_cls = layer_fusion_model_dict[layer_fusion_type]
            self.layer_fusion = layer_fusion_model_cls(
                n_layers=len(self.audio_encoder_layers), **layer_fusion_config
            )

        self.use_llm_proj = use_llm_proj
        if self.use_llm_proj:
            self.llm_proj = nn.Linear(adapter_embedding_dim, llm_embedding_dim)

        # self.adapters or self.adapter will be defined in sub-classes

    def forward(self, encoded_layers, lengths):
        # take the subset of layers
        encoded_layers = encoded_layers[self.audio_encoder_layers]

        if self.trim_extra_padding:
            encoded_layers = self.trim_padding_in_encoded_layers(
                encoded_layers=encoded_layers,
                lengths=lengths,
            )

        if self.pre_average:
            encoded_layers = self.fuse_outputs(encoded_layers)
            adapter_output = self.adapter(encoded_layers)
        else:
            adapter_output = []
            for adapter, encoded in zip(self.adapters, encoded_layers):
                adapter_output.append(adapter(encoded))
            adapter_output = torch.stack(adapter_output)
            adapter_output = self.fuse_outputs(adapter_output)

        if self.use_llm_proj:
            adapter_output = self.llm_proj(adapter_output)

        return adapter_output

    def fuse_outputs(self, outputs):
        if len(self.audio_encoder_layers) == 1:
            outputs = outputs[0]
        else:
            outputs = self.layer_fusion(outputs)
        return outputs

    def calc_length(self, lengths):
        return lengths

    def trim_padding_in_encoded_layers(self, encoded_layers, lengths):
        """
        Remove extra padding happening from encoder
        handling only audio of specific length (e.g. for Whisper).

        Padding needed for batch-processing is kept.
        """
        max_length = lengths.max()
        return encoded_layers[:, :, :max_length, :]

    def freeze_adapter(self):
        for param in self.parameters():
            param.requires_grad = False
