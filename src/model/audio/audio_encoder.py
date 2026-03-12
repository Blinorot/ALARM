import torch
from torch import nn

from src.model.audio.encoder_implementations import encoder_model_dict


class AudioEncoder(nn.Module):
    def __init__(
        self,
        encoder_configs: list[dict],
    ):
        super().__init__()
        encoders_dict = {}
        for encoder_config in encoder_configs:
            encoder_name = encoder_config["encoder_name"]
            encoder_cls = encoder_model_dict[encoder_name]
            encoder = encoder_cls(**encoder_config)
            encoders_dict[encoder_name] = encoder

        # sort to ensure that the order is fixed
        sorted_encoder_names = sorted(encoders_dict.keys())
        self.encoders = nn.ModuleDict()
        for name in sorted_encoder_names:
            self.encoders[name] = encoders_dict[name]

        self.frozen = False

    def calc_length(self, lengths):
        """
        Args:
            lengths (dict[tensor]): (encoder_name, feature_extractor lengths) dict.
        Returns:
            processed_lengths (dict[tensor]): lengths after encoder.
        """
        processed_lengths = {}
        for encoder_name, encoder in self.encoders.items():
            processed_lengths[encoder_name] = encoder.calc_length(lengths[encoder_name])
        return processed_lengths

    def forward(self, input_features, attention_mask):
        """
        Args:
            input_features (dict[tensor]): (encoder_name, feature_extractor features) dict.
        Returns:
            outputs (dict[tensor]): features after encoder.
        """
        outputs = {}
        for encoder_name, encoder in self.encoders.items():
            if self.frozen:
                with torch.no_grad():
                    encoder_output = encoder(
                        input_features=input_features[encoder_name],
                        attention_mask=attention_mask[encoder_name],
                    )
            else:
                encoder_output = encoder(
                    input_features=input_features[encoder_name],
                    attention_mask=attention_mask[encoder_name],
                )
            outputs[encoder_name] = encoder_output
        return outputs

    def train(self, mode=True):
        super().train(mode)
        # always in the eval mode
        for _, encoder in self.encoders.items():
            encoder.eval()

    def freeze_encoder(self):
        for _, encoder in self.encoders.items():
            encoder.freeze_encoder()
        self.frozen = True
