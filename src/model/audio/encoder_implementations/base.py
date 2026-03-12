import torch
from torch import nn


class BaseAudioEncoder(nn.Module):
    def __init__(
        self,
        max_time=30.0,
        frame_rate=100,
        feature_extractor_time_dim=-2,
        **kwargs,
    ):
        super().__init__()
        self.max_time = max_time
        self.frame_rate = frame_rate
        self.split_time = int(max_time * frame_rate)
        self.feature_extractor_time_dim = feature_extractor_time_dim

        # self.encoder is defined in the subclass

    def encode(self, input_features, attention_mask):
        hidden_states = self.encoder(
            input_features=input_features,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )["hidden_states"]

        encoded = torch.stack(hidden_states)

        return encoded

    def calc_length(self, lengths):
        return lengths

    def forward(self, input_features, attention_mask):
        input_features_list, attention_mask_list = self.get_split_lists(
            input_features=input_features, attention_mask=attention_mask
        )

        outputs_list = []
        id = 0
        for inputs, attns in zip(input_features_list, attention_mask_list):
            outputs = self.encode(inputs, attns)
            outputs_list.append(outputs)
            id += 1
        outputs = torch.cat(outputs_list, dim=-2)
        return outputs

    def get_split_lists(self, input_features, attention_mask):
        if self.feature_extractor_time_dim != -2:
            # the T and F must be swapped
            input_features = input_features.transpose(-1, -2)

        input_features_list = input_features.split(
            self.split_time, dim=self.feature_extractor_time_dim
        )
        attention_mask_list = attention_mask.split(self.split_time, dim=-1)
        return input_features_list, attention_mask_list

    def freeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = False
