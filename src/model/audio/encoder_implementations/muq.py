import torch
from muq import MuQ
from transformers import AutoModel

from src.model.audio.encoder_implementations.base import BaseAudioEncoder


class MuQEncoder(BaseAudioEncoder):
    def __init__(
        self,
        max_time=30.0,
        frame_rate=25,
        feature_extractor_time_dim=-1,
        sample_rate=24000,
        **kwargs,
    ):
        super().__init__(max_time, frame_rate, feature_extractor_time_dim)
        self.sample_rate = sample_rate
        self.length_factor = self.sample_rate // self.frame_rate
        self.split_time = int(self.max_time * self.sample_rate)
        self.encoder = MuQ.from_pretrained("OpenMuQ/MuQ-large-msd-iter")

    def encode(self, input_features, attention_mask):
        hidden_states = self.encoder(
            input_features,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )["hidden_states"]

        encoded = torch.stack(hidden_states)

        return encoded

    def get_split_lists(self, input_features, attention_mask):
        input_features_list = input_features.split(
            self.split_time, dim=self.feature_extractor_time_dim
        )
        attention_mask_list = attention_mask.split(self.split_time, dim=-1)
        return input_features_list, attention_mask_list

    def calc_length(self, lengths):
        return lengths // self.length_factor
