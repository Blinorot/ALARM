import torch
from transformers import AutoModel

from src.model.audio.encoder_implementations.base import BaseAudioEncoder


class SSLAMEncoder(BaseAudioEncoder):
    def __init__(
        self,
        max_time=10.24,
        frame_rate=100,
        feature_extractor_time_dim=-2,
        **kwargs,
    ):
        super().__init__(max_time, frame_rate, feature_extractor_time_dim)
        self.encoder = AutoModel.from_pretrained(
            "ta012/SSLAM_AS2M_Finetuned",
            trust_remote_code=True,
        )

    def encode(self, input_features, attention_mask):
        hidden_states = self.get_hidden_states(
            x=input_features.unsqueeze(1)  # add channel
        )

        encoded = torch.stack(hidden_states)

        return encoded

    def get_hidden_states(self, x):
        B = x.shape[0]
        x = self.encoder.model.local_encoder(x)
        if self.encoder.model.fixed_positional_encoder is not None:
            x = (
                x
                + self.encoder.model.fixed_positional_encoder(x, None)[
                    :, : x.size(1), :
                ]
            )
        x = torch.cat((self.encoder.model.extra_tokens.expand(B, -1, -1), x), dim=1)
        x = self.encoder.model.pre_norm(x)
        x = self.encoder.model.pos_drop(x)
        hidden_states = []
        for blk in self.encoder.model.blocks:
            x, _ = blk(x)
            hidden_states.append(x[:, 1:])  # remove CLS
        return hidden_states

    def calc_length(self, lengths):
        return lengths // 2
