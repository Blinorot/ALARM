from transformers import AutoModel

from src.model.audio.encoder_implementations.base import BaseAudioEncoder


class W2VBERT2Encoder(BaseAudioEncoder):
    def __init__(
        self,
        max_time=30.0,
        frame_rate=50,
        feature_extractor_time_dim=-2,
        **kwargs,
    ):
        super().__init__(max_time, frame_rate, feature_extractor_time_dim)
        self.encoder = AutoModel.from_pretrained("facebook/w2v-bert-2.0")
