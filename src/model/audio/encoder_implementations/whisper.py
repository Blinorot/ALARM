from transformers import AutoModel

from src.model.audio.encoder_implementations.base import BaseAudioEncoder


class WhisperEncoder(BaseAudioEncoder):
    def __init__(
        self,
        max_time=30.0,
        frame_rate=100,
        feature_extractor_time_dim=-1,
        **kwargs,
    ):
        super().__init__(max_time, frame_rate, feature_extractor_time_dim)
        self.encoder = AutoModel.from_pretrained(
            "openai/whisper-large-v3"
        ).get_encoder()

    def calc_length(self, lengths):
        return lengths // 2
