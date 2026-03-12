import torch
import torchaudio
from torch import nn
from transformers import AutoFeatureExtractor


class W2VBERT2FeatureExtractor:
    def __init__(self, target_sr=16000, max_time=30.0, **kwargs):
        """
        Args:
            target_sr (int): target sampling rate.
            max_time (float): max time in seconds for an audio input. The audio will be split
                if it is longer.
        """
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(
            "facebook/w2v-bert-2.0"
        )
        self.target_sr = target_sr
        self.max_time = max_time
        self.split_time = int(self.max_time * self.target_sr)

    def __call__(self, audio, sr):
        if sr != self.target_sr:
            audio = torchaudio.transforms.Resample(sr, self.target_sr)(audio)

        audio = nn.functional.pad(audio, (160, 160))
        audio_list = [audio[0].numpy()]

        audio_dict = self.feature_extractor(
            audio_list,
            sampling_rate=self.target_sr,
            return_attention_mask=True,
            return_tensors="pt",
        )

        audio = torch.cat(
            [elem for elem in audio_dict["input_features"]],
            dim=-2,
        )
        audio_attention = torch.cat(
            [elem for elem in audio_dict["attention_mask"]], dim=-1
        )
        audio_lengths = audio_attention.sum()

        return audio, audio_attention, audio_lengths
