import torch
import torchaudio
from torch import nn


class MuQFeatureExtractor:
    def __init__(self, target_sr=24000, max_time=30.0, **kwargs):
        """
        Args:
            target_sr (int): target sampling rate.
            max_time (float): maximum number of seconds. Pads or split to this time.
        """
        self.target_sr = target_sr
        self.max_time = max_time
        self.max_length = int(self.target_sr * self.max_time)

    def __call__(self, audio, sr):
        if sr != self.target_sr:
            audio = torchaudio.transforms.Resample(sr, self.target_sr)(audio)

        orig_len = audio.shape[-1]
        if orig_len % self.max_length == 0:
            target_length = orig_len
        else:
            target_length = (orig_len // self.max_length + 1) * self.max_length
        if orig_len < target_length:
            audio = torch.nn.functional.pad(audio, (0, target_length - orig_len))

        audio = audio[0]  # T

        audio_attention = torch.zeros(target_length, dtype=torch.long)
        audio_attention[:orig_len] = 1
        audio_lengths = torch.tensor(orig_len)

        return audio, audio_attention, audio_lengths
