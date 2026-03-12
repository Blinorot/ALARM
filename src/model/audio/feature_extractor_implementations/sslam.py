import torch
import torchaudio
from torch import nn


class SSLAMFeatureExtractor:
    def __init__(
        self,
        target_sr=16000,
        max_frames=1024,
        norm_mean=-4.268,
        norm_std=4.569,
        **kwargs
    ):
        """
        Args:
            target_sr (int): target sampling rate.
            max_frames (int): max number of frames per audio. Pads or split to
                this number.
            norm_mean (float): mean for mel normalization.
            norm_std (float): std for mel normalization.
        """
        self.target_sr = target_sr
        self.max_frames = max_frames
        # https://huggingface.co/ta012/SSLAM_AS2M_Finetuned
        self.norm_mean = norm_mean
        self.norm_std = norm_std

    def __call__(self, audio, sr):
        if sr != self.target_sr:
            audio = torchaudio.transforms.Resample(sr, self.target_sr)(audio)

        audio = audio - audio.mean()

        audio = nn.functional.pad(audio, (160, 160))
        audio = torchaudio.compliance.kaldi.fbank(
            audio,
            htk_compat=True,
            sample_frequency=self.target_sr,
            use_energy=False,
            window_type="hanning",
            num_mel_bins=128,
            dither=0.0,
            frame_shift=10,
        ).unsqueeze(0)

        # Pad or truncate
        n_frames = audio.shape[1]
        if n_frames % self.max_frames == 0:
            target_length = n_frames
        else:
            target_length = (n_frames // self.max_frames + 1) * self.max_frames

        if n_frames < target_length:
            audio = torch.nn.functional.pad(audio, (0, 0, 0, target_length - n_frames))

        # Normalize
        audio = (audio - self.norm_mean) / (self.norm_std * 2)
        audio = audio[0]  # T x F

        audio_attention = torch.zeros(target_length, dtype=torch.long)
        audio_attention[:n_frames] = 1
        audio_lengths = torch.tensor(n_frames)

        return audio, audio_attention, audio_lengths
