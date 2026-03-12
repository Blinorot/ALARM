import torch
import torchaudio
from torch import nn

from src.model.audio.adapter_implementations.base import BaseAudioAdapter


class DownsamplerConformerAdapter(BaseAudioAdapter):
    def __init__(
        self,
        audio_encoder_layers,
        adapter_embedding_dim,
        llm_embedding_dim,
        pre_average,
        trim_extra_padding=True,
        layer_fusion_config=None,
        use_llm_proj=True,
        encoder_embedding_dim=1024,
        downsampler_depth=1,
        conformer_heads=16,
        conf_linear_dims=1024 * 4,
        conf_depth_kernel_size=5,
        conf_dropout=0.1,
        n_conf_layers=1,
        conf_conv_first=False,
        norm_type="layer",
        use_downsampler=True,
        use_conformer=True,
        **kwargs,
    ):
        super().__init__(
            audio_encoder_layers=audio_encoder_layers,
            adapter_embedding_dim=adapter_embedding_dim,
            llm_embedding_dim=llm_embedding_dim,
            pre_average=pre_average,
            trim_extra_padding=trim_extra_padding,
            layer_fusion_config=layer_fusion_config,
            use_llm_proj=use_llm_proj,
        )
        if self.pre_average:
            self.adapter = DownsamplerConformer(
                encoder_embedding_dim=encoder_embedding_dim,
                embedding_dim=adapter_embedding_dim,
                downsampler_depth=downsampler_depth,
                conformer_heads=conformer_heads,
                conf_linear_dims=conf_linear_dims,
                conf_depth_kernel_size=conf_depth_kernel_size,
                conf_dropout=conf_dropout,
                n_conf_layers=n_conf_layers,
                conf_conv_first=conf_conv_first,
                norm_type=norm_type,
                use_downsampler=use_downsampler,
                use_conformer=use_conformer,
            )
        else:
            self.adapters = nn.ModuleList()
            for _ in range(len(self.audio_encoder_layers)):
                adapter = DownsamplerConformer(
                    encoder_embedding_dim=encoder_embedding_dim,
                    embedding_dim=adapter_embedding_dim,
                    downsampler_depth=downsampler_depth,
                    conformer_heads=conformer_heads,
                    conf_linear_dims=conf_linear_dims,
                    conf_depth_kernel_size=conf_depth_kernel_size,
                    conf_dropout=conf_dropout,
                    n_conf_layers=n_conf_layers,
                    conf_conv_first=conf_conv_first,
                    norm_type=norm_type,
                    use_downsampler=use_downsampler,
                    use_conformer=use_conformer,
                )
                self.adapters.append(adapter)

        self.use_downsampler = use_downsampler
        self.length_downsample_factor = 2**downsampler_depth

    def calc_length(self, lengths):
        if self.use_downsampler:
            return lengths // self.length_downsample_factor
        return lengths


class DownsamplerConformer(nn.Module):
    def __init__(
        self,
        encoder_embedding_dim=1024,
        embedding_dim=1024,
        downsampler_depth=1,
        conformer_heads=16,
        conf_linear_dims=1024 * 4,
        conf_depth_kernel_size=5,
        conf_dropout=0.1,
        n_conf_layers=1,
        conf_conv_first=False,
        norm_type="batch",
        use_downsampler=True,
        use_conformer=True,
        **kwargs,
    ):
        super().__init__()

        if encoder_embedding_dim != embedding_dim:
            self.input_proj = nn.Linear(encoder_embedding_dim, embedding_dim)
        else:
            self.input_proj = nn.Identity()

        self.embedding_dim = embedding_dim

        # Downsample: Conv to halve the sequence length
        if use_downsampler:
            if norm_type == "layer":
                norm_cls = TransposeLayerNorm
            elif norm_type == "batch":
                norm_cls = nn.BatchNorm1d
            else:
                raise NotImplementedError
            downsample_list = [
                nn.Sequential(
                    nn.Conv1d(
                        embedding_dim,
                        embedding_dim * 2,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                    ),
                    nn.GELU(),
                    norm_cls(embedding_dim * 2),
                ),
            ]
            for depth in range(downsampler_depth):
                input_channels = embedding_dim * 2 if depth == 0 else embedding_dim
                downsample_list.append(
                    nn.Sequential(
                        nn.Conv1d(
                            input_channels,
                            embedding_dim,
                            kernel_size=4,
                            stride=2,
                            padding=1,
                        ),  # halving
                        nn.GELU(),
                        norm_cls(embedding_dim),
                    )
                )
            self.downsampler = nn.Sequential(*downsample_list)

        # Conformer encoder and decoder
        if use_conformer:
            self.encoder = torchaudio.models.Conformer(
                input_dim=embedding_dim,
                num_heads=conformer_heads,
                ffn_dim=conf_linear_dims,
                num_layers=n_conf_layers,
                depthwise_conv_kernel_size=conf_depth_kernel_size,
                dropout=conf_dropout,
                convolution_first=conf_conv_first,
            )

        self.use_conformer = use_conformer
        self.use_downsampler = use_downsampler

    def forward(self, x):
        x = self.input_proj(x)
        # x: (batch, seq_len, embedding_dim)
        batch_size, seq_len, dim = x.size()

        # Pad to even length
        if seq_len % 2 != 0:
            pad_tensor = torch.zeros(batch_size, 1, dim, device=x.device, dtype=x.dtype)
            x = torch.cat([x, pad_tensor], dim=1)  # Now (batch, seq_len+1, dim)

        # Downsample
        if self.use_downsampler:
            x_ds = self.downsampler(x.transpose(1, 2)).transpose(1, 2)  # (B, seq//2, D)
        else:
            x_ds = x

        # Encode
        if self.use_conformer:
            lengths = torch.full(
                (x.size(0),), x_ds.size(1), device=x.device, dtype=torch.long
            )
            x_enc, _ = self.encoder(x_ds, lengths)  # (B, seq//2, D)
        else:
            x_enc = x_ds

        return x_enc


class TransposeLayerNorm(nn.Module):
    def __init__(
        self,
        normalized_shape,
        eps: float = 1e-5,
        elementwise_affine: bool = True,
        bias: bool = True,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(
            normalized_shape=normalized_shape,
            eps=eps,
            elementwise_affine=elementwise_affine,
            bias=bias,
            device=device,
            dtype=dtype,
        )

    def forward(self, x):
        x = x.transpose(-1, -2)
        x = self.norm(x)
        x = x.transpose(-1, -2)
        return x
