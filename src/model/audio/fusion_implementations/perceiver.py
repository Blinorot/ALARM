from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

from src.model.audio.fusion_implementations.helpers.perceiver.adapter import (
    InputAdapter,
)
from src.model.audio.fusion_implementations.helpers.perceiver.modules import (
    PerceiverEncoder,
)


class IdentityInputAdapter(InputAdapter):
    def forward(self, x):
        return x


class PerceiverFusion(nn.Module):
    def __init__(
        self,
        llm_embedding_dim: int = 1024,
        num_latents: int = 24,
        num_latent_channels: int = 1024,
        num_cross_attention_heads: int = 4,
        num_cross_attention_qk_channels: Optional[int] = None,
        num_cross_attention_v_channels: Optional[int] = None,
        num_cross_attention_layers: int = 1,
        first_cross_attention_layer_shared: bool = False,
        cross_attention_widening_factor: int = 1,
        num_self_attention_heads: int = 4,
        num_self_attention_qk_channels: Optional[int] = None,
        num_self_attention_v_channels: Optional[int] = None,
        num_self_attention_layers_per_block: int = 6,
        num_self_attention_blocks: int = 1,
        first_self_attention_block_shared: bool = True,
        self_attention_widening_factor: int = 1,
        dropout: float = 0.0,
        residual_dropout: float = 0.0,
        init_scale: float = 0.02,
        activation_checkpointing: bool = False,
        activation_offloading: bool = False,
        main_encoder="whisper",
        conditional_encoders=["w2vbert", "muq", "sslam"],
        **kwargs
    ):
        super().__init__()

        self.main_encoder = main_encoder
        self.conditional_encoders = conditional_encoders

        self.num_latents = num_latents
        self.perceiver = PerceiverEncoder(
            # our input is already adapted
            input_adapter=IdentityInputAdapter(num_input_channels=llm_embedding_dim),
            num_latents=num_latents,
            num_latent_channels=num_latent_channels,
            num_cross_attention_heads=num_cross_attention_heads,
            num_cross_attention_qk_channels=num_cross_attention_qk_channels,
            num_cross_attention_v_channels=num_cross_attention_v_channels,
            num_cross_attention_layers=num_cross_attention_layers,
            first_cross_attention_layer_shared=first_cross_attention_layer_shared,
            cross_attention_widening_factor=cross_attention_widening_factor,
            num_self_attention_heads=num_self_attention_heads,
            num_self_attention_qk_channels=num_self_attention_qk_channels,
            num_self_attention_v_channels=num_self_attention_v_channels,
            num_self_attention_layers_per_block=num_self_attention_layers_per_block,
            num_self_attention_blocks=num_self_attention_blocks,
            first_self_attention_block_shared=first_self_attention_block_shared,
            self_attention_widening_factor=self_attention_widening_factor,
            dropout=dropout,
            residual_dropout=residual_dropout,
            init_scale=init_scale,
            activation_checkpointing=activation_checkpointing,
            activation_offloading=activation_offloading,
        )

        self.out_proj = nn.Linear(num_latent_channels, llm_embedding_dim)
        self.cond_start_emb = nn.Parameter(torch.randn((1, llm_embedding_dim)))
        self.cond_end_emb = nn.Parameter(torch.randn((1, llm_embedding_dim)))

    def forward(self, adapter_outputs, lengths):
        main_output = adapter_outputs[self.main_encoder]

        joint_cond_output = []
        joint_pad_mask = []

        for encoder_name in self.conditional_encoders:
            cond_output = adapter_outputs[encoder_name]
            cond_padding_mask = self.get_padding_mask(
                cond_output, lengths[self.main_encoder]
            )
            joint_cond_output.append(cond_output)
            joint_pad_mask.append(cond_padding_mask)

        joint_cond_output = torch.cat(joint_cond_output, dim=1)  # B x (N * L) x D
        joint_pad_mask = torch.cat(joint_pad_mask, dim=1)  # B x (N * L)

        latent_output = self.perceiver(
            joint_cond_output, pad_mask=joint_pad_mask, return_adapted_input=False
        )
        latent_output = self.out_proj(latent_output)

        B, _, D = latent_output.shape

        main_output = torch.cat(
            [
                self.cond_start_emb.unsqueeze(0).expand(B, 1, D),
                latent_output,
                self.cond_end_emb.unsqueeze(0).expand(B, 1, D),
                main_output,
            ],
            dim=1,
        )

        return main_output

    def get_padding_mask(self, tensor, length):
        padding_mask = torch.arange(tensor.shape[-2], device=tensor.device)
        padding_mask = padding_mask[None, :] >= length[:, None]  # True = mask out
        return padding_mask

    def calc_length(self, lengths):
        """
        Calculated the unpadded lengths after fusion.
        Takes the main encoder lengths and adds num_latents.

        Args:
            lengths (dict[tensor]): (encoder_name, adapter lengths) dict.
        Returns:
            processed_lengths (tensor): lengths after fusion.
        """
        main_length = lengths[self.main_encoder]
        processed_lengths = main_length + self.num_latents + 2  # 2 separation embed
        return processed_lengths


class MultiPerceiverFusion(nn.Module):
    def __init__(
        self,
        llm_embedding_dim: int = 1024,
        num_latents: list[int] = [20, 20, 20],
        num_latent_channels: list[int] = [1024, 1024, 1024],
        num_cross_attention_heads: list[int] = [4, 4, 4],
        num_cross_attention_qk_channels: list[Optional[int]] = [None, None, None],
        num_cross_attention_v_channels: list[Optional[int]] = [None, None, None],
        num_cross_attention_layers: list[int] = [1, 1, 1],
        first_cross_attention_layer_shared: list[bool] = [False, False, False],
        cross_attention_widening_factor: list[int] = [1, 1, 1],
        num_self_attention_heads: list[int] = [4, 4, 4],
        num_self_attention_qk_channels: list[Optional[int]] = [None, None, None],
        num_self_attention_v_channels: list[Optional[int]] = [None, None, None],
        num_self_attention_layers_per_block: list[int] = [6, 6, 6],
        num_self_attention_blocks: list[int] = [1, 1, 1],
        first_self_attention_block_shared: list[bool] = [True, True, True],
        self_attention_widening_factor: list[int] = [1, 1, 1],
        dropout: list[float] = [0.0, 0.0, 0.0],
        residual_dropout: list[float] = [0.0, 0.0, 0.0],
        init_scale: list[float] = [0.02, 0.02, 0.02],
        activation_checkpointing: list[bool] = [False, False, False],
        activation_offloading: list[bool] = [False, False, False],
        main_encoder="whisper",
        conditional_encoders=["w2vbert", "muq", "sslam"],
        conditional_embedding_dim: list[int] = [1024, 1024, 768],
        **kwargs
    ):
        super().__init__()

        self.main_encoder = main_encoder
        self.conditional_encoders = conditional_encoders

        self.num_latents = num_latents
        self.total_num_latents = sum(self.num_latents)
        self.perceiver = nn.ModuleDict()
        self.out_proj = nn.ModuleDict()
        for i in range(len(self.conditional_encoders)):
            encoder_name = self.conditional_encoders[i]
            self.perceiver[encoder_name] = PerceiverEncoder(
                # our input is already adapted
                input_adapter=IdentityInputAdapter(
                    num_input_channels=conditional_embedding_dim[i]
                ),
                num_latents=num_latents[i],
                num_latent_channels=num_latent_channels[i],
                num_cross_attention_heads=num_cross_attention_heads[i],
                num_cross_attention_qk_channels=num_cross_attention_qk_channels[i],
                num_cross_attention_v_channels=num_cross_attention_v_channels[i],
                num_cross_attention_layers=num_cross_attention_layers[i],
                first_cross_attention_layer_shared=first_cross_attention_layer_shared[
                    i
                ],
                cross_attention_widening_factor=cross_attention_widening_factor[i],
                num_self_attention_heads=num_self_attention_heads[i],
                num_self_attention_qk_channels=num_self_attention_qk_channels[i],
                num_self_attention_v_channels=num_self_attention_v_channels[i],
                num_self_attention_layers_per_block=num_self_attention_layers_per_block[
                    i
                ],
                num_self_attention_blocks=num_self_attention_blocks[i],
                first_self_attention_block_shared=first_self_attention_block_shared[i],
                self_attention_widening_factor=self_attention_widening_factor[i],
                dropout=dropout[i],
                residual_dropout=residual_dropout[i],
                init_scale=init_scale[i],
                activation_checkpointing=activation_checkpointing[i],
                activation_offloading=activation_offloading[i],
            )
            self.out_proj[encoder_name] = nn.Linear(
                num_latent_channels[i], llm_embedding_dim
            )

        self.cond_start_emb = nn.Parameter(torch.randn((1, llm_embedding_dim)))
        self.cond_end_emb = nn.Parameter(torch.randn((1, llm_embedding_dim)))
        self.frozen = False

    def forward(self, adapter_outputs, lengths):
        main_output = adapter_outputs[self.main_encoder]

        if self.frozen:
            with torch.no_grad():
                cond_latents_list = self.get_cond_latents_list(adapter_outputs, lengths)
        else:
            cond_latents_list = self.get_cond_latents_list(adapter_outputs, lengths)

        latent_output = torch.cat(cond_latents_list, dim=1)  # B x (N * n_latents) x D

        B, _, D = latent_output.shape

        main_output = torch.cat(
            [
                self.cond_start_emb.unsqueeze(0).expand(B, 1, D),
                latent_output,
                self.cond_end_emb.unsqueeze(0).expand(B, 1, D),
                main_output,
            ],
            dim=1,
        )

        return main_output, None

    def get_cond_latents_list(self, adapter_outputs, lengths):
        cond_latents_list = []
        for encoder_name in self.conditional_encoders:
            cond_output = adapter_outputs[encoder_name]
            cond_padding_mask = self.get_padding_mask(
                cond_output, lengths[encoder_name]
            )
            cond_latents = self.perceiver[encoder_name](
                cond_output, pad_mask=cond_padding_mask, return_adapted_input=False
            )
            cond_latents = self.out_proj[encoder_name](cond_latents)
            cond_latents_list.append(cond_latents)
        return cond_latents_list

    def get_padding_mask(self, tensor, length):
        padding_mask = torch.arange(tensor.shape[-2], device=tensor.device)
        padding_mask = padding_mask[None, :] >= length[:, None]  # True = mask out
        return padding_mask

    def calc_length(self, lengths):
        """
        Calculated the unpadded lengths after fusion.
        Takes the main encoder lengths and adds num_latents.

        Args:
            lengths (dict[tensor]): (encoder_name, adapter lengths) dict.
        Returns:
            processed_lengths (tensor): lengths after fusion.
        """
        main_length = lengths[self.main_encoder]
        processed_lengths = (
            main_length + self.total_num_latents + 2
        )  # 2 separation embed\
        return processed_lengths

    def freeze_fusion(self):
        for param in self.parameters():
            param.requires_grad = False

        self.frozen = True
