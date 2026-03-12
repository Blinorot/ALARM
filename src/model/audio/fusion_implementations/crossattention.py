import torch
import torch.nn.functional as F
from torch import nn


class CrossAttentionFusion(nn.Module):
    def __init__(
        self,
        llm_embedding_dim=1024,
        hidden_dim=1024,
        ffn_dims=1024 * 4,
        dropout=0.1,
        num_attention_heads=16,
        num_hidden_layers=[2, 2, 2],
        main_encoder="whisper",
        conditional_encoders=["w2vbert", "muq", "sslam"],
        use_skip_connection=False,
        **kwargs
    ):
        super().__init__()

        self.main_encoder = main_encoder
        self.conditional_encoders = conditional_encoders
        self.condition_modules = nn.ModuleDict()
        for encoder_name, encoder_hidden_layers in zip(
            conditional_encoders, num_hidden_layers
        ):
            decoder = nn.ModuleList()
            for _ in range(encoder_hidden_layers):
                decoder.append(
                    nn.TransformerDecoderLayer(
                        d_model=hidden_dim,
                        nhead=num_attention_heads,
                        dim_feedforward=ffn_dims,
                        dropout=dropout,
                        activation=nn.functional.gelu,
                        norm_first=True,
                        batch_first=True,
                    )
                )
            self.condition_modules[encoder_name] = decoder

        self.in_proj = nn.ModuleDict()
        for encoder_name in [main_encoder] + conditional_encoders:
            self.in_proj[encoder_name] = nn.Linear(llm_embedding_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, llm_embedding_dim)
        self.use_skip_connection = use_skip_connection

    def forward(self, adapter_outputs, lengths):
        main_output = adapter_outputs[self.main_encoder]
        main_output = self.in_proj[self.main_encoder](main_output)
        main_padding_mask = self.get_padding_mask(
            main_output, lengths[self.main_encoder]
        )

        processed_output = main_output

        for encoder_name in self.conditional_encoders:
            condition_module = self.condition_modules[encoder_name]
            cond_output = adapter_outputs[encoder_name]
            cond_output = self.in_proj[encoder_name](cond_output)
            cond_padding_mask = self.get_padding_mask(
                cond_output, lengths[self.main_encoder]
            )
            for condition_layer in condition_module:
                processed_output = condition_layer(
                    tgt=processed_output,
                    memory=cond_output,
                    tgt_key_padding_mask=main_padding_mask,
                    memory_key_padding_mask=cond_padding_mask,
                    tgt_is_causal=False,
                    memory_is_causal=False,
                )

        if self.use_skip_connection:
            processed_output = processed_output + main_output

        main_output = self.out_proj(processed_output)

        return main_output

    def get_padding_mask(self, tensor, length):
        padding_mask = torch.arange(tensor.shape[-2], device=tensor.device)
        padding_mask = padding_mask[None, :] >= length[:, None]  # True = mask out
        return padding_mask

    def freeze_fusion(self):
        for param in self.parameters():
            param.requires_grad = False
