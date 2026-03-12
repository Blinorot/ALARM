from torch import nn

from src.model.audio.adapter_implementations.base import BaseAudioAdapter


class IdentityAdapter(BaseAudioAdapter):
    def __init__(
        self,
        audio_encoder_layers,
        adapter_embedding_dim,
        llm_embedding_dim,
        pre_average,
        trim_extra_padding=True,
        layer_fusion_config=None,
        use_llm_proj=True,
        **kwargs
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
            self.adapter = Identity()
        else:
            self.adapters = nn.ModuleList()
            for _ in range(len(self.audio_encoder_layers)):
                adapter = Identity()
                self.adapters.append(adapter)


class Identity(nn.Module):
    def forward(self, x):
        return x
