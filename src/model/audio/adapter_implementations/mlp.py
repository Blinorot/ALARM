from torch import nn

from src.model.audio.adapter_implementations.base import BaseAudioAdapter


class MLPAdapter(BaseAudioAdapter):
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
        hidden_dim=1024,
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
            self.adapter = MLP(
                encoder_embedding_dim=encoder_embedding_dim,
                hidden_dim=hidden_dim,
            )
        else:
            self.adapters = nn.ModuleList()
            for _ in range(len(self.audio_encoder_layers)):
                adapter = MLP(
                    encoder_embedding_dim=encoder_embedding_dim,
                    hidden_dim=hidden_dim,
                )
                self.adapters.append(adapter)


class MLP(nn.Module):
    def __init__(self, encoder_embedding_dim=1024, hidden_dim=1024):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(encoder_embedding_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )

    def forward(self, x):
        return self.model(x)
