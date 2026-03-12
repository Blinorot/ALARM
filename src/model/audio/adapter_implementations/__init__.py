from src.model.audio.adapter_implementations.downsampler_conformer import (
    DownsamplerConformerAdapter,
)
from src.model.audio.adapter_implementations.identity import IdentityAdapter
from src.model.audio.adapter_implementations.mlp import MLPAdapter

adapter_model_dict = {
    "identity": IdentityAdapter,
    "mlp": MLPAdapter,
    "downsampler_conformer": DownsamplerConformerAdapter,
}
