from src.model.audio.fusion_implementations.crossattention import CrossAttentionFusion
from src.model.audio.fusion_implementations.perceiver import (
    MultiPerceiverFusion,
    PerceiverFusion,
)

fusion_model_dict = {
    "crossattention": CrossAttentionFusion,
    "perceiver": PerceiverFusion,
    "multiperceiver": MultiPerceiverFusion,
}
