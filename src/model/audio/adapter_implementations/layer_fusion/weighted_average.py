import torch
from torch import nn


class WeightedAverageLayerFusion(nn.Module):
    def __init__(self, n_layers, **kwargs):
        super().__init__()
        self.layer_weights = nn.Parameter(torch.zeros(n_layers, dtype=torch.float32))

    def forward(self, outputs):
        weights = nn.functional.softmax(self.layer_weights, dim=0)
        ndims = outputs.ndim
        extra_dims = [1] * (ndims - 1)
        weights = weights.view(weights.shape[0], *extra_dims)
        outputs = outputs * weights
        outputs = outputs.sum(dim=0)
        return outputs
