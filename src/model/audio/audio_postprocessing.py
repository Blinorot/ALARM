from torch import nn


class AudioPostprocessing(nn.Module):
    def __init__(self, postprocessing_config):
        super().__init__()
        self.postprocessing_type = postprocessing_config["postprocessing_type"]
        if self.postprocessing_type == "identity":
            self.postprocessor = nn.Identity()

    def forward(self, x):
        return self.postprocessor(x)

    def calc_length(self, lengths):
        if self.postprocessing_type == "identity":
            return lengths
        return self.postprocessor.calc_lengths(lengths)
