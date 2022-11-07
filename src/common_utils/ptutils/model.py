import torch
from torch import nn


def model_device(model):
    return next(model.parameters()).device

class ModelBase(nn.Module):
    def __init__(self):
        super().__init__()

    def save(self, path, config=None):
        to_save = {}
        to_save["state_dict"] = self.state_dict()
        if config is not None:
            to_save["config"] = config
        torch.save(to_save, path)

    def load(self, path):
        data = torch.load(path, map_location=lambda storage, loc: storage)
        self.load_state_dict(data["state_dict"])

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def device(self):
        return model_device(self)