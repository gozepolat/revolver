import torch
from torch.nn import Module


class ParameterSimilarityLoss(Module):
    def __init__(self, ):
        super(ParameterSimilarityLoss, self).__init__()

    def forward(self, x, y):
       pass


class FeatureSimilarityLoss(Module):
    def __init__(self):
        super(FeatureSimilarityLoss, self).__init__()
        self.net = None

    def forward(self, x, y):
        pass
