import torch
from torch.nn import Module
from torch.nn.functional import log_softmax, softmax, kl_div, cosine_embedding_loss
from stacked.utils import common


class ParameterSimilarityLoss(Module):
    def __init__(self, ):
        super(ParameterSimilarityLoss, self).__init__()

    def forward(self, x, y):
        pass


class FeatureSimilarityLoss(Module):
    def __init__(self):
        super(FeatureSimilarityLoss, self).__init__()
        self.net_runner = None

    def forward(self, x, y):
        common.PREVIOUS_LABEL = common.CURRENT_LABEL
        common.CURRENT_LABEL = y

        features = common.CURRENT_FEATURES
        previous_features = common.PREVIOUS_FEATURES
        loss = 0
        for scope, modules in features.items():
            if scope in previous_features:
                for module_id, feature in modules.items():
                    previous_modules = previous_features[scope]
                    previous_feature = previous_modules.get(module_id, None)
                    difference = similarity_loss(previous_feature, feature)
                    loss = difference + loss
        return loss


def similarity_loss(previous_feature, current_feature):
    x = common.PREVIOUS_LABEL
    y = common.CURRENT_LABEL
    index = x == y
    label = torch.ones(index.size())
    label[index] = -1
    return cosine_embedding_loss(previous_feature, current_feature, label)


def collect_features(scope, module_id, x):
    """Collect current features and replace the previous ones"""
    features = common.CURRENT_FEATURES
    previous_features = common.PREVIOUS_FEATURES

    if scope not in features:
        features[scope] = dict()

    features = features[scope]

    if scope not in previous_features:
        previous_features[scope] = dict()

    previous_features[scope][module_id] = features.get(module_id, None)

    features[module_id] = x

