import torch
from torch.nn import Module
from torch.nn.functional import cosine_embedding_loss, cross_entropy
from stacked.utils import common, transformer
import numpy as np


class FeatureSimilarityLoss(Module):
    def __init__(self, default_loss=cross_entropy):
        super(FeatureSimilarityLoss, self).__init__()
        self.default_loss = default_loss

    def forward(self, x, y):
        return FeatureSimilarityLoss.function(x, y, self.default_loss)

    @staticmethod
    def get_scalar():
        return 0.004

    @staticmethod
    def function(x, y, default_loss=cross_entropy):
        loss = default_loss(x, y)
        if not common.TRAIN:
            return loss

        common.PREVIOUS_LABEL = common.CURRENT_LABEL
        common.CURRENT_LABEL = y

        features = common.CURRENT_FEATURES
        previous_features = common.PREVIOUS_FEATURES

        for scope, modules in features.items():
            if scope in previous_features:
                for module_id, feature in modules.items():
                    previous_modules = previous_features[scope]
                    previous_feature = previous_modules.get(module_id, None)

                    if previous_feature is None:
                        continue

                    difference = get_feature_similarity_loss(previous_feature, feature)
                    loss += difference * FeatureSimilarityLoss.get_scalar()

        return loss


def get_feature_similarity_loss(previous_feature, current_feature):
    """Penalize features that are not similar when the labels are the same"""
    x = common.PREVIOUS_LABEL
    y = common.CURRENT_LABEL

    if x is None or x.size() != y.size():
        return 0

    index = x != y
    label = torch.ones(index.size()).cuda()
    label[index] = -1
    second_dim = previous_feature.size()[1:]

    view = 1
    for d in second_dim:
        view *= d
    view = (previous_feature.size()[0], view)
    return cosine_embedding_loss(previous_feature.view(view),
                                 current_feature.view(view), label)


def collect_features(scope, module_id, x):
    """Collect current features and replace the previous ones"""
    if not common.TRAIN:
        return

    features = common.CURRENT_FEATURES
    previous_features = common.PREVIOUS_FEATURES

    if scope not in features:
        features[scope] = dict()

    features = features[scope]

    if scope not in previous_features:
        previous_features[scope] = dict()

    previous_features = previous_features[scope]

    if module_id in previous_features:
        if previous_features[module_id] is not None:
            previous_features[module_id].detach()

    previous_features[module_id] = features.get(module_id, None)

    features[module_id] = x


class ParameterSimilarityLoss(Module):
    def __init__(self, default_loss=FeatureSimilarityLoss.function):
        super(ParameterSimilarityLoss, self).__init__()
        self.default_loss = default_loss
        self.engine = None
        self.epoch = 1

    def get_current_parameters(self):
        """Get a dictionary of parameters according to shape"""
        param_dict = {str(v.size()): [] for v in self.engine.net.parameters()}
        for k, v in self.engine.net.named_parameters():
            # criterion on conv weights only
            if k.endswith('conv.weight'):
                param_dict[str(v.size())].append(v)
        return param_dict

    @staticmethod
    def get_scalar():
        return 0.1

    def forward(self, x, y):
        loss = self.default_loss(x, y)

        if not common.TRAIN:
            return loss

        scalar = self.get_scalar()
        param_dict = self.get_current_parameters()
        for params in param_dict.values():
            if len(params) > 1:
                combinations = transformer.list_to_pairs(params)
                for param1, param2 in combinations:
                    if np.random.random() < 0.5:
                        loss += get_parameter_similarity(param1, param2) * scalar

        return loss


def get_parameter_similarity(param1, param2):
    second_dim = param1.size()[1:]

    view = 1
    for d in second_dim:
        view *= d
    view = (param1.size()[0], view)
    return torch.nn.functional.mse_loss(param1.view(view),
                                        param2.view(view))