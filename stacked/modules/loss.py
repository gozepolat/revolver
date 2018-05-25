import torch
from torch.nn import Module
from torch.nn.functional import cosine_embedding_loss, cross_entropy
from stacked.utils import common


class ParameterSimilarityLoss(Module):
    def __init__(self, default_loss=cross_entropy):
        super(ParameterSimilarityLoss, self).__init__()
        self.default_loss = default_loss
        self.net = None

    def forward(self, x, y):
        common.PREVIOUS_PARAMETERS = None


class FeatureSimilarityLoss(Module):
    def __init__(self, default_loss=cross_entropy):
        super(FeatureSimilarityLoss, self).__init__()
        self.default_loss = default_loss

    def forward(self, x, y):
        common.PREVIOUS_LABEL = common.CURRENT_LABEL
        common.CURRENT_LABEL = y

        features = common.CURRENT_FEATURES
        previous_features = common.PREVIOUS_FEATURES

        loss = self.default_loss(x, y)

        if not common.TRAIN:
            return loss

        for scope, modules in features.items():
            if scope in previous_features:
                for module_id, feature in modules.items():
                    previous_modules = previous_features[scope]
                    previous_feature = previous_modules.get(module_id, None)
                    difference = similarity_loss(previous_feature, feature)
                    loss += difference * 0.0005
        return loss


def similarity_loss(previous_feature, current_feature):
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

