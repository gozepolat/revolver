import torch
from torch.nn import Module
from torch.nn.functional import cosine_embedding_loss, cross_entropy
from revolver.utils import common, transformer


class FeatureSimilarityLoss(Module):
    def __init__(self, default_loss=cross_entropy):
        super(FeatureSimilarityLoss, self).__init__()
        self.default_loss = default_loss

    def forward(self, x, y):
        return FeatureSimilarityLoss.function(x, y, self.default_loss)

    @staticmethod
    def get_scalar():
        return 0.04

    @staticmethod
    def function(x, y, default_loss=cross_entropy):
        loss = default_loss(x, y)
        if not common.TRAIN:
            return loss

        common.PREVIOUS_LABEL = common.CURRENT_LABEL
        common.CURRENT_LABEL = y

        features = common.CURRENT_FEATURES
        previous_features = common.PREVIOUS_FEATURES

        if previous_features is None:
            return loss

        for scope, modules in features.items():
            if scope in previous_features:
                for module_id, feature in modules.items():
                    previous_modules = previous_features[scope]

                    if previous_modules is None:
                        continue

                    previous_feature = previous_modules.get(module_id, None)

                    if previous_feature is None:
                        continue

                    difference = get_feature_similarity_loss(previous_feature, feature)
                    loss += difference * FeatureSimilarityLoss.get_scalar()

        # clear the old features
        for scope in previous_features:
            features = previous_features[scope]
            if features is not None:
                for module_id in features:
                    features[module_id] = None

        return loss


def get_feature_similarity_loss(previous_feature, current_feature):
    """Penalize features that are not similar when the labels are the same"""
    x = common.PREVIOUS_LABEL
    y = common.CURRENT_LABEL

    if x is None or x.size() != y.size():
        return 0

    index = x != y
    label = torch.ones(index.size())
    if torch.cuda.is_available():
        label.cuda()
    label[index] = -1
    second_dim = previous_feature.size()[1:]

    view = 1
    for d in second_dim:
        view *= d
    view = (previous_feature.size()[0], view)
    return cosine_embedding_loss(previous_feature.view(view),
                                 current_feature.view(view), label)


def collect_features(scope, module_id, x, *_):
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
        return 0.2

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


class FeatureConvergenceLoss(Module):
    def __init__(self, default_loss=cross_entropy):
        super(FeatureConvergenceLoss, self).__init__()
        self.default_loss = default_loss

    def forward(self, x, y):
        return FeatureConvergenceLoss.function(x, y, self.default_loss)

    @staticmethod
    def get_scalar(step):
        return min(1e-3 * 1.2 ** step, 0.4)

    @staticmethod
    def function(x, y, default_loss=cross_entropy):
        loss = default_loss(x, y)

        if not common.TRAIN:
            return loss

        depth = common.FEATURE_DEPTH_CTR
        common.FEATURE_DEPTH_CTR = 0

        last = 0
        for i in range(1, depth):
            prev_id = common.FEATURE_DEPTHS[i - 1]
            current_id = common.FEATURE_DEPTHS[i]
            prev_feature = common.CURRENT_FEATURES[prev_id]
            current_feature = common.CURRENT_FEATURES[current_id]

            if prev_feature.size() == current_feature.size():
                if i - last > 1:
                    difference = get_parameter_similarity(prev_feature, current_feature)
                    loss += difference * FeatureConvergenceLoss.get_scalar(i - last)
            else:
                last = i
        return loss


def collect_depthwise_features(_, module_id, x):
    """Collect current features, and keep the depthwise ids"""
    if not common.TRAIN:
        return

    common.FEATURE_DEPTHS[common.FEATURE_DEPTH_CTR] = module_id
    common.CURRENT_FEATURES[module_id] = x
    common.FEATURE_DEPTH_CTR += 1
