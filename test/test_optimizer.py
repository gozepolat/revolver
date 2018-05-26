import unittest
from stacked.modules.scoped_nn import ScopedFeatureSimilarityLoss, \
    ScopedParameterSimilarityLoss
from stacked.modules.loss import collect_features
from stacked.models.blueprinted.optimizer import ScopedEpochEngine
from stacked.meta.blueprint import make_module
from stacked.utils import common


class TestTrainer(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestTrainer, self).__init__(*args, **kwargs)

    @classmethod
    def setUpClass(cls):
        pass

    def test_feature_similarity_loss(self):
        common.BLUEPRINT_GUI = False
        blueprint = ScopedEpochEngine.describe_default(depth=10,
                                                       criterion=ScopedFeatureSimilarityLoss,
                                                       callback=collect_features,
                                                       batch_size=16)
        engine = make_module(blueprint)
        engine.start_epoch()
        engine.train_n_samples(48)
        engine.end_epoch()

    def test_parameter_similarity_loss(self):
        common.BLUEPRINT_GUI = False
        blueprint = ScopedEpochEngine.describe_default("new_engine", depth=10,
                                                       criterion=ScopedParameterSimilarityLoss,
                                                       callback=collect_features,
                                                       batch_size=16)
        engine = make_module(blueprint)
        engine.start_epoch()
        engine.train_n_samples(48)
        engine.end_epoch()