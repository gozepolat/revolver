import unittest
from stacked.modules.scoped_nn import ScopedFeatureSimilarityLoss, \
    ScopedParameterSimilarityLoss
from stacked.models.blueprinted.densesumgroup import ScopedDenseSumGroup
from stacked.models.blueprinted.bottleneckblock import ScopedBottleneckBlock
from stacked.models.blueprinted.densenet import ScopedDenseNet
from stacked.modules.loss import collect_features
from stacked.models.blueprinted.optimizer import ScopedEpochEngine
from stacked.meta.blueprint import make_module
from stacked.utils import common


class TestTrainer(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestTrainer, self).__init__(*args, **kwargs)

    def test_feature_similarity_loss(self):
        common.BLUEPRINT_GUI = False
        blueprint = ScopedEpochEngine.describe_default(prefix='OptimizerEpochEngineWithLoss',
                                                       depth=10,
                                                       block_module=ScopedBottleneckBlock,
                                                       #criterion=ScopedFeatureSimilarityLoss,
                                                       #callback=collect_features,
                                                       batch_size=16)

        engine = make_module(blueprint)
        engine.start_epoch()
        engine.train_n_batches(16)
        engine.state['epoch'] += 1

    @unittest.skip("Requires retain_graph option enabled")
    def test_parameter_similarity_loss(self):
        common.BLUEPRINT_GUI = False
        print("ScopedParamSimilarity")
        blueprint = ScopedEpochEngine.describe_default("new_engine", depth=10,
                                                       criterion=ScopedParameterSimilarityLoss,
                                                       callback=collect_features,
                                                       batch_size=16,
                                                       weight_decay=0.0005)
        engine = make_module(blueprint)
        engine.start_epoch()
        engine.train_n_batches(16)
        engine.state['epoch'] += 1

    def test_densenet(self):
        common.BLUEPRINT_GUI = False
        print("ScopedParamSimilarity")
        net = ScopedDenseNet.describe_default(prefix='OptimizerNet', num_classes=10,
                                              depth=22, width=1,
                                              block_depth=2, drop_p=0.5,
                                              group_module=ScopedDenseSumGroup,
                                              residual=False,
                                              block_module=ScopedBottleneckBlock,
                                              dense_unit_module=ScopedBottleneckBlock,
                                              input_shape=(16, 3, 32, 32), fractal_depth=3,
                                              head_kernel=3, head_stride=1, head_padding=1,
                                              head_modules=('conv', 'bn'))

        engine_blueprint = ScopedEpochEngine.describe_default(prefix='OptimizerEpochEngine',
                                                              net_blueprint=net,
                                                              max_epoch=1,
                                                              batch_size=16,
                                                              learning_rate=0.2,
                                                              lr_decay_ratio=0.2,
                                                              weight_decay=0.0005)

        engine = make_module(engine_blueprint)
        engine.start_epoch()
        engine.train_n_batches(16)
        engine.state['epoch'] += 1
