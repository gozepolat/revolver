import unittest
from stacked.models.blueprinted.optimizer import ScopedEpochEngine
from stacked.meta.blueprint import make_module, Blueprint
from stacked.utils.meanmodel import average_model


class TestMeanModel(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestMeanModel, self).__init__(*args, **kwargs)

    @unittest.skip("Download dataset")
    def test_resnet152(self):
        from torchvision.models import resnet152
        r152 = Blueprint("r152", "avg", None, False, resnet152, kwargs={'pretrained': True})

        blueprint = ScopedEpochEngine.describe_default("new_engine",
                                                       net_blueprint=r152,
                                                       batch_size=16,
                                                       dataset='ILSVRC2012',
                                                       num_thread=4,
                                                       weight_decay=0.0005)
        engine = make_module(blueprint)
        average_model(engine.net)
        engine.start_epoch()
        engine.end_epoch()
