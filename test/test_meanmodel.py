import unittest
from stacked.models.blueprinted.optimizer import ScopedEpochEngine
from stacked.meta.blueprint import make_module, Blueprint
from stacked.utils.meanmodel import average_model


class TestMeanModel(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestMeanModel, self).__init__(*args, **kwargs)

    @unittest.skip("Download dataset")
    def test_r18(self):
        from torchvision.models import resnet18

        def resnet(name=None, pretrained=False, *_, **__):
            return resnet18(pretrained)

        r18 = Blueprint("r152", "avg", None, False, resnet, kwargs={'pretrained': True})

        blueprint = ScopedEpochEngine.describe_default("new_engine",
                                                       net_blueprint=r18,
                                                       batch_size=32,
                                                       max_epoch=2,
                                                       learning_rate=0.0001,
                                                       dataset='ILSVRC2012',
                                                       num_thread=4, crop_size=224,
                                                       weight_decay=0.0001)
        engine = make_module(blueprint)

        engine.start_epoch()
        engine.train_n_batches(64)
        engine.end_epoch()

        average_model(engine.net)
        engine.start_epoch()
        engine.train_one_epoch()
        engine.end_epoch()

        engine.hook('on_end', engine.state)



