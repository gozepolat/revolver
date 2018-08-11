import unittest
from stacked.models.blueprinted.meta import ScopedMetaMasked
from stacked.models.blueprinted.optimizer import ScopedEpochEngine
from stacked.models.blueprinted.resnet import ScopedResNet
from stacked.models.blueprinted.bottleneckblock import ScopedBottleneckBlock
from stacked.utils import transformer, common
from stacked.meta.blueprint import visualize, Blueprint, make_module
from stacked.modules.scoped_nn import ScopedConv2d
from stacked.meta.blueprint import visit_modules
from PIL import Image
import glob
import torch


class TestScopedMetaMasked(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestScopedMetaMasked, self).__init__(*args, **kwargs)

    @classmethod
    def setUpClass(cls):
        image_paths = glob.glob("images/*")
        cls.out_size = (1, 10)
        cls.test_images = [(s, Image.open(s).resize((32, 32))) for s in image_paths]

    def model_run(self, model):
        for path, im in self.test_images:
            x = transformer.image_to_unsqueezed_cuda_variable(im)
            out = model(x)
            self.assertEqual(out.size(), self.out_size)

    def test_replace_conv_with_meta_layer(self):
        common.BLUEPRINT_GUI = False
        if common.BLUEPRINT_GUI and common.GUI is None:
            from tkinter import Tk
            common.GUI = Tk()

        bp = ScopedResNet.describe_default('ResNet_meta', depth=22,
                                           conv_module=ScopedMetaMasked,
                                           input_shape=(1, 3, 32, 32),
                                           num_classes=10)
        if common.BLUEPRINT_GUI:

            def make_conv2d_unique(bp, _, __):
                if issubclass(bp['type'], ScopedConv2d):
                    bp.make_unique()

            visit_modules(bp, None, None, make_conv2d_unique)
            visualize(bp)
        self.model_run(ScopedResNet('ResNet_meta', bp).cuda())

        common.BLUEPRINT_GUI = False
        common.GUI = None

    def _get_model_blueprint(self):
        return ScopedResNet.describe_default(prefix='ResNet_blueprint',
                                             num_classes=10,
                                             depth=22, width=2,
                                             block_depth=2, drop_p=0.5,
                                             conv_module=ScopedMetaMasked,
                                             dropout_p=0.2,
                                             residual=False,
                                             skeleton=(12, 24, 48),
                                             block_module=ScopedBottleneckBlock,
                                             input_shape=(1, 3, 32, 32))

    def _get_engine_blueprint(self, net_blueprint=None):
        return ScopedEpochEngine.describe_default(prefix='EpochEngine_blueprint',
                                                  net_blueprint=net_blueprint,
                                                  max_epoch=10,
                                                  batch_size=32,
                                                  learning_rate=0.1,
                                                  lr_decay_ratio=0.1,
                                                  dataset='CIFAR10',
                                                  num_thread=4,
                                                  use_tqdm=False, crop_size=32,
                                                  weight_decay=0.0002)

    def _recursive_assertEqual(self, bp1, bp2):
        if isinstance(bp1, Blueprint):
            for k in bp1:
                if k not in bp2:
                    return False
                if k == 'blueprint' or k == 'parent':
                    continue
                if k == 'children' or k == 'bps':
                    for c1, c2 in (bp1[k], bp2[k]):
                        if not self._recursive_assertEqual(c1, c2):
                            return False

                if not self._recursive_assertEqual(bp1[k], bp2[k]):
                    return False
        else:
            print(bp1, bp2)
            return self.assertEqual(bp1, bp2)
        return True

    def test_blueprint_dump(self):
        common.BLUEPRINT_GUI = False
        resnet = self._get_model_blueprint()

        from six.moves import cPickle as pickle
        fname = '/tmp/%s' % resnet['name']
        with open(fname, 'w') as f:
            pickle.dump(resnet, f, protocol=pickle.HIGHEST_PROTOCOL)

        with open(fname, 'r') as f:
            bp = pickle.load(f)

        self._recursive_assertEqual(bp, resnet)

    def test_model_dump(self):
        common.BLUEPRINT_GUI = False
        resnet = self._get_model_blueprint()

        engine_blueprint = self._get_engine_blueprint(resnet)
        engine = make_module(engine_blueprint)
        fname = '/tmp/{}_model_{}_bs_{}_decay_{}_lr_{}.pth.tar'.format(
                       engine.net.blueprint['name'],
                       'CIFAR10', 32, 0.0002, 0.1)
        torch.save({'model': engine.net.state_dict()},
                   fname)
        d = torch.load(fname)

        def make_unique(bp, _, __):
            bp.make_unique()

        visit_modules(resnet, None, None, make_unique)

        m = make_module(resnet)
        m.load_state_dict(d['model'])
        self.assertEqual(m, engine.net)
