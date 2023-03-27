import unittest

import torch.cuda

from stacked.models.blueprinted.meta import ScopedMetaMasked
from stacked.models.blueprinted.separable import ScopedDepthwiseSeparable
from stacked.models.blueprinted.densenet import ScopedDenseNet
from stacked.models.blueprinted.densesumgroup import ScopedDenseSumGroup
from stacked.models.blueprinted.denseconcatgroup import ScopedDenseConcatGroup
from stacked.models.blueprinted.bottleneckblock import ScopedBottleneckBlock
from stacked.utils import transformer, common
from stacked.meta.blueprint import visualize
from PIL import Image
import glob


class TestScopedDenseNet(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestScopedDenseNet, self).__init__(*args, **kwargs)

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

    def test_dense_sum_depth(self):
        common.BLUEPRINT_GUI = False
        if common.BLUEPRINT_GUI and common.GUI is None:
            from tkinter import Tk
            common.GUI = Tk()

        for dense_depth in range(1, 4):
            bp = ScopedDenseNet.describe_default('DenseNet_sum%d' % dense_depth,
                                                 depth=16,
                                                 conv_module=ScopedMetaMasked,
                                                 input_shape=(1, 3, 32, 32),
                                                 num_classes=10,
                                                 group_module=ScopedDenseSumGroup,
                                                 head_kernel=3, head_stride=1, head_padding=1,
                                                 head_pool_kernel=3, head_pool_stride=2,
                                                 head_pool_padding=1,
                                                 dense_unit_module=ScopedBottleneckBlock,
                                                 block_module=ScopedBottleneckBlock,
                                                 head_modules=('conv', 'bn', 'act', 'pool'))

            model = ScopedDenseNet('DenseNet_sum%d' % dense_depth, bp)
            if torch.cuda.is_available():
                model.cuda()
            self.model_run(model)
            if common.BLUEPRINT_GUI:
                visualize(bp)

        if common.BLUEPRINT_GUI:
            common.BLUEPRINT_GUI = False
            common.GUI = None

    def test_dense_concat_depth(self):
        common.BLUEPRINT_GUI = False
        if common.BLUEPRINT_GUI and common.GUI is None:
            from tkinter import Tk
            common.GUI = Tk()

        for dense_depth in range(1, 4):
            bp = ScopedDenseNet.describe_default('DenseNet_concat%d' % dense_depth,
                                                 depth=28,
                                                 conv_module=ScopedMetaMasked,
                                                 input_shape=(1, 3, 32, 32),
                                                 num_classes=10,
                                                 residual=False,
                                                 group_module=ScopedDenseConcatGroup)

            model = ScopedDenseNet('DenseNet_concat%d' % dense_depth, bp)
            if torch.cuda.is_available():
                model.cuda()
            self.model_run(model)
            if common.BLUEPRINT_GUI:
                visualize(bp)

        if common.BLUEPRINT_GUI:
            common.BLUEPRINT_GUI = False
            common.GUI = None

    def test_dense_concat_bottleneck_depth(self):
        common.BLUEPRINT_GUI = False
        if common.BLUEPRINT_GUI and common.GUI is None:
            from tkinter import Tk
            common.GUI = Tk()

        for dense_depth in range(1, 4):
            bp = ScopedDenseNet.describe_default('DenseNet_concat_bottleneck%d' % dense_depth,
                                                 depth=28,
                                                 conv_module=ScopedMetaMasked,
                                                 input_shape=(1, 3, 32, 32),
                                                 num_classes=10,
                                                 group_module=ScopedDenseConcatGroup,
                                                 dense_unit_module=ScopedBottleneckBlock,
                                                 block_module=ScopedBottleneckBlock,
                                                 residual=False)

            model = ScopedDenseNet('DenseNet_concat_bottleneck%d' % dense_depth, bp)
            if torch.cuda.is_available():
                model.cuda()
            self.model_run(model)
            if common.BLUEPRINT_GUI:
                visualize(bp)

        if common.BLUEPRINT_GUI:
            common.BLUEPRINT_GUI = False
            common.GUI = None

    def test_dense_sum_bottleneck_depthwise_separable(self):
        common.BLUEPRINT_GUI = False
        if common.BLUEPRINT_GUI and common.GUI is None:
            from tkinter import Tk
            common.GUI = Tk()

        for dense_depth in range(1, 4):
            bp = ScopedDenseNet.describe_default('DenseNet_separable%d' % dense_depth,
                                                 depth=28,
                                                 conv_module=ScopedDepthwiseSeparable,
                                                 input_shape=(1, 3, 32, 32),
                                                 num_classes=10,
                                                 group_module=ScopedDenseSumGroup,
                                                 dense_unit_module=ScopedBottleneckBlock,
                                                 block_module=ScopedBottleneckBlock,
                                                 residual=False)
            print("%s" % bp)
            model = ScopedDenseNet('DenseNet_separable%d' % dense_depth, bp)
            if torch.cuda.is_available():
                model.cuda()
            self.model_run(model)
            if common.BLUEPRINT_GUI:
                visualize(bp)

        if common.BLUEPRINT_GUI:
            common.BLUEPRINT_GUI = False
            common.GUI = None
