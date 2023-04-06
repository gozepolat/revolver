import unittest

import torch.cuda

from revolver.models.blueprinted.meta import ScopedMetaMasked
from revolver.models.blueprinted.resnet import ScopedResNet
from revolver.models.blueprinted.convunit import ScopedConvUnit
from revolver.models.blueprinted.densefractalgroup import ScopedDenseFractalGroup
from revolver.utils import transformer, common
from revolver.meta.blueprint import visualize
from PIL import Image
import glob


class TestScopedDenseFractalGroup(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestScopedDenseFractalGroup, self).__init__(*args, **kwargs)

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

    def test_dense_depth(self):
        common.BLUEPRINT_GUI = False
        if common.BLUEPRINT_GUI and common.GUI is None:
            from tkinter import Tk
            common.GUI = Tk()

        for dense_depth in range(1, 4):
            bp = ScopedResNet.describe_default('ResNet_densefractal%d' % dense_depth,
                                               depth=10,
                                               conv_module=ScopedMetaMasked,
                                               input_shape=(1, 3, 32, 32),
                                               num_classes=10,
                                               block_module=ScopedConvUnit,
                                               group_module=ScopedDenseFractalGroup,
                                               fractal_depth=dense_depth)

            model = ScopedResNet('ResNet_densefractal%d' % dense_depth, bp)
            if torch.cuda.is_available():
                model.cuda()

            self.model_run(model)
            if common.BLUEPRINT_GUI:
                visualize(bp)

        if common.BLUEPRINT_GUI:
            common.BLUEPRINT_GUI = False
            common.GUI = None


