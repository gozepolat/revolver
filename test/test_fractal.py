import unittest
from stacked.models.blueprinted.meta import ScopedMetaMasked
from stacked.models.blueprinted.resnet import ScopedResNet
from stacked.models.blueprinted.fractalgroup import ScopedFractalGroup
from stacked.utils import transformer, common
from stacked.meta.blueprint import visualize
from stacked.modules.scoped_nn import ScopedConv2d
from stacked.meta.blueprint import visit_modules
from PIL import Image
import glob


class TestScopedFractalGroup(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestScopedFractalGroup, self).__init__(*args, **kwargs)

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

    def test_fractal_depth(self):
        common.BLUEPRINT_GUI = False
        if common.BLUEPRINT_GUI and common.GUI is None:
            from tkinter import Tk
            common.GUI = Tk()

        for fractal_depth in range(1, 4):
            bp = ScopedResNet.describe_default('ResNet_meta%d' % fractal_depth,
                                               depth=16,
                                               conv_module=ScopedMetaMasked,
                                               input_shape=(1, 3, 32, 32),
                                               num_classes=10,
                                               group_module=ScopedFractalGroup,
                                               fractal_depth=fractal_depth)
            if common.BLUEPRINT_GUI:

                def make_conv2d_unique(bp, _, __):
                    if issubclass(bp['type'], ScopedConv2d):
                        bp.make_unique()

                visit_modules(bp, None, None, make_conv2d_unique)
                visualize(bp)

            model = ScopedResNet('ResNet_meta%d' % fractal_depth, bp).cuda()
            self.model_run(model)

        common.BLUEPRINT_GUI = False
        common.GUI = None

