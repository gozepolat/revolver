import unittest
from stacked.models.blueprinted.meta import ScopedMetaMasked
from stacked.models.blueprinted.resnet import ScopedResNet
from stacked.utils import transformer, common
from PIL import Image
import glob


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

    @unittest.skip("in construction")
    def test_replace_conv_with_meta_layer(self):
        bp = ScopedResNet.describe_default('ResNet_meta', depth=22,
                                           conv_module=ScopedMetaMasked,
                                           input_shape=(1, 3, 32, 32),
                                           num_classes=10)

        self.model_run(ScopedResNet('ResNet_meta', bp))


