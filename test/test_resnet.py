# -*- coding: utf-8 -*-
import unittest
from PIL import Image
from stacked.utils import transformer
from stacked.models.resnet import ResNet
from stacked.models.scoped_resnet import ScopedResNet
import glob


class TestResNet(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestResNet, self).__init__(*args, **kwargs)

    @classmethod
    def setUpClass(cls):
        image_paths = glob.glob("images/*")
        cls.test_images = [(s, Image.open(s).resize((128, 128))) for s in image_paths]
        cls.model = ResNet(depth=16, width=1, num_classes=100).cuda()
        cls.scoped_model = ScopedResNet("ResNet16", depth=16, width=1, num_classes=100).cuda()
        cls.out_size = (1, 100)

    def test_forward(self):
        for path, im in self.test_images:
            x = transformer.image_to_unsqueezed_cuda_variable(im)
            out = self.model(x)
            self.assertEqual(out.size(), self.out_size)

    def test_forward_scoped(self):
        for path, im in self.test_images:
            x = transformer.image_to_unsqueezed_cuda_variable(im)
            out = self.scoped_model(x)
            self.assertEqual(out.size(), self.out_size)


if __name__ == '__main__':
    unittest.main()
