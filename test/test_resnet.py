# -*- coding: utf-8 -*-
import unittest
from PIL import Image
from stacked.utils import transformer
from stacked.models.resnet import ResNet
import glob


class TestResNet(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestResNet, self).__init__(*args, **kwargs)

    @classmethod
    def setUpClass(cls):
        image_paths = glob.glob("images/*")
        cls.test_images = [(s, Image.open(s).resize((128, 128))) for s in image_paths]
        cls.model = ResNet(depth=16, width=1, num_classes=100).cuda()
        cls.out_size = (1, 100)

    def test_forward(self):
        for path, im in self.test_images:
            out = self.model(transformer.image_to_unsqueezed_cuda_variable(im))
            self.assertEqual(out.size(), self.out_size)


if __name__ == '__main__':
    unittest.main()
