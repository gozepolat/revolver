# -*- coding: utf-8 -*-
import unittest
from PIL import Image
from stacked.utils import transformer
from stacked.modules.conv import Conv3d2d
from stacked.modules.scoped_nn import ScopedConv3d2d, ScopedConv2d
from stacked.models.blueprinted.conv_unit import ScopedConvUnit
from torch.nn import Conv2d
import glob


class TestConv3d2d(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestConv3d2d, self).__init__(*args, **kwargs)

    @classmethod
    def setUpClass(cls):
        image_paths = glob.glob("images/*")
        cls.test_images = [(s, Image.open(s).resize((128, 128))) for s in image_paths]
        cls.conv2d = Conv2d(3, 160, 3, 1, 1).cuda()
        cls.conv3d2d = Conv3d2d(2, 2, 3, (2, 1, 1), 1).cuda()
        cls.out_size = (1, 80, 128, 128)
        ScopedConv3d2d("scoped_conv3d-0", 2, 2, 3, (2, 1, 1), 1)
        cls.scoped = ScopedConv3d2d("scoped_conv3d-0").cuda()

    def test_channel_squeeze(self):
        for path, im in self.test_images:
            out = self.conv2d(transformer.image_to_unsqueezed_cuda_variable(im))
            out = self.conv3d2d(out)
            self.assertEqual(out.size(), self.out_size)

    def test_scope_equality(self):
        ScopedConv3d2d("scoped_conv3d-1", 2, 2, 3, (2, 1, 1), 1)
        scoped4 = ScopedConv3d2d("scoped_conv3d-0").cuda()
        scoped5 = ScopedConv3d2d("scoped_conv3d-1").cuda()
        self.assertEqual(self.scoped, scoped4)
        self.assertNotEqual(self.scoped, scoped5)

    def test_scoped_modules(self):
        for path, im in self.test_images:
            out = self.conv2d(transformer.image_to_unsqueezed_cuda_variable(im))
            x = self.scoped(out)
            self.assertEqual(x.size(), self.out_size)

    def test_adjust_arguments(self):
        conv3d_args = None
        kwargs = {'in_channels': 16, 'out_channels': 16,
                  'kernel_size': 7, 'stride': 1,
                  'padding': 7 // 2, 'dilation': 1,
                  'groups': 1, 'bias': True}

        conv3d_args = ScopedConv3d2d.adjust_args(conv3d_args, ScopedConv3d2d, **kwargs)
        conv3d_args2 = ScopedConv3d2d.adjust_args(conv3d_args, ScopedConvUnit,
                                                  32, 64, 3, 2, 1, 1, 1, True)
        conv_args = ScopedConv3d2d.adjust_args(conv3d_args, ScopedConv2d,
                                               32, 64, 3, 2, 1, 1, 1, True)
        self.assertEqual(conv3d_args, conv3d_args2)
        self.assertNotEqual(conv3d_args, conv_args)


if __name__ == '__main__':
    unittest.main()
