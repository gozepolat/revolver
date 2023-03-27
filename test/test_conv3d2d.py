# -*- coding: utf-8 -*-
import unittest

import torch.cuda
from PIL import Image
from stacked.utils import transformer
from stacked.modules.conv import Conv3d2d
from stacked.modules.scoped_nn import ScopedConv3d2d, ScopedConv2d
from stacked.models.blueprinted.convunit import ScopedConvUnit
from stacked.meta.blueprint import make_module
from torch.nn import Conv2d
import glob


class TestConv3d2d(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestConv3d2d, self).__init__(*args, **kwargs)

    @classmethod
    def setUpClass(cls):
        image_paths = glob.glob("images/*")
        cls.test_images = [(s, Image.open(s).resize((128, 128))) for s in image_paths]
        cls.conv2d = Conv2d(3, 160, 3, 1, 1)
        cls.conv3d2d = Conv3d2d(2, 2, 3, (2, 1, 1), 1)

        cls.out_size = (1, 80, 128, 128)
        ScopedConv3d2d("scoped_conv3d-0", 2, 2, 3, (2, 1, 1), 1)
        cls.scoped = ScopedConv3d2d("scoped_conv3d-0")
        if torch.cuda.is_available():
            cls.conv2d = cls.conv2d.cuda()
            cls.conv3d2d = cls.conv3d2d.cuda()
            cls.scoped = cls.scoped.cuda()

    def test_channel_squeeze(self):
        for path, im in self.test_images:
            out = self.conv2d(transformer.image_to_unsqueezed_cuda_variable(im))
            out = self.conv3d2d(out)
            self.assertEqual(out.size(), self.out_size)

    def test_scope_equality(self):
        ScopedConv3d2d("scoped_conv3d-1", 2, 2, 3, (2, 1, 1), 1)
        scoped4 = ScopedConv3d2d("scoped_conv3d-0")
        scoped5 = ScopedConv3d2d("scoped_conv3d-1")
        if torch.cuda.is_available():
            scoped4 = scoped4.cuda()
            scoped5 = scoped5.cuda()
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

    def test_describe_from_blueprint(self):
        default = ScopedConv3d2d.describe_default(input_shape=(1, 3, 128, 128),
                                                  in_channels=3, out_channels=3,
                                                  kernel_size=7, stride=1, padding=3,
                                                  dilation=1, groups=1, bias=False)
        kwargs = {'in_channels': 1, 'out_channels': 1}
        conv3d2s_from_blueprint = ScopedConv3d2d.describe_from_blueprint('conv3d2dX', '', default, None, kwargs)

        # out channels = 20 x 3 = 60
        conv3d2dx_blueprint = ScopedConv3d2d.describe_from_blueprint('conv3d2dX_1', '', default, None,
                                                                     {'in_channels': 1, 'out_channels': 20,
                                                                      'stride': 2})
        conv3d2d = make_module(conv3d2s_from_blueprint)
        conv3d2dx = make_module(conv3d2dx_blueprint)

        if torch.cuda.is_available():
            conv3d2d = conv3d2d.cuda()
            conv3d2dx = conv3d2dx.cuda()

        for path, im in self.test_images:
            x = transformer.image_to_unsqueezed_cuda_variable(im)

            # test same # channels
            out = conv3d2d(x)
            self.assertEqual(out.size(), (1, 3, 128, 128))

            # test different # channels
            out_x = conv3d2dx(x)
            self.assertEqual(out_x.size(), (1, 60, 64, 64))


if __name__ == '__main__':
    unittest.main()
