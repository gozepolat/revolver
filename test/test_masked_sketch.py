# -*- coding: utf-8 -*-
import unittest
from stacked.meta import masked_sketch
from PIL import Image
from torch.nn import Conv2d
from stacked.utils import transformer
import glob


class TestEnsemble(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestEnsemble, self).__init__(*args, **kwargs)

    @classmethod
    def setUpClass(cls):
        image_paths = glob.glob("images/*")
        cls.test_images = [(s, Image.open(s).resize((256, 256))) for s in image_paths]

    def test_forward(self):
        in_channels = 3
        out_channels = 3
        kernel_size = 3
        stride = 1
        args = (in_channels, out_channels, kernel_size, stride)
        kwargs = {'padding': 1}
        args = [(Conv2d, args, kwargs)] * 5
        out_size = (1, 3, 256, 256)
        input_shape = transformer.image_to_unsqueezed_cuda_variable(self.test_images[0][1]).size()
        ensemble = masked_sketch.Ensemble(args, input_shape).cuda()
        self.assertEqual(out_size, ensemble.output_shape)

        # test forward with various images
        for path, im in self.test_images:
            out = ensemble(transformer.image_to_unsqueezed_cuda_variable(im))
            self.assertEqual(out.size(), out_size)


if __name__ == '__main__':
    unittest.main()
