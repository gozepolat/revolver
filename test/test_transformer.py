# -*- coding: utf-8 -*-
import torch
import unittest
from stacked.utils import transformer
from PIL import Image
import glob


class TestTransformImage(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestTransformImage, self).__init__(*args, **kwargs)

    @classmethod
    def setUpClass(self):
        image_paths = glob.glob("images/*")
        self.test_images = [(s, Image.open(s)) for s in image_paths]

    def test_image_variable_conversion(self):
        variables = []
        variable_type = torch.autograd.Variable
        image_type = Image.Image

        # image to variable
        for path, im in self.test_images:
            variable = transformer.image_to_variable(im)
            variables.append((path, variable))
            self.assertEqual(type(variable), variable_type,
                             "{} is not converted correctly from image".format(path))

        # variable to image
        for path, v in variables:
            image = transformer.variable_to_image(v)
            self.assertEqual(type(image), image_type,
                             "{} is not converted back correctly from variable".format(path))

    def test_numpy_to_unsqueezed_image(self):
        for path, im in self.test_images:
            image = transformer.image_to_numpy(im)
            data = transformer.image_numpy_to_unsqueezed_cuda_tensor(image)
            self.assertEqual(data.dim(), 4,
                             "{} is not unsqueezed to tensor correctly from image".format(path))

            data = transformer.image_to_unsqueezed_cuda_variable(image)
            self.assertTrue(data.size()[1] == 3 and data.dim() == 4,
                            "{} is not unsqueezed to variable correctly from image".format(path))

            data = transformer.image_to_unsqueezed_cuda_variable(path)
            self.assertTrue(data.size()[1] == 3 and data.dim() == 4,
                            "{} is not unsqueezed to variable correctly from path".format(path))

    def test_normalize(self):
        for path, im in self.test_images:
            normalized = transformer.normalize(path)
            self.assertTrue(0 <= torch.min(normalized) < torch.max(normalized) <= 1.0,
                            "{} is not normalized correctly from path".format(path))

            normalized = transformer.normalize(transformer.image_to_variable(im))
            self.assertTrue(0 <= torch.min(normalized) < torch.max(normalized) <= 1.0,
                            "{} is not normalized correctly from variable".format(path))


if __name__ == '__main__':
    unittest.main()
