# -*- coding: utf-8 -*-
import unittest
from PIL import Image
from revolver.utils import transformer
from revolver.models import scoped_resnet_sketch
from revolver.utils import common
import glob
import copy
import torch

class TestResNet(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestResNet, self).__init__(*args, **kwargs)

    @classmethod
    def setUpClass(cls):
        common.BLUEPRINT_GUI = False
        image_paths = glob.glob("images/*")
        cls.out_size = (1, 100)
        cls.test_images = [(s, Image.open(s).resize((128, 128))) for s in image_paths]
        cls.scoped_model = scoped_resnet_sketch.ResNet("ResNet16", None, None,
                                                       depth=16, width=1, num_classes=100)
        cls.scoped_model = cls.scoped_model.cuda() if torch.cuda.is_available() else cls.scoped_model

    def test_unique_group_scoped(self):
        new_blueprint = copy.deepcopy(self.scoped_model.blueprint)
        same_model = scoped_resnet_sketch.ResNet("ResNet16", None, None,
                                                 depth=16, width=1, num_classes=100)
        same_model = same_model.cuda() if torch.cuda.is_available() else same_model
        ref = dict(new_blueprint['group_elements'])['group0']['uniques']
        ref.add('block_container')
        self.assertNotEqual(new_blueprint, self.scoped_model.blueprint)
        new_model = scoped_resnet_sketch.ResNet("ResNet16", None, new_blueprint,
                                                depth=16, width=1, num_classes=100)
        new_model = new_model.cuda() if torch.cuda.is_available() else new_model
        self.assertEqual(same_model.group_container, self.scoped_model.group_container)
        self.assertNotEqual(new_model.group_container, self.scoped_model.group_container)

    def test_unique_block_scoped(self):
        new_blueprint = copy.deepcopy(self.scoped_model.blueprint)
        ref = dict(dict(new_blueprint['group_elements'])['group0']['block_elements'])
        ref['block0']['uniques'].add('conv1')
        new_model = scoped_resnet_sketch.ResNet("ResNet16", None, new_blueprint,
                                                depth=16, width=1, num_classes=100)
        new_model = new_model.cuda() if torch.cuda.is_available() else new_model
        self.assertNotEqual(new_model.blueprint, self.scoped_model.blueprint)
        self.assertNotEqual(new_model.group_container, self.scoped_model.group_container)

    def test_forward_scoped(self):
        for path, im in self.test_images:
            x = transformer.image_to_unsqueezed_cuda_variable(im)
            out = self.scoped_model(x)
            self.assertEqual(out.size(), self.out_size)