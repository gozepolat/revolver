# -*- coding: utf-8 -*-
import unittest
from PIL import Image
from stacked.utils import transformer
from stacked.models.simple_resnet import ResNet
from stacked.models.blueprinted.resnet import ScopedResNet
from stacked.meta.blueprint import visualize, collect_keys
from stacked.models.blueprinted.ensemble import ScopedEnsembleMean
from stacked.utils import common
import glob
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

        cls.vanilla_model = ResNet(depth=16, width=1, num_classes=100)
        if torch.cuda.is_available():
            cls.vanilla_model.cuda()

        cls.blueprint = ScopedResNet.describe_default('ResNet28',
                                                      depth=28,
                                                      input_shape=(1, 3, 128, 128),
                                                      width=1,
                                                      num_classes=100)
        cls.blueprinted_model = ScopedResNet(cls.blueprint['name'],
                                             cls.blueprint)
        if torch.cuda.is_available():
            cls.blueprinted_model.cuda()

    def test_forward_vanilla_model(self):
        for path, im in self.test_images:
            x = transformer.image_to_unsqueezed_cuda_variable(im)
            out = self.vanilla_model(x)
            self.assertEqual(out.size(), self.out_size)

    def test_forward_blueprinted_model(self):
        for path, im in self.test_images:
            x = transformer.image_to_unsqueezed_cuda_variable(im)
            out = self.blueprinted_model(x)
            self.assertEqual(out.size(), self.out_size)

    def test_make_unique(self):
        common.BLUEPRINT_GUI = False
        # conv[in]_[out]_[kernel]_[stride]_[padding]_[dilation]_[groups]_[bias]
        conv_name = 'ResNet28/group/block/unit/conv16_16_3_1_1_1_1_0'

        # ResNet -> group -> block -> unit -> conv
        group = self.blueprint.get_element(0)
        block = group.get_element(0)
        unit = block.get_element(0)
        conv = unit.get_element('conv')

        self.assertEqual(conv['name'], conv_name)
        conv.make_unique()
        self.assertNotEqual(conv['name'], conv_name)

        # check whether the child is available in self.blueprint
        module_list = set(collect_keys(self.blueprint, 'name'))

        self.assertTrue(conv['name'] in module_list)

    def test_get_element_with_tuple_index(self):
        convdim_name = 'ResNet28/group/block/convdim16_16_1_1_0_1_1_0'
        convdim = self.blueprint.get_element((0, 0, 'convdim'))
        self.assertEqual(convdim['name'], convdim_name)

    def test_ensemble_instead_of_conv(self):
        common.BLUEPRINT_GUI = False
        # replace conv0 of ResNet with Ensemble
        conv0 = ScopedEnsembleMean.describe_from_blueprint('ensemble', '',
                                                           self.blueprint['conv'])
        conv0.make_unique()
        self.blueprint['conv'] = conv0

        # replace the 1st conv of the 2nd block of the 3rd group with Ensemble
        index = (2, 1, 0, 'conv')
        conv2 = self.blueprint.get_element(index)
        conv2 = ScopedEnsembleMean.describe_from_blueprint('ensemble', '2', conv2)
        conv2.make_unique()
        self.blueprint.set_element(index, conv2)

        new_model = ScopedResNet('ResNet28_ensemble', self.blueprint)
        if torch.cuda.is_available():
            new_model.cuda()
        for path, im in self.test_images:
            x = transformer.image_to_unsqueezed_cuda_variable(im)
            out = new_model(x)
            self.assertEqual(out.size(), self.out_size)

    #@unittest.skip("GUI test for uniqueness skipped")
    def test_visual_change_blueprinted(self):
        common.BLUEPRINT_GUI = True
        if common.GUI is None:
            from tkinter import Tk
            common.GUI = Tk()
        blueprint = ScopedResNet.describe_default('ResNet28',
                                                  depth=28,
                                                  input_shape=(1, 3, 128, 128),
                                                  width=1,
                                                  num_classes=100)
        common.BLUEPRINT_GUI = True
        # group[0] -> block[1] -> unit[0].conv
        blueprint.get_element((0, 1, 0, 'conv')).make_unique()

        # make the new ScopedResNet name different
        blueprint['suffix'] = '_new'
        blueprint.refresh_name()

        new_model = ScopedResNet(blueprint['name'], blueprint)
        if torch.cuda.is_available():
            new_model.cuda()
        self.assertEqual(self.blueprinted_model.container[1],
                         new_model.container[1])
        self.assertNotEqual(self.blueprinted_model.container[0],
                            new_model.container[0])
        self.assertEqual(self.blueprinted_model.container[1].container[1],
                         new_model.container[1].container[1])
        self.assertNotEqual(self.blueprinted_model.container,
                            new_model.container)
        for path, im in self.test_images:
            x = transformer.image_to_unsqueezed_cuda_variable(im)
            out = new_model(x)
            self.assertEqual(out.size(), self.out_size)

        visualize(blueprint)
        common.BLUEPRINT_GUI = False
        common.GUI = None


if __name__ == '__main__':
    unittest.main()
