# -*- coding: utf-8 -*-
import unittest
from PIL import Image
from stacked.utils import transformer
from stacked.models import resnet
from stacked.models import scoped_resnet
from stacked.models import blueprinted_resnet
from stacked.meta.blueprint import visualize, get_module_names
from torch.nn import Conv2d
from stacked.modules.scoped_nn import ScopedEnsemble
from stacked.modules.conv3d2d import Conv3d2d
from stacked.utils import common
import glob
import copy


class TestResNet(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestResNet, self).__init__(*args, **kwargs)

    @classmethod
    def setUpClass(cls):
        common.BLUEPRINT_GUI = True
        from tkinter import Tk
        common.GUI = Tk()

        image_paths = glob.glob("images/*")
        cls.out_size = (1, 100)
        cls.test_images = [(s, Image.open(s).resize((128, 128))) for s in image_paths]
        cls.vanilla_model = resnet.ResNet(depth=16, width=1, num_classes=100).cuda()
        cls.scoped_model = scoped_resnet.ResNet("ResNet16", None, None,
                                                depth=16, width=1, num_classes=100).cuda()
        cls.blueprint = blueprinted_resnet.ScopedResNet.describe_default('ResNet28', depth=28,
                                                                         width=1, num_classes=100)
        cls.blueprinted_model = blueprinted_resnet.ScopedResNet(cls.blueprint['name'],
                                                                cls.blueprint).cuda()

    def test_unique_group_scoped(self):
        new_blueprint = copy.deepcopy(self.scoped_model.blueprint)
        same_model = scoped_resnet.ResNet("ResNet16", None, None,
                                          depth=16, width=1, num_classes=100).cuda()
        ref = dict(new_blueprint['group_elements'])['group0']['uniques']
        ref.add('block_container')
        self.assertNotEqual(new_blueprint, self.scoped_model.blueprint)
        new_model = scoped_resnet.ResNet("ResNet16", None, new_blueprint,
                                         depth=16, width=1, num_classes=100).cuda()
        self.assertEqual(same_model.group_container, self.scoped_model.group_container)
        self.assertNotEqual(new_model.group_container, self.scoped_model.group_container)

    def test_unique_block_scoped(self):
        new_blueprint = copy.deepcopy(self.scoped_model.blueprint)
        dref = dict(dict(new_blueprint['group_elements'])['group0']['block_elements'])
        dref['block0']['uniques'].add('conv1')
        new_model = scoped_resnet.ResNet("ResNet16", None, new_blueprint,
                                         depth=16, width=1, num_classes=100).cuda()
        self.assertNotEqual(new_model.blueprint, self.scoped_model.blueprint)
        self.assertNotEqual(new_model.group_container, self.scoped_model.group_container)

    def test_forward(self):
        for path, im in self.test_images:
            x = transformer.image_to_unsqueezed_cuda_variable(im)
            out = self.vanilla_model(x)
            self.assertEqual(out.size(), self.out_size)

    def test_forward_scoped(self):
        for path, im in self.test_images:
            x = transformer.image_to_unsqueezed_cuda_variable(im)
            out = self.scoped_model(x)
            self.assertEqual(out.size(), self.out_size)

    def test_forward_blueprinted(self):
        for path, im in self.test_images:
            x = transformer.image_to_unsqueezed_cuda_variable(im)
            out = self.blueprinted_model(x)
            self.assertEqual(out.size(), self.out_size)

    def test_module_names_blueprinted(self):
        # ResNet -> group16_16
        group = self.blueprint.get_child(0)
        # group16_16 -> block16_16
        block = group.get_child(0)
        # block16_16 -> conv16_16_3
        child = block.get_child((1, 2))
        self.assertEqual(child['name'], 'ResNet28/group16_16/block16_16/conv16_16_3')
        child.make_unique()
        self.assertNotEqual(child['name'], 'ResNet28/group16_16/block16_16/conv16_16_3')
        module_list = set()
        get_module_names(self.blueprint, module_list)
        self.assertTrue(child['name'] in module_list)

    #@unittest.skip("GUI test for uniqueness skipped")
    def test_visual_change_blueprinted(self):
        common.BLUEPRINT_GUI = True
        # group[0] -> block[0] -> unit[1 -> conv] (0: act, 1: bn, 2: conv)
        self.blueprint.get_child((0, 1, 1, 2)).make_unique()
        visualize(self.blueprint)
        new_model = blueprinted_resnet.ScopedResNet('Resnet28', self.blueprint).cuda()
        self.assertEqual(self.blueprinted_model.group_container[1],
                         new_model.group_container[1])
        self.assertNotEqual(self.blueprinted_model.group_container[0],
                            new_model.group_container[0])
        self.assertEqual(self.blueprinted_model.group_container[1].block_container[1],
                         new_model.group_container[1].block_container[1])
        self.assertNotEqual(self.blueprinted_model.group_container,
                            new_model.group_container)
        for path, im in self.test_images:
            x = transformer.image_to_unsqueezed_cuda_variable(im)
            out = new_model(x)
            self.assertEqual(out.size(), self.out_size)

    def test_ensemble_instead_of_conv(self):
        # replace conv0 of ResNet with Ensemble
        kwargs = {'in_channels': 3, 'out_channels': 16,
                  'kernel_size': 3, 'padding': 1, 'stride': 1}
        args = [(Conv2d, [], kwargs)] * 5
        input_shape = transformer.image_to_unsqueezed_cuda_variable(self.test_images[0][1]).size()
        conv0 = self.blueprint['conv0']
        conv0['prefix'] = 'ResNet28_ensemble/stacked0'
        conv0['type'] = ScopedEnsemble
        conv0['kwargs'] = {'iterable_args': args, 'input_shape': input_shape}
        conv0.make_unique()

        # replace the 2nd conv of the 2nd block of the 3rd group with Ensemble
        input_shape = (1, 64, 32, 32)
        kwargs = copy.deepcopy(kwargs)
        kwargs['in_channels'] = 64
        kwargs['out_channels'] = 64
        args = [(Conv3d2d, [], kwargs)] * 5
        conv2 = self.blueprint.get_child((2, 1, 1, 2))
        conv2['prefix'] = 'ResNet/stacked2_2_3'
        conv2['type'] = ScopedEnsemble
        conv2['kwargs'] = {'iterable_args': args, 'input_shape': input_shape}
        conv2.make_unique()
        new_model = blueprinted_resnet.ScopedResNet('ResNet28_ensemble',
                                                    self.blueprint).cuda()
        for path, im in self.test_images:
            x = transformer.image_to_unsqueezed_cuda_variable(im)
            out = new_model(x)
            self.assertEqual(out.size(), self.out_size)


if __name__ == '__main__':
    unittest.main()
