# -*- coding: utf-8 -*-
import unittest
from PIL import Image
from stacked.utils import transformer
from stacked.models import resnet
from stacked.models import scoped_resnet_sketch
from stacked.models import blueprinted
from stacked.meta.blueprint import visualize, visit_modules
from torch.nn import Conv2d
from stacked.models.blueprinted import ScopedEnsemble
from stacked.modules.conv import Conv3d2d
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
        cls.scoped_model = scoped_resnet_sketch.ResNet("ResNet16", None, None,
                                                       depth=16, width=1, num_classes=100).cuda()
        cls.blueprint = blueprinted.ScopedResNet.describe_default('ResNet28', depth=28,
                                                                  width=1, num_classes=100)
        cls.blueprinted_model = blueprinted.ScopedResNet(cls.blueprint['name'],
                                                         cls.blueprint).cuda()

    def test_unique_group_scoped(self):
        new_blueprint = copy.deepcopy(self.scoped_model.blueprint)
        same_model = scoped_resnet_sketch.ResNet("ResNet16", None, None,
                                                 depth=16, width=1, num_classes=100).cuda()
        ref = dict(new_blueprint['group_elements'])['group0']['uniques']
        ref.add('block_container')
        self.assertNotEqual(new_blueprint, self.scoped_model.blueprint)
        new_model = scoped_resnet_sketch.ResNet("ResNet16", None, new_blueprint,
                                                depth=16, width=1, num_classes=100).cuda()
        self.assertEqual(same_model.group_container, self.scoped_model.group_container)
        self.assertNotEqual(new_model.group_container, self.scoped_model.group_container)

    def test_unique_block_scoped(self):
        new_blueprint = copy.deepcopy(self.scoped_model.blueprint)
        dref = dict(dict(new_blueprint['group_elements'])['group0']['block_elements'])
        dref['block0']['uniques'].add('conv1')
        new_model = scoped_resnet_sketch.ResNet("ResNet16", None, new_blueprint,
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
        # ResNet -> group
        group = self.blueprint.get_element(0)
        # group -> block
        block = group.get_element(0)
        # block -> conv (in/out channels 16/16, kernel 3, stride 1, padding 1)
        child = block.get_element((0, 'conv'))
        self.assertEqual(child['name'], 'ResNet28/group/block/unit/conv16_16_3_1_1')
        child.make_unique()
        self.assertNotEqual(child['name'], 'ResNet28/group/block/unit/conv16_16_3_1_1')
        convdim = self.blueprint.get_element((0, 0, 'convdim'))
        self.assertEqual(convdim['name'], 'ResNet28/group/block/convdim16_16_1_1_0')
        module_list = set()

        def collect(bp, key, out):
            out.add(bp[key])

        visit_modules(self.blueprint, 'name', module_list, collect)
        self.assertTrue(child['name'] in module_list)


    #@unittest.skip("GUI test for uniqueness skipped")
    def test_visual_change_blueprinted(self):
        common.BLUEPRINT_GUI = True
        # group[0] -> block[1] -> unit[0].conv
        self.blueprint.get_element((0, 1, 0, 'conv')).make_unique()
        visualize(self.blueprint)
        new_model = blueprinted.ScopedResNet('Resnet28', self.blueprint).cuda()
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

    def test_ensemble_instead_of_conv(self):
        # replace conv0 of ResNet with Ensemble
        kwargs = {'in_channels': 3, 'out_channels': 16,
                  'kernel_size': 3, 'padding': 1, 'stride': 1}
        args = [(Conv2d, [], kwargs)] * 5
        input_shape = transformer.image_to_unsqueezed_cuda_variable(self.test_images[0][1]).size()
        conv0 = self.blueprint['conv']
        conv0['prefix'] = 'ResNet28/stacked'
        conv0['type'] = ScopedEnsemble
        conv0['iterable_args'] = args
        conv0['input_shape'] = input_shape
        conv0['output_shape'] = None
        conv0['kwargs'] = {'blueprint': conv0, 'iterable_args': args,
                           'input_shape': input_shape,
                           'output_shape': None}
        conv0.make_unique()

        # replace the 2nd conv of the 2nd block of the 3rd group with Ensemble
        input_shape = (1, 64, 32, 32)
        kwargs = copy.deepcopy(kwargs)
        kwargs['in_channels'] = 64
        kwargs['out_channels'] = 64
        args = [(Conv3d2d, [], kwargs)] * 5
        conv2 = self.blueprint.get_element((2, 1, 0, 'conv'))
        conv2['prefix'] = 'ResNet/stacked2_2_3'
        conv2['type'] = ScopedEnsemble
        conv2['iterable_args'] = args
        conv2['input_shape'] = input_shape
        conv2['output_shape'] = input_shape
        conv2['kwargs'] = {'blueprint': conv2, 'iterable_args': args,
                           'input_shape': input_shape,
                           'output_shape': input_shape}
        conv2.make_unique()
        new_model = blueprinted.ScopedResNet('ResNet28_ensemble',
                                             self.blueprint).cuda()
        for path, im in self.test_images:
            x = transformer.image_to_unsqueezed_cuda_variable(im)
            out = new_model(x)
            self.assertEqual(out.size(), self.out_size)


if __name__ == '__main__':
    unittest.main()
