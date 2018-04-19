# -*- coding: utf-8 -*-
import unittest
from PIL import Image
from stacked.utils import transformer
from stacked.models import resnet
from stacked.models import scoped_resnet
from stacked.models import blueprinted_resnet
from stacked.meta.blueprint import visualize, get_module_names
import glob
import copy


class TestResNet(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestResNet, self).__init__(*args, **kwargs)

    @classmethod
    def setUpClass(cls):
        image_paths = glob.glob("images/*")
        cls.out_size = (1, 100)
        cls.test_images = [(s, Image.open(s).resize((128, 128))) for s in image_paths]
        cls.vanilla_model = resnet.ResNet(depth=16, width=1, num_classes=100).cuda()
        cls.scoped_model = scoped_resnet.ResNet("ResNet16", None, None,
                                                depth=16, width=1, num_classes=100).cuda()
        cls.blueprint = blueprinted_resnet.ScopedResNet.describe_default(depth=28, width=1, num_classes=100)
        cls.blueprinted_model = blueprinted_resnet.ScopedResNet(cls.blueprint['name'], cls.blueprint).cuda()

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
        self.assertEqual(child['name'], 'ResNet/group16_16/block16_16/conv16_16_3')
        child.make_unique()
        self.assertNotEqual(child['name'], 'ResNet/group16_16/block16_16/conv16_16_3')
        print(child['name'])
        module_list = set()
        get_module_names(self.blueprint, module_list)
        self.assertTrue(child['name'] in module_list)

    @unittest.skip("GUI test for uniqueness skipped")
    def test_visualize_blueprinted(self):
        visualize(self.blueprint)


if __name__ == '__main__':
    unittest.main()
