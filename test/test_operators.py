import unittest
from PIL import Image
from stacked.models.blueprinted.resnet import ScopedResNet
from stacked.utils import transformer
from stacked.models.blueprinted.ensemble import ScopedEnsembleMean
from stacked.utils.domain import ClosedList
from stacked.meta.heuristics.operators import mutate, crossover, copyover
from stacked.meta.blueprint import visit_modules, visualize
from stacked.utils import common
import glob
import copy


def make_unique(bp, *_):
    bp.make_unique()


class TestMetaOperators(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestMetaOperators, self).__init__(*args, **kwargs)

    @classmethod
    def setUpClass(cls):
        # don't create buttons
        common.BLUEPRINT_GUI = False
        cls.out_size = (1, 100)
        image_paths = glob.glob("images/*")
        cls.blueprint = ScopedResNet.describe_default(depth=28,
                                                      width=1,
                                                      num_classes=100)
        cls.test_images = [(s, Image.open(s).resize((128, 128))) for s in image_paths]

    def test_index(self):
        common.BLUEPRINT_GUI = False
        conv = self.blueprint.get_element([0, 1, 0, 'conv'])
        conv.make_unique()
        convdim = self.blueprint.get_element((0, 0, 'convdim'))
        self.assertEqual(conv.get_index_from_root(), [0, 1, 0, 'conv'])
        self.assertEqual(convdim.get_index_from_root(), [0, 0, 'convdim'])

    def test_mutate(self):
        common.BLUEPRINT_GUI = False
        old_conv = copy.deepcopy(self.blueprint['conv'])
        conv0 = ScopedEnsembleMean.describe_from_blueprint('ensemble', '',
                                                           self.blueprint['conv'])

        conv0_alternatives = [self.blueprint['conv'], conv0]
        self.blueprint['mutables'] = {
            'conv': ClosedList(conv0_alternatives)
        }

        for i in range(100):
            mutate(self.blueprint, 'conv', 1.0, 0.95)
            if self.blueprint['conv'] != old_conv:
                break

        self.assertNotEqual(self.blueprint['conv'], old_conv)

    def model_run(self, blueprint):
        # run and test a model created from the blueprint
        blueprint.make_common()
        blueprint.make_unique()
        model = ScopedResNet(blueprint['name'], blueprint).cuda()
        for path, im in self.test_images:
            x = transformer.image_to_unsqueezed_cuda_variable(im)
            out = model(x)
            self.assertEqual(out.size(), self.out_size)

    def test_crossover(self):
        common.BLUEPRINT_GUI = True
        if common.GUI is None:
            from tkinter import Tk
            common.GUI = Tk()

        blueprint1 = ScopedResNet.describe_default('ResNet_28', depth=28,
                                                   width=1, num_classes=100)
        blueprint1_bk = {k: v for k, v in blueprint1.items()}
        blueprint2 = ScopedResNet.describe_default('ResNet_40', depth=40,
                                                   width=1, num_classes=100)
        blueprint3 = ScopedResNet.describe_default('ResNet_22', depth=22,
                                                   width=1, num_classes=100)
        blueprint4 = ScopedResNet.describe_default('ResNet_46', depth=46,
                                                   width=1, num_classes=100)

        # visit all module blueprints and mark them unique
        visit_modules(blueprint2, None, [], make_unique)

        is_crossed = False
        for i in range(500):
            if crossover(blueprint1, blueprint2):
                is_crossed = True
            if crossover(blueprint2, blueprint1):
                is_crossed = True
            if crossover(blueprint3, blueprint4):
                is_crossed = True
            if crossover(blueprint3, blueprint1):
                is_crossed = True

        self.assertTrue(is_crossed)
        self.assertNotEqual(blueprint1, blueprint1_bk)
        self.model_run(blueprint1)
        self.model_run(blueprint2)
        self.model_run(blueprint3)
        self.model_run(blueprint4)

        visualize(blueprint3)
        common.GUI = None
        common.BLUEPRINT_GUI = False

    def test_copyover(self):
        common.BLUEPRINT_GUI = False
        blueprint1 = ScopedResNet.describe_default('ResNet28x', depth=28,
                                                   width=1, num_classes=100)
        blueprint1_bk = copy.deepcopy(blueprint1)
        blueprint2 = ScopedResNet.describe_default('ResNet46x', depth=46,
                                                   width=1, num_classes=100)

        # visit all module blueprints and mark them unique
        visit_modules(blueprint2, None, [], make_unique)

        is_crossed = False
        for i in range(500):
            if copyover(blueprint1, blueprint2):
                is_crossed = True
            if copyover(blueprint2, blueprint1):
                is_crossed = True

        self.assertTrue(is_crossed)
        self.assertNotEqual(blueprint1, blueprint1_bk)
        self.model_run(blueprint1)
        self.model_run(blueprint2)


if __name__ == '__main__':
    unittest.main()
