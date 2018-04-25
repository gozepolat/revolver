import unittest
from stacked.models.blueprinted import ScopedResNet
from stacked.utils import transformer
from torch.nn import Conv2d
from stacked.modules.scoped_nn import ScopedEnsemble
from stacked.utils.domain import ClosedList
from stacked.meta.heuristics import mutate, crossover
from stacked.utils import common
import copy


class TestMetaHeuristics(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestMetaHeuristics, self).__init__(*args, **kwargs)

    @classmethod
    def setUpClass(cls):
        # don't create buttons
        common.BLUEPRINT_GUI = False
        cls.blueprint = ScopedResNet.describe_default(depth=28,
                                                      width=1,
                                                      num_classes=100)

    def test_index(self):
        common.BLUEPRINT_GUI = False
        conv = self.blueprint.get_element([0, 1, 1, 'conv'])
        conv.make_unique()
        convdim = self.blueprint.get_element((0, 0, 'convdim'))
        self.assertEqual(conv.get_index_from_root(), [0, 1, 1, 'conv'])
        self.assertEqual(convdim.get_index_from_root(), [0, 0, 'convdim'])

    def test_mutate(self):
        common.BLUEPRINT_GUI = False
        kwargs = {'in_channels': 3, 'out_channels': 16,
                  'kernel_size': 3, 'padding': 1, 'stride': 1}
        args = [(Conv2d, [], kwargs)] * 5

        old_conv = copy.deepcopy(self.blueprint['conv'])
        conv0 = copy.deepcopy(self.blueprint['conv'])
        conv0['prefix'] = 'ResNet/stacked0'
        conv0['type'] = ScopedEnsemble
        conv0['kwargs'] = {'iterable_args': args,
                           'inp_shape': (1, 3, 128, 128)}
        conv0.make_unique()

        conv0_alternatives = [self.blueprint['conv'], conv0]
        self.blueprint['mutables'] = {
            'conv': ClosedList(conv0_alternatives)
        }

        for i in range(100):
            mutate(self.blueprint, 'conv', 1.0, 0.95)
            if self.blueprint['conv'] != old_conv:
                break

        self.assertNotEqual(self.blueprint['conv'], old_conv)

    def test_crossover(self):
        common.BLUEPRINT_GUI = False
        blueprint1 = ScopedResNet.describe_default('ResNet28', depth=28,
                                                   width=1, num_classes=100)
        blueprint1_bk = copy.deepcopy(blueprint1)
        blueprint2 = ScopedResNet.describe_default('ResNet40', depth=40,
                                                   width=1, num_classes=100)
        is_crossed = False
        for i in range(100):
            if crossover(blueprint1, blueprint2):
                is_crossed = True

        self.assertTrue(is_crossed)
        self.assertNotEqual(blueprint1, blueprint1_bk)


if __name__ == '__main__':
    unittest.main()
