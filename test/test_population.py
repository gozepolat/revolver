import unittest
from stacked.meta.heuristics import population
from stacked.models import blueprinted
from stacked.utils import transformer


class TestPopulation(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestPopulation, self).__init__(*args, **kwargs)

    @classmethod
    def setUpClass(cls):
        pass

    def model_run(self, blueprint):
        # run and test a model created from the blueprint
        model = blueprinted.ScopedResNet(blueprint['name'],
                                         blueprint).cuda()
        for path, im in self.test_images:
            x = transformer.image_to_unsqueezed_cuda_variable(im)
            out = model(x)
            self.assertEqual(out.size(), self.out_size)

    def test_init_population(self):
        p = population.generate(100)
        for i in p.individuals:
            self.model_run(i)

    def test_estimate_cost(self):
        pass

    def test_utility(self):
        pass

    def test_evolve_population(self):
        pass
