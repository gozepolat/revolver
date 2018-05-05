import unittest
from stacked.meta.heuristics import population
from stacked.utils import transformer, common
from PIL import Image
import glob


class TestPopulation(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestPopulation, self).__init__(*args, **kwargs)

    @classmethod
    def setUpClass(cls):
        image_paths = glob.glob("images/*")
        cls.out_size = (1, 10)
        cls.test_images = [(s, Image.open(s).resize((32, 32)))
                           for s in image_paths]

    def model_run(self, model):
        for path, im in self.test_images:
            x = transformer.image_to_unsqueezed_cuda_variable(im)
            out = model(x)
            self.assertEqual(out.size(), self.out_size)

    def test_init_population(self):
        common.BLUEPRINT_GUI = False
        p = population.Population(50)
        for model in p.phenotypes:
            self.model_run(model)

    def test_estimate_cost(self):
        common.BLUEPRINT_GUI = False
        blueprints = population.generate(50)
        for bp in blueprints:
            cost = population.estimate_cost(bp)
            self.assertTrue(cost > 0)

    def test_utility(self):
        pass

    def test_evolve_population(self):
        pass
