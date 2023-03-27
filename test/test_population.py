import unittest
from stacked.meta.heuristics import population
from stacked.utils import transformer, common, usage_helpers
from stacked.meta.blueprint import make_module
from PIL import Image
import glob
from stacked.models.blueprinted.resnet import ScopedResNet
from stacked.models.blueprinted.denseconcatgroup import ScopedDenseConcatGroup
from stacked.models.blueprinted.bottleneckblock import ScopedBottleneckBlock
from stacked.modules.scoped_nn import ScopedCrossEntropyLoss, ScopedConv2d
from stacked.utils.transformer import all_to_none
import argparse


def get_options():
    parsed = argparse.ArgumentParser()
    parsed.add_argument('test.test_population', default='')
    options = parsed.parse_args()

    options.dataset = 'MNIST'
    options.num_classes = 10
    options.block_depth = 2
    options.population_size = 4
    options.num_samples = 256
    options.lr_decay_ratio = 0.1
    options.crop_size = 28
    options.weight_decay = 0.0001
    options.group_depths = None
    options.input_shape = (8, 1, 28, 28)
    options.skeleton = (3, 3, 3)
    options.batch_size = 8
    options.lr = 0.1
    options.epochs = 3
    options.lr_drop_epochs = (2, 3, 4)
    options.gpu_id = 0
    options.save_folder = ''
    options.load_path = ''
    options.save_png_folder = ''
    options.num_thread = 2
    options.width = 1
    options.depth = 22
    options.mode = 'population_train'
    options.net = ScopedResNet
    options.conv_module = ScopedConv2d
    # log base for the number of parameters
    options.params_favor_rate = 100

    options.epoch_per_generation = 1
    options.dropout_p = 0.0
    options.drop_p = 0.5
    options.fractal_depth = 4
    options.net = ScopedResNet
    options.callback = all_to_none
    options.criterion = ScopedCrossEntropyLoss
    options.residual = False
    options.group_module = ScopedDenseConcatGroup
    options.block_module = ScopedBottleneckBlock
    options.dense_unit_module = ScopedBottleneckBlock
    options.head_kernel = 3
    options.head_stride = 1
    options.head_padding = 1
    options.head_pool_kernel = 3
    options.head_pool_stride = 2
    options.head_pool_padding = 1
    options.head_modules = ('conv', 'bn')
    options.unique = ('bn',)

    # number of updated individuals per generation
    options.sample_size = 2
    options.update_score_weight = 0.2
    options.max_iteration = 3

    # default heuristics
    options.generator = population.generate_net_blueprints
    options.utility = population.get_phenotype_score
    options.engine_maker = usage_helpers.create_single_engine

    # disable engine loading, and tqdm
    options.load_path = ''
    options.use_tqdm = True
    return options


class TestPopulation(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestPopulation, self).__init__(*args, **kwargs)

    @classmethod
    def setUpClass(cls):
        image_paths = glob.glob("images/*")
        cls.out_size = (1, 10)
        cls.test_images = [(s, Image.open(s).convert('L').resize((28, 28)))
                           for s in image_paths]
        cls.options = get_options()

    def model_run(self, model):
        for path, im in self.test_images:
            x = transformer.image_to_unsqueezed_cuda_variable(im)
            out = model(x)
            self.assertEqual(out.size(), self.out_size)

    @unittest.skip("Skipped due to slow runtime")
    def test_init_population(self):
        common.BLUEPRINT_GUI = False
        p = population.Population(self.options)
        for bp in p.genotypes:
            self.model_run(make_module(bp))

    @unittest.skip("Skipped due to slow runtime")
    def test_estimate_cost(self):
        common.BLUEPRINT_GUI = False
        blueprints = population.generate_net_blueprints(self.options)
        for bp in blueprints:
            cost = population.estimate_rough_contexts(bp)
            self.assertTrue(cost > 0)

    @unittest.skip("Skipped due to slow runtime")
    def test_evolve_population_once(self):
        common.BLUEPRINT_GUI = False
        p = population.Population(self.options)
        names = [bp['name'] for bp in p.genotypes]
        new_names = [bp['name'] for bp in p.genotypes]
        self.assertEqual(names, new_names)
        p.evolve_generation()
        new_names = [bp['name'] for bp in p.genotypes]
        self.assertNotEqual(names, new_names)
        for bp in p.genotypes:
            self.model_run(make_module(bp))

    def test_pick_indices(self):
        common.BLUEPRINT_GUI = False
        self.options.population_size = 500
        p = population.Population(self.options)
        p.pick_indices(100)

    @unittest.skip("Slow")
    def test_population_convergence(self):
        common.BLUEPRINT_GUI = False
        p = population.Population(self.options)
        index = p.get_the_best_index()
        best_init = p.genotypes[index]['meta']['score']

        for i in range(2):
            p.evolve_generation()
            print('Population generation: %d' % i)

        index = p.get_the_best_index()
        best_final = p.genotypes[index]['meta']['score']

        print(best_final, best_init)
        self.assertTrue(best_init > best_final)

