from stacked.meta.blueprint import visit_modules
from stacked.modules.scoped_nn import ScopedConv2d, ScopedLinear
from stacked.models.blueprinted.resnet import ScopedResNet
from stacked.meta.heuristics import extensions
from stacked.utils.domain import ClosedList
from stacked.utils import common
from logging import warning
import numpy as np


def log(log_func, msg):
    if common.DEBUG_POPULATION:
        log_func(msg)


def collect_modules(blueprint):
    """Collect all module blueprints"""
    def collect(bp, _, out):
        out.append(bp)

    module_list = []
    visit_modules(blueprint, None, module_list, collect)
    return module_list


def get_layer_cost(blueprint):
    """Input and output shape dependent cost for convolution"""
    input_shape = blueprint['input_shape']
    output_shape = blueprint['output_shape']
    return np.prod(input_shape) * np.prod(output_shape)


def get_ensemble_cost(blueprint):
    """Input / output shape, and size dependent cost for ensemble"""
    input_shape = blueprint['input_shape']
    output_shape = blueprint['output_shape']
    n = len(blueprint['iterable_args'])
    return np.prod(input_shape) * np.prod(output_shape) * n * 3


def estimate_cost(blueprint):
    """Calculate current cost"""
    module_list = collect_modules(blueprint)
    cost = 0
    for module in module_list:
        _type = module['type']
        if _type == ScopedConv2d or _type == ScopedLinear:
            cost += get_layer_cost(module)
        else:  # other layer types, e.g. locally_connected, ensemble
            pass
    return cost


def utility(individual):
    """Train individual for n epochs return the cost/loss"""
    pass


class Population(object):
    def __init__(self, population_size):
        self.genotypes = []
        self.uuids = set()
        self.phenotypes = []
        self.generate_new(population_size)

    def add_individual(self, blueprint, phenotype=None):
        if blueprint.uuid in self.uuids:
            log(warning, "Individual %s is already in population"
                % blueprint['name'])
            return

        self.genotypes.append(blueprint)
        self.uuids.add(blueprint.uuid)

        if phenotype is None:  # cuda
            phenotype = ScopedResNet(blueprint['name'], blueprint).cuda()

        self.phenotypes.append(phenotype)

    def generate_new(self, population_size):
        genotypes = generate(population_size)
        for blueprint in genotypes:
            self.add_individual(blueprint)


def generate(population_size):
    max_width = 4
    max_depth = 28

    depths = ClosedList(list(range(16, max_depth + 1, 6)))
    widths = ClosedList(list(range(1, max_width + 1)))

    def make_mutable_and_randomly_unique(bp, p_unique, _):
        if 'conv' in bp:
            extensions.extend_conv_mutables(bp, ensemble_size=3, block_depth=2)
        if np.random.random() < p_unique:
            bp.make_unique()

    blueprints = []
    for i in range(population_size):
        blueprint = ScopedResNet.describe_default('ResNet',
                                                  depth=depths.pick_random()[1],
                                                  width=widths.pick_random()[1],
                                                  num_classes=10)

        visit_modules(blueprint, 0.01, [], make_mutable_and_randomly_unique)
        blueprints.append(blueprint)

    return blueprints

