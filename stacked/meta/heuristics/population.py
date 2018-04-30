from stacked.meta.blueprint import visit_modules
from stacked.modules.scoped_nn import ScopedConv2d
from stacked.models.blueprinted import ScopedResNet
from stacked.meta.heuristics import extensions
from stacked.utils.domain import ClosedList
import numpy as np


def collect_modules(blueprint):
    """Collect all module blueprints"""
    def collect(bp, key, out):
        out.add(bp[key])

    module_list = []
    visit_modules(blueprint, 'name', module_list, collect)
    return module_list


def get_conv_cost(conv_blueprint):
    """Input and output shape dependent cost for convolution"""
    input_shape = conv_blueprint['input_shape']
    output_shape = conv_blueprint['output_shape']
    return np.prod(input_shape) * np.prod(output_shape)


def estimate_cost(blueprint):
    """Calculate current cost"""
    module_list = collect_modules(blueprint)
    cost = 0
    for module in module_list:
        if module['type'] == ScopedConv2d:
            cost += get_conv_cost(module)
        else:  # other layer types, e.g. locally_connected, linear
            pass
    return cost


def utility(individual):
    """Train individual for n epochs return the cost/loss"""
    pass


class Population(object):
    def __init__(self):
        self.individuals = []


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

    individuals = []
    for i in range(population_size):
        blueprint = ScopedResNet.describe_default('ResNet',
                                                  depth=depths.pick_random()[1],
                                                  width=widths.pick_random()[1],
                                                  num_classes=10)

        visit_modules(blueprint, 0.01, [], make_mutable_and_randomly_unique)
        individuals.append(blueprint)

    p = Population()
    p.individuals = individuals
    return p

