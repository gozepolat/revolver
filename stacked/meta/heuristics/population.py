from stacked.meta.blueprint import visit_modules, \
    collect_keys, collect_modules
from stacked.models.blueprinted.resnet import ScopedResNet
from stacked.modules.conv import Conv3d2d
from torch.nn import Conv2d, Linear
from stacked.meta.heuristics.operators import mutate, \
    crossover, copyover
from stacked.meta.heuristics import extensions
from stacked.utils.domain import ClosedList
from stacked.utils import common
from logging import warning
import numpy as np
import math


def log(log_func, msg):
    if common.DEBUG_POPULATION:
        log_func(msg)


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
        if (issubclass(_type, Conv2d) or
                issubclass(_type, Linear) or
                issubclass(_type, Conv3d2d)):
            cost += get_layer_cost(module)
        # ignore other layer types, e.g. locally_connected

    return cost


def get_share_ratio(blueprint):
    """Collect module names and get the ratio of the same names"""
    names = collect_keys(blueprint, 'name')
    all_size = len(names)
    shared_size = all_size - len(set(names))
    return float(shared_size) / all_size


def utility(individual, *_, **__):
    """Train individual for n epochs return the cost/loss"""
    share_ratio = get_share_ratio(individual)
    cost = estimate_cost(individual)
    return share_ratio / cost


def update_score(blueprint, new_score, weight=0.2):
    modules = collect_modules(blueprint)
    for bp in modules:
        if 'score' not in bp['meta']:
            bp['meta']['score'] = 0
        score = bp['meta']['score']
        score = new_score * weight + score * (1.0 - weight)
        bp['meta']['score'] = score


def generate(population_size):
    max_width = 4
    max_depth = 28

    depths = ClosedList(list(range(16, max_depth + 1, 6)))
    widths = ClosedList(list(range(1, max_width + 1)))

    def make_mutable_and_randomly_unique(bp, p_unique, _):
        if 'conv' in bp:
            extensions.extend_conv_mutables(bp, ensemble_size=3,
                                            block_depth=2)
        if np.random.random() < p_unique:
            bp.make_unique()

    blueprints = []
    for i in range(population_size):
        depth = depths.pick_random()[1]
        width = widths.pick_random()[1]
        blueprint = ScopedResNet.describe_default('ResNet',
                                                  depth=depth,
                                                  width=width,
                                                  num_classes=10)

        visit_modules(blueprint, 0.01, [],
                      make_mutable_and_randomly_unique)
        blueprints.append(blueprint)

    return blueprints


class Population(object):
    def __init__(self, population_size):
        assert(population_size > 3)
        self.genotypes = []
        self.uuids = set()
        self.phenotypes = []
        self.population_size = population_size
        self.iteration = 1
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

    def update_utility_scores(self, iteration):
        # accummulate scores with lower and lower weights
        weight = 0.4 / math.log(iteration)
        for bp, model in zip(self.genotypes, self.phenotypes):
            new_score = utility(bp, model)
            update_score(bp, new_score, weight=weight)

    def get_min_indices(self):
        # get the indices of the worst two individuals
        min_score2 = min_score = 10 ** 20
        i, r1, r2 = (0, 0, 0)
        for bp in self.genotypes:
            score = bp['meta']['score']
            if min_score > score:
                min_score2 = min_score
                min_score = score
                r1 = i
                r2 = r1
            elif min_score2 > score:
                min_score2 = score
                r2 = i
            i += 1
        return r1, r2

    def evolve_generation(self):
        if len(self.genotypes) == 0:
            self.generate_new(self.population_size)

        self.iteration += 1
        self.update_utility_scores(self.iteration)

        r1, r2 = self.get_min_indices()
        indices = np.random.choice(self.population_size,
                                   2, replace=False)

        clone1 = self.genotypes[indices[0]].clone()
        clone2 = self.genotypes[indices[1]].clone()

        clone1.mutate()
        clone2.mutate()
        if np.random.random() < 0.9:
            crossover(clone1, clone2)
            self.genotypes[r2] = clone1
            self.uuids[r2] = clone1.uuid
            self.phenotypes[r2] = ScopedResNet(clone1['name'],
                                               clone1).cuda()
        else:
            copyover(clone1, clone2)

        self.genotypes[r1] = clone2
        self.uuids[r1] = clone2.uuid
        self.phenotypes[r2] = ScopedResNet(clone2['name'],
                                           clone2).cuda()










