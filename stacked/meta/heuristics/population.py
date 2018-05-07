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
import copy


def log(log_func, msg):
    if common.DEBUG_POPULATION:
        log_func("stacked.meta.heuristics.population: %s" % msg)


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
    unique_size = len(set(names))
    return float(unique_size) / all_size


def get_score(genotype, *_, **__):
    """Estimate a score only by looking at the genotype"""
    unique_ratio = get_share_ratio(genotype)
    cost = estimate_cost(genotype)
    return unique_ratio * cost


def update_score(blueprint, new_score, weight=0.2):
    modules = collect_modules(blueprint)
    for bp in modules:
        if 'score' not in bp['meta']:
            bp['meta']['score'] = np.inf
        score = bp['meta']['score']
        if score < np.inf:
            score = new_score * weight + score * (1.0 - weight)
        else:
            score = new_score
        bp['meta']['score'] = score


def make_mutable_and_randomly_unique(bp, p_unique, _):
    extensions.extend_conv_mutables(bp, ensemble_size=3,
                                    block_depth=2)
    extensions.extend_depth_mutables(bp)

    if np.random.random() < p_unique:
        bp.make_unique()


def generate(population_size):
    max_width = 4
    max_depth = 28

    depths = ClosedList(list(range(16, max_depth + 1, 6)))
    widths = ClosedList(list(range(1, max_width + 1)))
    blueprints = []

    for i in range(population_size):
        depth = depths.pick_random()[1]
        width = widths.pick_random()[1]
        blueprint = ScopedResNet.describe_default('ResNet', str(i),
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
        self.ids = []
        self.phenotypes = []
        self.population_size = population_size
        self.iteration = 1
        self.generate_new(population_size)

    def replace_individual(self, index, genotype, phenotype=None):
        self.genotypes[index] = genotype
        self.ids[index] = id(genotype)

        if phenotype is None:
            phenotype = ScopedResNet(genotype['name'],
                                     genotype).cuda()
        self.phenotypes[index] = phenotype

    def add_individual(self, blueprint, phenotype=None):
        if id(blueprint) in set(self.ids):
            log(warning, "Individual %s is already in population"
                % blueprint['name'])
            return

        self.genotypes.append(blueprint)
        self.ids.append(id(blueprint))

        if phenotype is None:  # cuda
            phenotype = ScopedResNet(blueprint['name'], blueprint).cuda()

        self.phenotypes.append(phenotype)

    def generate_new(self, population_size):
        genotypes = generate(population_size)
        for blueprint in genotypes:
            self.add_individual(blueprint)

    def update_scores(self, iteration, score_fn=get_score):
        """Accummulate scores with lower and lower weights"""
        weight = 0.4 / math.log(iteration)
        for bp, model in zip(self.genotypes, self.phenotypes):
            new_score = score_fn(bp, model)
            update_score(bp, new_score, weight=weight)

    def get_average_score(self):
        total_score = 0.0
        n = 1

        assert(len(self.genotypes) > 0)
        for bp in self.genotypes:
            if 'score' in bp['meta']:
                score = bp['meta']['score']
                if score < np.inf:
                    total_score += score
                    n += 1

        return total_score / n

    def get_the_best_index(self):
        best_score = np.inf
        index = 0

        assert(len(self.genotypes) > 0)
        for i, bp in enumerate(self.genotypes):
            if 'score' in bp['meta']:
                score = bp['meta']['score']
                if score < best_score:
                    index = i
                    best_score = score
        return index

    def get_max_indices(self):
        """Get the indices of the worst two individuals"""
        min_score2 = min_score = -np.inf
        i, r1, r2 = (0, 0, 0)

        for bp in self.genotypes:
            score = bp['meta']['score']
            if min_score < score:
                min_score2 = min_score
                min_score = score
                r1 = i
                r2 = r1
            elif min_score2 < score:
                min_score2 = score
                r2 = i
            i += 1

        return r1, r2

    def evolve_generation(self, score_fn=get_score):
        """A single step of evolution"""
        if len(self.genotypes) == 0:
            self.generate_new(self.population_size)

        self.iteration += 1
        self.update_scores(self.iteration, score_fn)

        # indices for replacement with new individuals
        r1, r2 = self.get_max_indices()

        selected_indices = [i for i in range(self.population_size)
                            if i not in (r1, r2)]
        index1, index2 = np.random.choice(selected_indices,
                                          2, replace=False)

        clone1 = copy.deepcopy(self.genotypes[index1])
        clone2 = copy.deepcopy(self.genotypes[index2])

        visit_modules(clone1, 0.1, [],
                      make_mutable_and_randomly_unique)

        visit_modules(clone2, 0.1, [],
                      make_mutable_and_randomly_unique)

        mutate(clone1, p=0.5)
        mutate(clone2, p=0.5)

        if np.random.random() < 0.8:
            crossover(clone1, clone2)
            self.replace_individual(r2, clone1)
        else:
            copyover(clone1, clone2)

        self.replace_individual(r1, clone2)










