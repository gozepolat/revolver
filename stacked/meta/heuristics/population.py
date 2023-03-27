from stacked.meta.blueprint import visit_modules, \
    collect_keys, collect_modules, make_module
from stacked.models.blueprinted.resnet import ScopedResNet
from stacked.models.blueprinted.densenet import ScopedDenseNet
from stacked.models.blueprinted.convdeconv import ScopedConv2dDeconv2d
from stacked.modules.scoped_nn import ScopedConv2d
from stacked.modules.conv import Conv3d2d
from stacked.meta.scope import unregister
from stacked.models.blueprinted.bottleneckblock import ScopedBottleneckBlock
from stacked.models.blueprinted.resbottleneckblock import ScopedResBottleneckBlock
from stacked.models.blueprinted.resblock import ScopedResBlock
from stacked.models.blueprinted.resgroup import ScopedResGroup
from stacked.models.blueprinted.denseconcatgroup import ScopedDenseConcatGroup
from stacked.models.blueprinted.densesumgroup import ScopedDenseSumGroup
from torch.nn import Conv2d, Linear
from stacked.meta.heuristics.operators import mutate, \
    crossover, copyover
from stacked.meta.heuristics import extensions
from stacked.utils.domain import ClosedList
from stacked.utils.engine import get_num_parameters
from stacked.utils import common
from stacked.utils.transformer import softmax
from stacked.utils.usage_helpers import make_net_blueprint
from stacked.models.blueprinted.meta import ScopedMetaMasked
from stacked.models.blueprinted.separable import ScopedDepthwiseSeparable
from logging import warning, exception
import numpy as np
import copy
import math


def log(log_func, msg):
    if common.DEBUG_POPULATION:
        log_func("stacked.meta.heuristics.population: %s" % msg)


def get_layer_cost(blueprint):
    """Input and output shape dependent cost for convolution"""
    input_shape = blueprint['input_shape']
    output_shape = blueprint['output_shape']
    return math.log2(math.sqrt(np.prod(input_shape) * np.prod(output_shape)))


def get_ensemble_cost(blueprint):
    """Input / output shape, and size dependent cost for ensemble"""
    input_shape = blueprint['input_shape']
    output_shape = blueprint['output_shape']
    n = len(blueprint['iterable_args'])
    return math.log(np.prod(input_shape) * np.prod(output_shape) * n * 3)


def get_num_params(blueprint):
    if 'kwargs' in blueprint:
        c = blueprint['kwargs']
        if 'in_channels' in c:
            return c['in_channels'] * c['out_channels'] * c['kernel_size'] * c['kernel_size'] + c['out_channels']
        elif 'in_features' in c:
            return (c['in_features'] + 1) * c['out_features']

    raise ValueError(f"can not calculate num_params in: {blueprint}")


def estimate_rough_contexts(blueprint):
    """Calculate current rough number of contexts"""
    module_list = collect_modules(blueprint)
    cost = 0
    total_params = 0
    total_weights = 0
    name_set = set()
    for module in module_list:
        _type = module['type']
        if (issubclass(_type, Conv2d) or
                issubclass(_type, Linear) or
                issubclass(_type, Conv3d2d)):
            cost += get_layer_cost(module)
            params = get_num_params(module)
            if module['name'] not in name_set:
                total_params += params
                name_set.add(module['name'])
            total_weights += params
        # ignores other layer types, e.g. locally_connected
    if total_weights <= 0:
        print(f"Zero : {blueprint}")
        return 0.01
    return math.log2(total_params * cost / total_weights) * common.POPULATION_COST_ESTIMATION_SCALE


def get_share_ratio(blueprint):
    """Collect module names and get the ratio of the same names"""
    names = collect_keys(blueprint, 'name')
    all_size = len(names)
    unique_size = len(set(names))
    return float(unique_size) / all_size


def get_genotype_score(genotype, *_, **__):
    """Estimate a score only by looking at the genotype"""
    utility = np.clip(estimate_rough_contexts(genotype), 0, None)
    print(f"genotype utility: {utility}, score: {20./(utility+1)}")
    return 20.0 / (utility + 1.)
    # unique_ratio = get_share_ratio(genotype)
    # cost = estimate_cost(genotype)
    # return unique_ratio * cost


def get_phenotype_score(genotype, options):
    """Estimate score by training / testing the model"""
    engine_maker = options.engine_maker
    n_samples = options.num_samples // 100
    epoch = options.epoch_per_generation
    favor_params = options.params_favor_rate
    engine = engine_maker(genotype, options)

    log(warning, "Getting score for the phenotype {} with id: {}".format(genotype['name'],
                                                                         id(engine.net.blueprint)))
    log(warning, "=====================")
    log(warning, engine.net)
    log(warning, "=====================")

    for j in range(epoch):
        engine.start_epoch()
        engine.train_n_samples(n_samples)
        if j == epoch - 1:
            engine.end_epoch()
        else:
            engine.state['epoch'] += 1

    engine.hook('on_end', engine.state)

    # favor lower number of parameters
    num_parameters = get_num_parameters(engine.state['network'].net)
    num_parameters = math.log(num_parameters, favor_params)
    score = engine.state['score'] * (1.0 + num_parameters
                                     * common.POPULATION_COST_NUM_PARAMETER_SCALE)
    print(f"phenotype score: {score}")
    # create / delete the engine each time to save memory
    unregister(engine.blueprint['name'])

    return score


def update_score(blueprint, new_score, weight=0.2):
    """Update the score of blueprint components, and get the avg"""
    modules = collect_modules(blueprint)
    average_score = 0.0

    for bp in modules:
        if 'score' not in bp['meta']:
            bp['meta']['score'] = get_genotype_score(bp)
        score = bp['meta']['score']
        if score < np.inf:
            score = new_score * weight + score * (1.0 - weight)
        else:
            score = new_score
        bp['meta']['score'] = score
        average_score += score

    # the full hierarchy
    bp = blueprint
    if 'score' not in bp['meta']:
        bp['meta']['score'] = get_genotype_score(bp)

    print(f"Previous score {bp['meta']['score']}")
    bp['meta']['score'] = new_score * weight + bp['meta']['score'] * (1.0 - weight)
    print(f"New score {bp['meta']['score']}")

    return average_score


def make_mutable_and_randomly_unique(bp, p_unique, *_, **__):
    extensions.extend_conv_mutables(bp, ensemble_size=3,
                                    block_depth=2)
    extensions.extend_depth_mutables(bp)
    extensions.extend_mutation_mutables(bp)

    if np.random.random() < p_unique:
        bp.make_unique()


def generate_net_blueprints(options):
    """Randomly generate genotypes"""
    max_width = options.width
    max_depth = options.depth
    population_size = options.population_size

    depths = ClosedList(list(range(22, max_depth + 1, 6)))
    widths = ClosedList(list(range(1, max_width + 1)))
    conv_module = ClosedList([ScopedDepthwiseSeparable, ScopedConv2d, ScopedConv2dDeconv2d])
    residual = ClosedList([True, False])
    skeleton = ClosedList([(3, 6, 12), (6, 6, 6), (6, 12, 12)])
    block_module = ClosedList([ScopedBottleneckBlock, ScopedResBlock, ScopedResBottleneckBlock])
    group_module = ClosedList([ScopedDenseConcatGroup, ScopedDenseSumGroup, ScopedResGroup])
    drop_p = ClosedList([0, 0.1, 0.25, 0.5])
    block_depth = ClosedList([1, 2])
    nets = ClosedList([ScopedResNet, ScopedDenseNet])

    blueprints = []

    for i in range(population_size):
        options.depth = depths.pick_random()[1]
        options.width = widths.pick_random()[1]
        options.net = nets.pick_random()[1]
        options.block_module = block_module.pick_random()[1]
        options.group_module = group_module.pick_random()[1]
        options.conv_module = conv_module.pick_random()[1]
        options.residual = residual.pick_random()[1]
        options.skeleton = skeleton.pick_random()[1]
        options.drop_p = drop_p.pick_random()[1]
        options.block_depth = block_depth.pick_random()[1]

        blueprint = make_net_blueprint(options, str(i))
        visit_modules(blueprint, 0.01, [],
                      make_mutable_and_randomly_unique)
        blueprints.append(blueprint)

    return blueprints


class Population(object):
    def __init__(self, options):
        assert (options.population_size > 3)
        self.options = options
        self.genotypes = []
        self.ids = []
        self.population_size = 0
        self.iteration = 1
        self.generate_new()

    def replace_individual(self, index, blueprint):
        """Replace the individual at the given index with a new one"""
        if 'score' not in blueprint['meta']:
            blueprint['meta']['score'] = get_genotype_score(blueprint)

        self.genotypes[index] = blueprint
        self.ids[index] = id(blueprint)

    def add_individual(self, blueprint):
        """Add a single individual to the population"""
        if id(blueprint) in set(self.ids):
            log(warning, "Individual %s is already in population"
                % blueprint['name'])
            return

        if 'score' not in blueprint['meta']:
            blueprint['meta']['score'] = get_genotype_score(blueprint)

        self.genotypes.append(blueprint)
        self.ids.append(id(blueprint))
        self.population_size += 1

    def generate_new(self):
        """Randomly generate genotypes and then create individuals"""
        genotypes = self.options.generator(self.options)

        for blueprint in genotypes:
            self.add_individual(blueprint)

    def pick_indices(self, sample_size=0):
        """Randomly pick genotype indices, sometimes favor lower scores"""
        if sample_size == 0:
            sample_size = self.options.sample_size

        p = 0.7 - 0.4 * float(self.iteration) / self.options.max_iteration
        if np.random.random() < p:
            return np.random.choice(range(len(self.genotypes)),
                                    sample_size, replace=False)

        distribution = np.array([bp['meta']['score'] for bp in self.genotypes])
        transformed = np.max(distribution) - distribution * 0.5

        distribution = transformed / np.sum(transformed)
        # distribution = softmax()

        if np.count_nonzero(distribution) <= sample_size:
            log(warning, "Population.pick_indices: Scores are not diverse enough")
            return np.random.choice(len(self.genotypes),
                                    sample_size, replace=False)

        try:
            choice = np.random.choice(len(self.genotypes),
                                      sample_size, p=distribution, replace=False)
            return choice

        except ValueError:
            exception("Population.pick_indices: Caught exception with weighted choice")
            return np.random.choice(len(self.genotypes),
                                    sample_size, replace=False)

    def update_scores(self):
        """Evaluate and improve a portion of the population according to the scores"""
        weight = self.options.update_score_weight
        indices = self.pick_indices()
        for i in indices:
            bp = self.genotypes[i]

            # penalize any individual that caused an exception
            new_score = 1e20
            try:
                new_score = self.options.utility(bp, self.options)
            except (RuntimeError, ValueError):
                exception("Population.update_scores: Caught exception when scoring the model")
                log(warning, "This individual caused exception: %s" % bp)

            update_score(bp, new_score, weight=weight)

    def get_average_score(self):
        """Get average score for the population"""
        total_score = 0.0
        n = 1

        for bp in self.genotypes:
            if 'score' in bp['meta']:
                score = bp['meta']['score']
                if score < np.inf:
                    total_score += score
                    n += 1

        if total_score == 0:
            return np.inf

        return total_score / n

    def get_the_best_index(self):
        """Get the best scoring individual in the population"""
        best_score = np.inf
        index = 0

        assert (len(self.genotypes) > 0)
        for i, bp in enumerate(self.genotypes):
            if 'score' in bp['meta']:
                score = bp['meta']['score']
                if score < best_score:
                    index = i
                    best_score = score
        return index

    def get_max_indices(self):
        """Get the indices of the worst two individuals"""
        score2 = score1 = -np.inf
        r1 = r2 = 0

        for i, bp in enumerate(self.genotypes):
            if 'score' in bp['meta']:
                score = bp['meta']['score']

                if score1 < score:
                    score2 = score1
                    r2 = r1
                    score1 = score
                    r1 = i
                elif score2 < score:
                    score2 = score
                    r2 = i

        return r1, r2

    def evolve_generation(self, options=None):
        """A single step of evolution"""
        if options is not None:
            self.options = options

        if len(self.genotypes) == 0:
            self.generate_new()

        self.iteration += 1
        self.update_scores()

        # bad scored indices will be replaced with new individuals
        r1, r2 = self.get_max_indices()

        # favor weighted pick eventually
        index1, index2 = self.pick_indices(2)

        if index1 in (r1, r2) or index2 in (r1, r2):
            selected_indices = [i for i in range(self.population_size)
                                if i not in (r1, r2)]
            index1, index2 = np.random.choice(selected_indices, 2, replace=False)

        clone1 = copy.deepcopy(self.genotypes[index1])
        clone2 = copy.deepcopy(self.genotypes[index2])

        # restrict gpu memory usage
        gpu_usage_dict = common.get_gpu_memory_info()
        (used, total) = gpu_usage_dict[self.options.gpu_id]

        log(warning, "Overall gpu info: {}".format(gpu_usage_dict))

        # adjust mutation settings and uniqueness
        p_unique = 0.16 - 0.25 * used / total
        visit_modules(clone1, p_unique, [],
                      make_mutable_and_randomly_unique)

        visit_modules(clone2, p_unique, [],
                      make_mutable_and_randomly_unique)

        if np.random.random() < 0.8 - used / total:
            mutate(clone1, p=0.5)
            mutate(clone2, p=0.5)

        if np.random.random() < 0.9:
            crossover(clone1, clone2)
            self.replace_individual(r2, clone1)
        else:
            copyover(clone1, clone2)

        self.replace_individual(r1, clone2)
