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
        # print(f"Zero : {blueprint[]}")
        return 0.01
    return math.log2(total_params * cost / total_weights)


def get_share_ratio(blueprint):
    """Collect module names and get the ratio of the same names"""
    names = collect_keys(blueprint, 'name')
    all_size = len(names)
    unique_size = len(set(names))
    return float(unique_size) / all_size


def get_genotype_fitness(genotype, verbose=False, *_, **__):
    """Estimate a score only by looking at the genotype"""
    utility = np.clip(estimate_rough_contexts(genotype), 0, None)
    fitness = common.POPULATION_GENOTYPE_COST_COEFFICIENT / (utility + 1.)
    fitness *= common.POPULATION_COST_ESTIMATION_SCALE
    if verbose:
        print(f"genotype utility: {utility}, cost: {fitness}")
    return fitness


def get_phenotype_score(genotype, options):
    """Estimate score by training / testing the model"""
    engine_maker = options.engine_maker

    n_batches_per_phenotype = max(options.num_samples // (options.batch_size * options.sample_size), 200)
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
        engine.train_n_batches(n_batches_per_phenotype)
        if j == epoch - 1:
            engine.end_epoch()
        else:
            engine.state['epoch'] += 1

    score = engine.state['score']
    if score < common.POPULATION_TOP_VALIDATION_SCORE:
        common.POPULATION_TOP_VALIDATION_SCORE = score
        # Also show the test_acc
        engine.hook('on_end', engine.state)

    # favor lower number of parameters
    num_parameters = get_num_parameters(engine.state['network'].net)
    num_parameters = math.log(num_parameters, favor_params)
    print({k: v for k, v in engine.state.items() if k != 'sample'})
    score *= (1.0 + num_parameters * common.POPULATION_COST_NUM_PARAMETER_SCALE)
    print(f"phenotype score: {score}")

    # delete the engine each time to save a tiny amount of memory as it is a unique obj
    # this does not delete the data_loaders as they will be reused
    unregister(engine.blueprint['name'])
    if np.isnan(score):
        raise ValueError("Score is nan")

    return score


def update_score(blueprint, new_score, weight=0.2):
    """Update the score of blueprint components, and get the avg"""
    modules = collect_modules(blueprint)
    average_score = 0.0

    for bp in modules:
        if 'score' not in bp['meta']:
            bp['meta']['score'] = get_genotype_fitness(bp)
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
        bp['meta']['score'] = get_genotype_fitness(bp)

    print(f"Previous score {bp['meta']['score']}")
    bp['meta']['score'] = new_score * weight + bp['meta']['score'] * (1.0 - weight)
    print(f"New score {bp['meta']['score']}, updated with {new_score}")

    return average_score


def make_mutable_and_randomly_unique(bp, inp, *_, **__):
    p_unique, refresh_unique_suffixes = inp
    extensions.extend_conv_mutables(bp, ensemble_size=3,
                                    block_depth=2)
    extensions.extend_depth_mutables(bp, min_depth=2)
    extensions.extend_mutation_mutables(bp)
    extensions.extend_bn_mutables(bp, min_momentum=0.01, max_momentum=0.99)
    extensions.extend_conv_kernel_size_mutables(bp, min_kernel_size=1, max_kernel_size=6)

    if np.random.random() < p_unique:
        bp.make_unique(refresh_unique_suffixes=refresh_unique_suffixes)


def generate_net_blueprints(options, num_individuals=None, conv_extend=None, skeleton_extend=None):
    """Randomly generate genotypes"""
    max_width = options.width
    max_depth = options.depth
    population_size = options.population_size

    depths = ClosedList(list(range(22, max_depth + 1, 6)))
    widths = ClosedList(list(range(1, max_width + 1)))
    conv_list = [ScopedDepthwiseSeparable, ScopedConv2d, ScopedConv2dDeconv2d]
    if conv_extend:
        conv_list.extend(conv_extend)

    conv_module = ClosedList(conv_list)
    residual = ClosedList([True, False])
    skeleton_list = [(3, 6, 12), (6, 6, 6), (6, 9, 12)]
    if skeleton_extend:
        skeleton_list.extend(skeleton_extend)

    skeleton = ClosedList(skeleton_list)
    block_module = ClosedList([ScopedBottleneckBlock, ScopedResBlock, ScopedResBottleneckBlock])
    group_module = ClosedList([ScopedDenseConcatGroup, ScopedDenseSumGroup, ScopedResGroup])
    drop_p = ClosedList([0, 0.1, 0.25, 0.5])
    block_depth = ClosedList([1, 2])
    nets = ClosedList([ScopedResNet, ScopedDenseNet])

    blueprints = []
    if num_individuals is None:
        num_individuals = population_size

    for i in range(num_individuals):
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
        visit_modules(blueprint, (0.01, False), [],
                      make_mutable_and_randomly_unique)
        blueprint.make_unique(refresh_unique_suffixes=False)
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
        self.populate_with_random()
        if hasattr(self.options, 'max_skeleton_width'):
            w = self.options.max_skeleton_width
        else:
            w = 32
        if hasattr(self.options, 'max_skeleton_depth'):
            d = self.options.max_skeleton_depth
        else:
            d = 3

        self.skeletons = [[np.random.randint(low, low * 2) for _ in range(d)] for low in range(3, w)]

    def replace_individual(self, index, blueprint):
        """Replace the individual at the given index with a new one"""
        if 'score' not in blueprint['meta']:
            blueprint['meta']['score'] = get_genotype_fitness(blueprint)

        log(warning, f"Removing {self.genotypes[index]['name']} "
                     f"which had cost {self.genotypes[index]['meta']['score']} "
                     f"in favor of {blueprint['name']} "
                     f"with cost {blueprint['meta']['score']}.")

        # remove all the relevant unique items from the scope
        for bp in collect_modules(self.genotypes[index]):
            if bp['unique']:
                unregister(bp['name'])

        unregister(self.genotypes[index]['name'])

        self.genotypes[index] = blueprint
        self.ids[index] = id(blueprint)

    def add_individual(self, blueprint):
        """Add a single individual to the population"""
        if id(blueprint) in set(self.ids):
            log(warning, "Individual %s is already in population"
                % blueprint['name'])
            return

        if 'score' not in blueprint['meta']:
            blueprint['meta']['score'] = get_genotype_fitness(blueprint, verbose=True)

        self.genotypes.append(blueprint)
        self.ids.append(id(blueprint))
        self.population_size += 1

    def generate_sorted_with_scores(self, num_individuals):
        genotypes = self.options.generator(self.options,
                                           num_individuals=num_individuals)
        sorted_genotypes = sorted([(get_genotype_fitness(bp), bp) for bp in genotypes], key=lambda x: x[0])
        return sorted_genotypes

    def populate_with_random(self, warmup_x=0):
        """Randomly generate genotypes and then add them"""
        if warmup_x < 1:
            warmup_x = self.options.warmup_x

        sorted_genotypes = self.generate_sorted_with_scores(self.options.population_size * warmup_x)
        for score, blueprint in sorted_genotypes[0:self.options.population_size]:
            blueprint['meta']['score'] = score
            self.add_individual(blueprint)

    def random_pick(self, sample_size=0):
        return np.random.choice(range(len(self.genotypes)),
                                sample_size, replace=False)

    def pick_indices(self, sample_size=0):
        """Randomly pick genotype indices, sometimes favor lower scores"""
        if sample_size == 0:
            sample_size = self.options.sample_size

        p = 0.3
        if np.random.random() < p:
            return self.random_pick(sample_size)

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

    def update_scores(self, calculate_phenotype_score=True):
        """Evaluate and improve a portion of the population according to the scores"""
        weight = self.options.update_score_weight
        indices = self.pick_indices()
        for i in indices:
            bp = self.genotypes[i]

            if 'score' not in bp['meta']:
                bp['meta']['score'] = get_genotype_fitness(bp)

            if not calculate_phenotype_score:
                continue

            # potentially train and test the model for bp
            # remove it if it causes exception
            new_score = 1e15
            try:
                new_score = self.options.utility(bp, self.options)
            except (RuntimeError, ValueError):
                exception("Population.update_scores: Caught exception when scoring the model")
                log(warning, "This individual caused exception: %s" % bp)

            update_score(bp, new_score, weight=weight)

            if bp["meta"]["score"] > 1e8:
                log(warning, f"Removing individual{bp['name']}")
                bp.dump_pickle(f"errored_{bp['name']}.pkl")
                # time to kill it
                # mutate another individual and replace it
                new_index = np.random.choice([j for j in self.pick_indices(3) if j != i])
                clone = copy.deepcopy(self.genotypes[new_index])

                mutate_counter = 0
                for _ in range(20):
                    mutate_counter += mutate(clone, p=0.5)
                    if mutate_counter > 2:
                        break

                clone.refresh_name()
                self.replace_individual(i, clone)

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

    def random_pick_n_others(self, picked, n):
        selected_indices = [i for i in range(self.population_size)
                            if i not in picked]
        return np.random.choice(selected_indices, n, replace=False)

    def maybe_pick_better_random(self, score):
        selected_bp = None
        for i in range(common.POPULATION_RANDOM_SEARCH_ITERATIONS):
            bp = generate_net_blueprints(self.options, num_individuals=1,
                                         skeleton_extend=self.skeletons)[0]
            bp_score = bp['meta']['score'] = get_genotype_fitness(bp)
            if bp_score < score:
                log(warning, f"Search successful. "
                             f" Genotype cost coefficient was {common.POPULATION_GENOTYPE_COST_COEFFICIENT}")
                selected_bp = bp
                break
        return selected_bp

    def get_all_scores(self):
        return {bp['name']: bp['meta']['score'] for bp in self.genotypes}

    def random_search(self, options=None):
        if options is not None:
            self.options = options
        gpu_usage_dict = common.get_gpu_memory_info()
        log(warning, "Overall gpu info: {}".format(gpu_usage_dict))

        if np.random.random() < common.POPULATION_CLEANUP_P:
            # clean up population
            # bad scored indices will be replaced with new individuals
            r1, r2 = self.get_max_indices()
        else:
            # competition
            # random indices will be pitted against each other
            r1, r2 = self.random_pick_n_others([], 2)

        replace_count = 0
        for index in (r1, r2):
            score = self.genotypes[index]['meta']['score']
            bp = self.maybe_pick_better_random(score)

            if bp is not None:
                common.POPULATION_GENOTYPE_COST_COEFFICIENT *= 1.01
                self.replace_individual(index, bp)
                replace_count += 1
            else:
                log(warning, f"Could not find a random genotype with score < {score}")

        if replace_count != 0:
            return

        worst_ix, _ = self.sort_based_on_score([r1, r2])

        for i in range(2):
            score = score * 1.5
            bp = self.maybe_pick_better_random(score)
            if bp is not None:
                log(warning, f"Replacing as if the score was instead {score}")
                self.replace_individual(worst_ix, bp)
                return

        log(warning, f"Could not find a random genotype even when score < {score}")
        common.POPULATION_GENOTYPE_COST_COEFFICIENT *= .99
        common.POPULATION_GENOTYPE_COST_COEFFICIENT = max(common.POPULATION_MIN_GENOTYPE_COST_COEFFICIENT,
                                                          common.POPULATION_GENOTYPE_COST_COEFFICIENT)

    def sort_based_on_score(self, indices):
        """Sort reversed based on the score s.t. first individual is the worst

        e.g. for indices = (2,    6,   3)
                 scores  = (12.4, 0.5, 4.4)
        sort_based_on_score((2,6,3)) would return 2, 3, 6"""
        scores = [(self.genotypes[index]['meta']['score'], index) for index in indices]
        return [index for score, index in sorted(scores, reverse=True)]

    def evolve_generation(self, options=None):
        """A single step of evolution"""
        if options is not None:
            self.options = options

        if len(self.genotypes) == 0:
            self.populate_with_random()

        # favor weighted pick eventually
        index1, index2 = self.pick_indices(2)

        if np.random.random() < common.POPULATION_CLEANUP_P:
            # clean up population
            # bad scored indices will be replaced with new individuals
            r1, r2 = self.get_max_indices()
        else:
            # competition
            # random indices will be pitted against each other
            r1, r2 = self.random_pick_n_others((index1, index2), 2)

        if index1 in {r1, r2} or index2 in {r1, r2}:
            index1, index2 = self.random_pick_n_others((r1, r2), 2)

        # make sure that the discarded r1, r2 have the worst scores
        r1, r2, index1, index2 = self.sort_based_on_score((r1, r2, index1, index2))

        clones = []
        if np.random.random() < common.POPULATION_IMMIGRATION_P:
            for _ in range(2):
                bp = self.maybe_pick_better_random(self.genotypes[index1]['meta']['score'])
                if bp is None:
                    clones.append(copy.deepcopy(self.genotypes[index1]))
                else:
                    clones.append(bp)
            clone1, clone2 = clones
        else:
            clone1 = copy.deepcopy(self.genotypes[index1])
            clone2 = copy.deepcopy(self.genotypes[index2])

        # restrict gpu memory usage
        gpu_usage_dict = common.get_gpu_memory_info()
        (used, total) = gpu_usage_dict[self.options.gpu_id]

        log(warning, "Overall gpu info: {}".format(gpu_usage_dict))

        # adjust mutation settings and uniqueness
        p_unique = (common.POPULATION_MUTATION_COEFFICIENT - used / total) * 0.2
        visit_modules(clone1, (p_unique, False), [],
                      make_mutable_and_randomly_unique)

        visit_modules(clone2, (p_unique, False), [],
                      make_mutable_and_randomly_unique)

        if np.random.random() < common.POPULATION_MUTATION_COEFFICIENT - used / total:
            for i in range(10):
                mutated = mutate(clone1, p=0.5)
                if mutated:
                    break
            for i in range(10):
                mutated = mutate(clone2, p=0.5)
                if mutated:
                    break

        if np.random.random() < common.POPULATION_CROSSOVER_COEFFICIENT:
            for i in range(50):
                successful = crossover(clone1, clone2)
                if successful:
                    break
            clone1.refresh_name()
            log(warning, f"Crossed {successful}. "
                         f"Removing {self.genotypes[r2]['name']} which had cost {self.genotypes[r2]['meta']['score']} "
                         f"in favor of {clone1['name']} "
                         f"with cost {clone1['meta']['score']}.")

            self.replace_individual(r2, clone1)
        else:
            copyover(clone1, clone2)

        clone2.refresh_name()
        log(warning, f"Removing {self.genotypes[r1]['name']} which had cost {self.genotypes[r1]['meta']['score']} "
                     f"in favor of {clone2['name']} "
                     f"with cost {clone2['meta']['score']}.")
        self.replace_individual(r1, clone2)
