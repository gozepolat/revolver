from revolver.meta.blueprint import visit_modules, \
    collect_keys, collect_modules, make_module, model_diagnostics
from revolver.models.blueprinted.ensemble import ScopedEnsembleMean
from revolver.models.blueprinted.resnet import ScopedResNet
from revolver.models.blueprinted.densenet import ScopedDenseNet
from revolver.models.blueprinted.convdeconv import ScopedConv2dDeconv2d
from revolver.modules.scoped_nn import ScopedConv2d
from revolver.modules.conv import Conv3d2d
from revolver.meta.scope import unregister
from revolver.models.blueprinted.bottleneckblock import ScopedBottleneckBlock
from revolver.models.blueprinted.resblock import ScopedResBlock
from revolver.models.blueprinted.resgroup import ScopedResGroup
from revolver.models.blueprinted.denseconcatgroup import ScopedDenseConcatGroup
from revolver.models.blueprinted.densesumgroup import ScopedDenseSumGroup
from torch.nn import Conv2d, Linear
from revolver.meta.heuristics.operators import mutate, \
    crossover, copyover
from revolver.meta.heuristics import extensions
from revolver.utils.domain import ClosedList
from revolver.utils.engine import get_num_parameters
from revolver.utils import common
from revolver.utils.transformer import softmax
from revolver.utils.usage_helpers import make_net_blueprint
from revolver.models.blueprinted.separable import ScopedDepthwiseSeparable
from revolver.meta.blueprint import Blueprint
from logging import warning, exception
import numpy as np
import copy
import math
import inspect
import torch


def log(log_func, msg):
    if common.DEBUG_POPULATION:
        log_func("revolver.meta.heuristics.population: %s" % msg)


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
        if not inspect.isclass(_type):
            continue
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

    if total_weights > 16000000:
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

    training_budget = max((common.POPULATION_AVERAGE_SCORE / genotype['meta']['score'])**3, 0.2)
    n_batches_per_phenotype = common.POPULATION_NUM_BATCH_PER_INDIVIDUAL * training_budget

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
        genotype['meta']['_test_loss'] = engine.state['score']
        genotype['meta']['_test_acc'] = engine.state['test_acc']

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


def get_mean_score(bp, new_score, weight, id_set=None):
    if id_set is None:
        id_set = set()

    self_id = id(bp)
    if self_id in id_set:
        return bp['meta']['mean_score']

    id_set.add(self_id)
    n = 0.
    total = 0.
    if 'children' in bp:
        for child in bp['children']:
            total += update_fitness(child, new_score, weight, id_set)
            n += 1

    for k, v in bp.items():
        if isinstance(v, Blueprint) and k != 'parent':
            total += update_fitness(v, new_score, weight, id_set)
            n += 1

    if n == 0:
        return None

    return total / n


def update_fitness(bp, new_score, weight, id_set=None):
    if 'meta' not in bp:
        bp['meta'] = {}

    mean_score = get_mean_score(bp, new_score, weight, id_set)
    bp['meta']['mean_score'] = mean_score

    module_score = common.POPULATION_COMPONENT_SCORES_DICT.get(bp['name'], 0)

    if 'score' not in bp['meta']:
        bp['meta']['score'] = module_score if module_score != 0 else get_genotype_fitness(bp)

    score = bp['meta']['score']
    if mean_score is None:
        mean_score = score
    if score < np.inf:
        global_score = mean_score if module_score == 0 else module_score
        score = weight * (new_score * .7 + mean_score * .2 + global_score * .1) + score * (1.0 - weight)
    else:
        log(warning, "Score was infinity?!!")
        score = new_score

    if bp['meta']['mean_score'] is None:
        bp['meta']['mean_score'] = score
    bp['meta']['score'] = score

    if not bp['unique']:
        weight *= .1
        if module_score == 0:
            module_score = score
        common.POPULATION_COMPONENT_SCORES_DICT[bp['name']] = score * weight + module_score * (1 - weight)

    return score


def update_score(blueprint, new_score, weight=0.2):
    """Update the score of blueprint components, and get the avg"""

    log(warning, f"Previous score {blueprint['meta']['score']}")
    score = update_fitness(blueprint, new_score, weight)
    log(warning, f"New score {blueprint['meta']['score']} {score}, updated with {new_score}")

    return score


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
    conv_list = [ScopedConv2d, ScopedConv2dDeconv2d, ScopedEnsembleMean, ScopedBottleneckBlock,
                 ScopedResBlock, ScopedDepthwiseSeparable]
    conv_list.extend([ScopedConv2d] * 10)

    if conv_extend:
        conv_list.extend(conv_extend)

    conv_module = ClosedList(conv_list)
    residual = ClosedList([True, False])
    skeleton_list = [(8, 8, 8), (8, 8, 16), (8, 16, 32), (8, 16, 32, 32), (8, 32, 64),
                     (16, 16, 32), (16, 8, 32), (16, 32, 64), (16, 32, 64, 64)]
    if skeleton_extend:
        skeleton_list.extend(skeleton_extend)

    skeleton_list = list(set(skeleton_list))

    skeleton = ClosedList(skeleton_list)
    block_module = ClosedList([ScopedBottleneckBlock, ScopedResBlock])
    group_module = ClosedList([ScopedDenseConcatGroup, ScopedDenseSumGroup, ScopedResGroup])
    drop_p = ClosedList([0, .1, .1, .25, .25, .25, .35, .35, .5])
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
        visit_modules(blueprint, (options.p_initialize_with_unique, False), [],
                      make_mutable_and_randomly_unique)
        blueprint.make_unique(refresh_unique_suffixes=False)
        blueprints.append(blueprint)

    return blueprints


class Population(object):
    def __init__(self, options):
        assert options.population_size > 7, "Minimum population size must be 8"
        self.options = options
        self.genotypes = []
        self.ids = []
        self.population_size = 0
        self.iteration = 1
        self.populate_with_random()
        if hasattr(self.options, 'max_skeleton_width'):
            w = self.options.max_skeleton_width
        else:
            w = 64
        if hasattr(self.options, 'max_skeleton_depth'):
            d = self.options.max_skeleton_depth
        else:
            d = 6

        self.skeletons = self.make_skeletons(w, d)

    @staticmethod
    def make_skeletons(width, depth):
        skeletons = []
        for low_w in range(8, width, 8):
            for d in range(3, depth):
                skeletons.append(tuple([low_w * min(pow(2, d_i), 4) for d_i in range(d)]))
        return skeletons

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
        for score, blueprint in sorted_genotypes[:self.options.population_size]:
            blueprint['meta']['score'] = score
            self.add_individual(blueprint)

    def random_pick(self, sample_size=0):
        return np.random.choice(len(self.genotypes), sample_size, replace=False)

    def pick_indices(self, sample_size=0, exclude_set=None):
        """Randomly pick genotype indices, sometimes favor lower scores"""
        if sample_size == 0:
            sample_size = self.options.sample_size

        if np.random.random() < common.POPULATION_RANDOM_PICK_P:
            return self.random_pick_n_others(exclude_set=exclude_set, n=sample_size)

        if exclude_set is None:
            exclude_set = set()

        # by the end of population training, focus on training the top individuals only
        if common.POPULATION_FOCUS_PICK_RATIO < 1.:
            indices = self.get_sorted_indices()
            cut_index = int(common.POPULATION_FOCUS_PICK_RATIO * self.population_size)
            cut_index = max(sample_size + 2, cut_index)

            for i in range(self.population_size - 1, cut_index, -1):
                exclude_set.add(indices[i])

        scores = [bp['meta']['score'] for i, bp in enumerate(self.genotypes) if i not in exclude_set]
        indices = [i for i, _ in enumerate(self.genotypes) if i not in exclude_set]
        distribution = np.array(scores)

        # lower score (i.e. fitness) is better and should be selected more often
        transformed = np.max(distribution) - distribution

        distribution = transformed / np.sum(transformed)

        try:
            choice = np.random.choice(indices, sample_size, p=distribution, replace=False)
            return choice

        except ValueError:
            exception("Population.pick_indices: Caught exception with weighted choice")
            return np.random.choice(indices, sample_size, replace=False)

    def handle_bad_individual(self, i):
        bp = self.genotypes[i]
        log(warning, "This individual caused exception: %s" % bp['name'])
        log(warning, f"Removing individual{bp['name']}")
        bp.dump_pickle(f"../errored_{bp['name']}.pkl")

        # time to kill it
        score = bp['meta']['score']
        while clone is None:
            score *= 1.2
            clone = self.maybe_pick_from_randomly_generated(score=score)
            if clone is not None:
                self.replace_individual(i, clone)

    def update_scores(self, calculate_phenotype_fitness=True, additional_indices=None):
        """Evaluate and improve a portion of the population, according to the scores"""
        weight = self.options.update_score_weight
        indices = self.pick_indices()
        if additional_indices is not None:
            np.append(indices, additional_indices)
        for i in indices:
            bp = self.genotypes[i]

            if 'score' not in bp['meta']:
                bp['meta']['score'] = get_genotype_fitness(bp)

            if not calculate_phenotype_fitness:
                continue

            # potentially train and test the model for bp
            # remove it if it causes exception
            try:
                new_score = self.options.utility(bp, self.options)
                update_score(bp, new_score, weight=weight)
                continue
            except (RuntimeError, ValueError, RecursionError):
                exception("Population.update_scores: Caught exception when scoring the model")
                # model_diagnostics(bp)
                self.handle_bad_individual(i)

            # second try in case training failed
            bp = self.genotypes[i]
            try:
                new_score = self.options.utility(bp, self.options)
                update_score(bp, new_score, weight=weight)
            except (RuntimeError, ValueError, RecursionError):
                exception("Population.update_scores: Caught exception again when scoring new model")
                # model_diagnostics(bp)
                self.handle_bad_individual(i)

    def get_average_score(self):
        """Get average score for the population"""
        return np.mean(self.get_all_scores())

    def get_the_best_index(self):
        """Get the best scoring individual index in the population"""
        return np.argmin(self.get_all_scores())

    def get_sorted_indices(self):
        """Get the indices of individuals sorted from the best to worst"""
        return np.argsort(self.get_all_scores())

    def get_all_sorted_scores_dict(self):
        sorted_indices = self.get_sorted_indices()
        return {self.genotypes[i]['name']: self.genotypes[i]['meta']['score'] for i in sorted_indices}

    def get_all_scores(self):
        return [bp['meta']['score'] for bp in self.genotypes]

    def sort_based_on_score(self, indices):
        """Sort based on the score s.t. first individual is the best

        e.g. for indices = (2,    6,   3)
                 scores  = (12.4, 0.5, 4.4)
        sort_based_on_score((2,6,3)) would return 6, 3, 2"""
        scores = [(self.genotypes[index]['meta']['score'], index) for index in indices]
        return [index for score, index in sorted(scores)]

    def random_pick_n_others(self, exclude_set, n):
        if exclude_set is None:
            exclude_set = set()
        selected_indices = [i for i, _ in enumerate(self.genotypes)
                            if i not in exclude_set]
        return np.random.choice(selected_indices, n, replace=False)

    def maybe_pick_from_randomly_generated(self, score):
        for i in range(common.POPULATION_RANDOM_SEARCH_ITERATIONS):
            bp = generate_net_blueprints(self.options, num_individuals=1,
                                         skeleton_extend=self.skeletons)[0]
            bp_score = bp['meta']['score'] = get_genotype_fitness(bp)
            if bp_score < score:
                log(warning, f"Search successful. "
                             f" Genotype cost coefficient was {common.POPULATION_GENOTYPE_COST_COEFFICIENT}")
                return bp

        return None

    def maybe_pick_from_genetic_algorithm(self, score, exclude_set=None):
        if np.random.random() < common.POPULATION_IMMIGRATION_P:
            bp = self.maybe_pick_from_randomly_generated(score)
            if bp is not None:
                return bp

        # favor better models sometimes
        index1, index2 = self.pick_indices(2, exclude_set=exclude_set)

        clone1 = copy.deepcopy(self.genotypes[index1])
        clone2 = copy.deepcopy(self.genotypes[index2])

        p_unique = common.POPULATION_MUTATION_COEFFICIENT * common.UNIQUENESS_TOGGLE_P
        visit_modules(clone1, (p_unique, False), [],
                      make_mutable_and_randomly_unique)

        visit_modules(clone2, (p_unique, False), [],
                      make_mutable_and_randomly_unique)

        if np.random.random() < common.POPULATION_MUTATION_COEFFICIENT:
            for i in range(50):
                mutated = mutate(clone1, p=0.5)
                if mutated:
                    log(warning, f"Mutated {mutated} after {i} tries")
                    break
            for i in range(50):
                mutated = mutate(clone2, p=0.5)
                if mutated:
                    log(warning, f"Mutated {mutated} after {i} tries")
                    break

        if np.random.random() < common.POPULATION_CROSSOVER_COEFFICIENT:
            op = crossover
        else:
            op = copyover

        op_type = "Cloned"
        for i in range(100):
            successful = crossover(clone1, clone2)
            if successful:
                op_type = f"{op} {successful}"
                break

        if op_type == "Cloned" and np.random.random() > .5:
            bp = self.maybe_pick_from_randomly_generated(score)
            if bp is not None:
                op_type = "Random"
                clone1 = bp

        log(warning, f"{op_type} after {i} tries")
        clone1['meta']['score'] = get_genotype_fitness(clone1)
        clone2['meta']['score'] = get_genotype_fitness(clone2)

        if clone1['meta']['score'] > clone2['meta']['score'] and np.random.random() > .5:
            clone1, clone2 = clone2, clone1

        # do not trust genotype fitness too much
        if clone1['meta']['score'] < score or np.random.random() > .5:
            return clone1

        if clone2['meta']['score'] < score or np.random.random() > .25:
            return clone2

        return None

    def add_next_gen(self, search_mode="random", adjust_coefficient=True,
                     exclude_set=None, options=None, num_offsprings=2):
        """Search step for replacing a number of individuals (<=2) with the next generation"""
        if options is not None:
            self.options = options

        if len(self.genotypes) == 0:
            self.populate_with_random()

        if exclude_set is None:
            exclude_set = set()

        assert self.population_size >= num_offsprings * 3 + len(exclude_set) > 0

        sorted_indices = self.get_sorted_indices()
        if np.random.random() < common.POPULATION_CLEANUP_P:
            # clean up population
            # worst indices will be replaced with new individuals
            old_gen_ix = [i for i in sorted_indices if i not in exclude_set][-3 * num_offsprings:]
            old_gen_ix = np.random.choice(old_gen_ix, num_offsprings, replace=False)
        else:
            # competition
            # mid range indices will be pitted against each other
            old_gen_ix = self.random_pick_n_others(exclude_set=exclude_set | set(sorted_indices[:2 * num_offsprings]),
                                                   n=num_offsprings)

        replaced_indices = []
        for index in old_gen_ix:
            score = self.genotypes[index]['meta']['score']
            if search_mode == "random":
                bp = self.maybe_pick_from_randomly_generated(score)
            else:
                bp = self.maybe_pick_from_genetic_algorithm(score, exclude_set=set(old_gen_ix))

            if bp is not None:
                self.replace_individual(index, bp)
                replaced_indices.append(index)
                if adjust_coefficient:
                    common.POPULATION_GENOTYPE_COST_COEFFICIENT *= 1.005
            else:
                log(warning, f"Could not find a genotype with score < {score}")
                if not adjust_coefficient:
                    continue
                common.POPULATION_GENOTYPE_COST_COEFFICIENT *= .985
                common.POPULATION_GENOTYPE_COST_COEFFICIENT = max(common.POPULATION_MIN_GENOTYPE_COST_COEFFICIENT,
                                                                  common.POPULATION_GENOTYPE_COST_COEFFICIENT)

        return replaced_indices
