# -*- coding: utf-8 -*-
import math

from revolver.meta.scope import unregister
from revolver.models.blueprinted.convdeconv import ScopedConv2dDeconv2d
from revolver.models.blueprinted.optimizer import ScopedEpochEngine
from revolver.models.blueprinted.resblock import ScopedResBlock
from revolver.models.blueprinted.resgroup import ScopedResGroup
from revolver.models.blueprinted.resnet import ScopedResNet
from revolver.models.blueprinted.densenet import ScopedDenseNet
from revolver.models.blueprinted.denseconcatgroup import ScopedDenseConcatGroup
from revolver.models.blueprinted.bottleneckblock import ScopedBottleneckBlock
from revolver.modules.scoped_nn import ScopedConv2d, ScopedBatchNorm2d, \
    ScopedFeatureSimilarityLoss, ScopedCrossEntropyLoss
from revolver.modules.loss import collect_features
from revolver.meta.blueprint import make_module, visit_modules, collect_modules
from revolver.utils import common
import json
import os

from revolver.utils.transformer import all_to_none
from revolver.utils.visualize import plot_model
from logging import warning, info
import glob
import pandas as pd
import re
import inspect


def log(log_func, msg):
    if common.DEBUG_POPULATION:
        log_func("revolver.utils.usage_helpers: %s" % msg)


def make_net_blueprint(options, suffix=''):
    prefix = re.findall("[a-zA-Z0-9_]+", str(options.net).split('.')[-1])[0]
    net = options.net.describe_default(prefix=prefix, suffix=suffix,
                                       num_classes=options.num_classes,
                                       depth=options.depth,
                                       width=options.width,
                                       block_depth=options.block_depth,
                                       drop_p=options.drop_p,
                                       conv_module=options.conv_module,
                                       dropout_p=options.dropout_p,
                                       callback=options.callback,
                                       group_module=options.group_module,
                                       residual=options.residual,
                                       skeleton=options.skeleton,
                                       group_depths=options.group_depths,
                                       block_module=options.block_module,
                                       dense_unit_module=options.dense_unit_module,
                                       input_shape=options.input_shape,
                                       fractal_depth=options.fractal_depth,
                                       head_kernel=options.head_kernel,
                                       head_stride=options.head_stride,
                                       head_padding=options.head_padding,
                                       head_pool_kernel=options.head_pool_kernel,
                                       head_pool_stride=options.head_pool_stride,
                                       head_pool_padding=options.head_pool_padding,
                                       head_modules=options.head_modules)
    return net


def create_single_engine(net_blueprint, options):
    if not hasattr(options, 'engine_pkl') or options.engine_pkl is None or options.mode == "test":
        engine_blueprint = ScopedEpochEngine.describe_default(prefix='EpochEngine',
                                                              net_blueprint=net_blueprint,
                                                              max_epoch=options.epochs,
                                                              batch_size=options.batch_size,
                                                              learning_rate=options.lr,
                                                              lr_decay_ratio=options.lr_decay_ratio,
                                                              lr_drop_epochs=options.lr_drop_epochs,
                                                              criterion=options.criterion,
                                                              callback=options.callback,
                                                              dataset=options.dataset,
                                                              num_thread=options.num_thread,
                                                              use_tqdm=options.use_tqdm,
                                                              crop_size=options.crop_size,
                                                              weight_decay=options.weight_decay,
                                                              test_mode=options.mode == "test",
                                                              validation_ratio=options.validation_ratio)
    else:
        log(warning, f'Loading the engine blueprint from {options.engine_pkl} and disregarding all the other options')
        engine_blueprint = pd.read_pickle(options.engine_pkl)

    single_engine = make_module(engine_blueprint)

    if len(options.load_path) > 0:
        log(warning, "Loading the engine state dictionary from: %s" % options.load_path)
        if not os.path.isfile(options.load_path):
            log(warning, f'Loading failed, no such file: {options.load_path}')
            return single_engine
        single_engine.load_state_dict(options.load_path)

    return single_engine


def make_checkpoint_path(path, epoch):
    no_extension = path[:-7]
    ix = no_extension.rfind('epoch')
    return f'{no_extension[:ix]}epoch_{epoch}.pth.tar'


def remove_older_checkpoints(path, epoch, keep_last_n, oldest_kept):
    while oldest_kept < epoch - keep_last_n + 1:
        ckpt_path = make_checkpoint_path(path, oldest_kept)
        if os.path.exists(ckpt_path):
            os.remove(ckpt_path)
        oldest_kept += 1
    return oldest_kept


def train_with_single_engine(model, options):
    print("=================", options.mode)
    if not os.path.isdir("ckpt"):
        os.mkdir("ckpt")

    if options.mode == 'test':
        options.lr = 0.000000001

    def get_epoch_number(path):
        return int(path.split('.')[-3].split('_')[-1])

    if options.load_latest_checkpoint:
        ckpt_regex = make_checkpoint_path(options.load_path, '*')

        ckpt_paths = glob.glob(ckpt_regex)
        latest_paths = sorted(ckpt_paths,
                              key=get_epoch_number,
                              reverse=True)
        if len(latest_paths) > 0:
            latest_path = latest_paths[0]
            log(info, f'Setting load_path to {latest_path} and loading from the most recent checkpoint')
            options.load_path = latest_path

    engine = create_single_engine(model, options)

    net_type = re.findall("[a-zA-Z]+", str(type(engine.net)))[0]
    name = f"e{engine.blueprint['name']}m{net_type}d{options.depth}" \
           f"w{options.width}bs{options.batch_size}decay{options.weight_decay}" \
           f"lr{options.lr}ds{options.dataset}"

    filename = os.path.join(options.save_folder, name)

    log(warning, "Network architecture:")
    log(warning, "=====================")
    log(warning, engine.net)
    log(warning, "=====================")

    if options.mode == 'test':
        log(warning, "Test mode is activated")
        engine.hook('on_end', engine.state)
        return

    if 'epoch' in engine.state:
        log(info, f"{engine.state['epoch']}")
    test_every_nth = options.test_every_nth
    keep_last_n = options.keep_last_n
    oldest_kept = 0
    engine.state['maxepoch'] = options.epochs

    if test_every_nth > 0:
        for j in range(engine.state['epoch'], options.epochs, 1):
            engine.start_epoch()
            engine.train_n_batches(options.num_samples)
            if j % test_every_nth == test_every_nth - 1:
                engine.end_epoch()
                engine.hook('on_end', engine.state)
                # ckpt_name = make_checkpoint_path(os.path.join("ckpt", name), j)
                # engine.dump_state(ckpt_name)
                # oldest_kept = remove_older_checkpoints(name, j, keep_last_n, oldest_kept)
            else:
                engine.state['epoch'] += 1
    else:
        for j in range(engine.state['epoch'], options.epochs, 1):
            engine.train_one_epoch()

    # dump the state for allowing more training later
    engine.dump_state(filename)

    # dump the conv weights to a png file for visualization
    if len(options.save_png_folder) > 0:
        filename = os.path.join(options.save_png_folder, name)
        plot_model(engine.state['network'].net.cpu(), filename)


def get_default_resnet(options):
    prefix = str(ScopedResNet).split('.')[-1]
    net = ScopedResNet.describe_default(prefix=prefix, num_classes=options.num_classes,
                                        depth=options.depth, width=options.width,
                                        block_depth=options.block_depth,
                                        head_modules=options.head_modules,
                                        skeleton=(12, 24, 48), group_depths=options.group_depths,
                                        input_shape=options.input_shape)
    adjust_uniqueness(net, options)
    return net


def get_default_densenet(options):
    prefix = str(ScopedDenseNet).split('.')[-1]
    net = ScopedDenseNet.describe_default(prefix=prefix, num_classes=options.num_classes,
                                          depth=options.depth, width=options.width,
                                          block_depth=options.block_depth,
                                          group_module=ScopedDenseConcatGroup, residual=False,
                                          skeleton=(1, 1, 1), group_depths=options.group_depths,
                                          block_module=ScopedBottleneckBlock,
                                          dense_unit_module=ScopedBottleneckBlock,
                                          input_shape=options.input_shape,
                                          head_kernel=3, head_stride=1, head_padding=1,
                                          head_modules=('conv', 'bn'))
    adjust_uniqueness(net, options)
    return net


def adjust_uniqueness(net, options):
    def make_unique(bp, _, __):
        if 'all' in options.unique:
            bp.make_unique(refresh_unique_suffixes=False)
            return

        if 'bn' in options.unique:
            if inspect.isclass(bp['type']) and issubclass(bp['type'], ScopedBatchNorm2d):
                bp.make_unique(refresh_unique_suffixes=False)
                bp['kwargs']['momentum'] = 0.1

        if 'convdim' in options.unique:
            if inspect.isclass(bp['type']) and issubclass(bp['type'], ScopedConv2d):
                if 'kernel_size' in bp['kwargs'] and bp['kwargs']['kernel_size'] == 1:
                    bp.make_unique(refresh_unique_suffixes=False)

        if 'conv' in options.unique:
            if inspect.isclass(bp['type']) and issubclass(bp['type'], ScopedConv2d):
                if 'kernel_size' in bp['kwargs'] and bp['kwargs']['kernel_size'] == 3:
                    bp.make_unique(refresh_unique_suffixes=False)

    visit_modules(net, None, None, make_unique)


def add_seed_individuals(population, options, resnet_shape, densenet_shape):
    default_unique = options.unique
    default_width = options.width
    default_depth = options.depth

    index = population.get_the_best_index()
    score = population.genotypes[index]['meta']['score'] * common.POPULATION_COST_SEED_INDIVIDUAL_SCALE

    # seed ResNet
    def add_resnet():
        net = get_default_resnet(options)
        net['meta']['score'] = score
        population.add_individual(net)

    options.height = resnet_shape[0]
    options.width = resnet_shape[1]
    options.unique = ('all',)
    add_resnet()
    options.unique = ('bn', 'convdim')
    add_resnet()

    # seed DenseNet,
    def add_densenet():
        network = get_default_densenet(options)
        network['meta']['score'] = score
        population.add_individual(network)

    options.height = densenet_shape[0]
    options.width = densenet_shape[1]
    options.unique = ('all',)
    add_densenet()
    options.unique = ('bn', 'convdim')
    add_densenet()

    options.unique = default_unique
    options.depth = default_depth
    options.width = default_width


def increase_exploration(top_score_stuck_ctr):
    if top_score_stuck_ctr < 15:
        return increase_exploitation()

    # getting stuck, make exploration less costly
    if common.POPULATION_GENOTYPE_COST_COEFFICIENT >= 1. + common.POPULATION_MIN_GENOTYPE_COST_COEFFICIENT:
        common.POPULATION_GENOTYPE_COST_COEFFICIENT -= 1
    if common.POPULATION_CROSSOVER_COEFFICIENT < .95:
        common.POPULATION_CROSSOVER_COEFFICIENT += .03
    if common.POPULATION_MUTATION_COEFFICIENT < .95:
        common.POPULATION_MUTATION_COEFFICIENT += .03


def increase_exploitation():
    """Gradually make exploration more costly"""
    common.POPULATION_GENOTYPE_COST_COEFFICIENT *= 1.04
    if common.POPULATION_CROSSOVER_COEFFICIENT > .35:
        common.POPULATION_CROSSOVER_COEFFICIENT -= .001
    if common.POPULATION_MUTATION_COEFFICIENT > .35:
        common.POPULATION_MUTATION_COEFFICIENT -= .001


def train_population(population, options, default_resnet_shape, default_densenet_shape):
    if options.add_seed not in common.NO_SET:
        add_seed_individuals(population, options, default_resnet_shape, default_densenet_shape)

    net_blueprint = None
    # for options such as: evolve_warmup_random, evolve_warmup, random_warmup, random_warmup_evolve
    if "warmup" in options.search_mode:
        search_mode = options.search_mode.split("_warmup")[0]
        log(warning, f"Warming up the population with improved genotypes. Search mode {search_mode}")
        for i in range(options.warmup_epoch):
            population.add_next_gen(search_mode=search_mode, adjust_coefficient=False)
            bp = population.genotypes[population.get_the_best_index()]
            log(warning, f"{i}: top {bp['meta']['score']} "
                         f"mean {population.get_average_score()} "
                         f"size {len(population.genotypes)} "
                         f"name {bp['name']}")

    if "warmup_single_train" in options.search_mode:
        indices = population.get_sorted_indices()
        new_options = options
        set_default_options_for_single_network(new_options)
        for i, index in enumerate(indices):
            net_blueprint = population.genotypes[index]
            net_blueprint.dump_pickle(f"../top_{i}_single_train.pkl")
            try:
                print(f"============ Training top {i}th blueprint ============")
                train_single_network(new_options, net_blueprint)
                break
            except (RuntimeError, ValueError, RecursionError):
                print("Caught exception when trying to train the {i}th top genotype. Skipping...")
                for bp in collect_modules(net_blueprint):
                    if bp['unique']:
                        unregister(bp['name'])

                unregister(net_blueprint['name'])

        return net_blueprint

    population.update_scores()
    search_mode = "random" if options.search_mode in {"random", "random_warmup", "evolve_warmup_random"} else "evolve"
    prev_top_index = -1

    # drop lr every iteration, so it goes from options.lr to options.min_lr
    exp = (options.max_iteration - options.gradual_lr_drop)
    lr_drop = 2**((math.log2(options.min_lr) - math.log2(options.lr)) / exp)

    for i in range(options.max_iteration):
        log(warning, 'Population generation: %d' % i)
        gpu_usage_dict = common.get_gpu_memory_info()
        log(warning, "Overall gpu info: {}".format(gpu_usage_dict))
        (used, total) = gpu_usage_dict[options.gpu_id]

        common.POPULATION_RANDOM_PICK_P = (total - used) / total * .5 + .1

        if i > options.max_iteration * .8:
            common.POPULATION_FOCUS_PICK_RATIO = 1. - float(i) / options.max_iteration
            common.POPULATION_RANDOM_PICK_P = .0
            # Train top individuals more for the final scores
            common.POPULATION_NUM_BATCH_PER_INDIVIDUAL = 512

        # attempt search 3 times
        indices = set()
        for _ in range(3):
            new_indices = population.add_next_gen(search_mode=search_mode, exclude_set=indices)
            indices = indices | set(new_indices)
            # new individuals to be added
            if len(indices) > 1:
                break

        population.update_scores(additional_indices=list(indices))
        if i > options.gradual_lr_drop:
            options.lr *= lr_drop

        index = population.get_the_best_index()

        net_blueprint = population.genotypes[index]
        best_score = net_blueprint['meta']['score']
        common.POPULATION_AVERAGE_SCORE = average_score = population.get_average_score()
        log(warning, "{} Current top score: {}, id: {}, name {}".format(i, best_score, id(net_blueprint),
                                                                        net_blueprint['name']))
        log(warning, f"Current mean score {average_score}, size {len(population.genotypes)}")

        if prev_top_index != index:
            log(warning, f"New top individual detected. Updated ranking: \n{population.get_all_sorted_scores_dict()}")
            test_acc = net_blueprint['meta'].get('_test_acc', 'unk')
            net_blueprint.dump_pickle(f"../top_{i}_{net_blueprint['name']}_test_acc_{test_acc}"
                                      f"{options.genotype_cost}_{options.search_mode}_{options.dataset}.pkl")
            prev_top_index = index
            with open(f"../state_{i}_population_component_scores_dict_test_acc_"
                      f"{test_acc}_{options.genotype_cost}_{options.search_mode}_{options.dataset}.json", "w") as f:
                json.dump(common.POPULATION_COMPONENT_SCORES_DICT, f)

    return net_blueprint


def train_single_network(options, net=None):
    if net is None:
        net = make_net_blueprint(options)

    adjust_uniqueness(net, options)

    trainer = train_with_double_engine

    if options.single_engine:
        trainer = train_with_single_engine

    trainer(net, options)


def adjust_options(options):
    num_channels = 3
    width = height = 32
    group_depths = None
    skeleton = json.loads(options.skeleton)

    if options.dataset == 'ILSVRC2012':
        num_classes = 1000
        width = height = 224
        num_samples = 1200000

        # DenseNet 161 architecture
        group_depths = (6, 12, 35, 24)
        skeleton = [48, 48, 48, 48]

        # ResNet 152 architecture
        if options.net == ScopedResNet:
            group_depths = (3, 8, 36, 3)
            skeleton = [64, 128, 256, 512]
            options.width = 4
    elif options.dataset == 'tiny-imagenet-200':
        num_classes = 200
        width = height = 56
        num_samples = 100000
        group_depths = (1, 4, 8, 1)
        skeleton = [8, 24, 48, 64]
        options.block_depth = 3
    elif options.dataset == 'CIFAR100':
        num_classes = 100
        num_samples = 50000
    else:  # CIFAR10 or MNIST
        num_classes = 10
        num_samples = 50000
        if options.dataset == 'MNIST':
            num_channels = 1
            width = height = 28
            num_samples = 60000
        elif options.dataset == 'SVHN':
            num_samples = 73257

    options.lr_drop_epochs = json.loads(options.lr_drop_epochs)
    options.group_depths = group_depths
    options.crop_size = max(width, height)
    options.skeleton = skeleton
    options.num_classes = num_classes
    options.num_samples = num_samples
    options.input_shape = (options.batch_size, num_channels, width, height)


# below is an older usage
##############################################################
# training with an engine pair having different learning rates


def create_engine_pair(net_blueprint, options, epochs, crop):
    """Engines to train different portions of the given model"""

    def common_picker(model):
        for k, v in model.named_parameters():
            if 'generator' not in k:
                yield v

    def generator_picker(model):
        for k, v in model.named_parameters():
            if 'generator' in k:
                yield v

    common_engine_blueprint = ScopedEpochEngine.describe_default(prefix='CommonEpochEngine',
                                                                 net_blueprint=net_blueprint,
                                                                 max_epoch=options.epochs,
                                                                 batch_size=options.batch_size,
                                                                 learning_rate=options.lr,
                                                                 lr_decay_ratio=options.lr_decay_ratio,
                                                                 lr_drop_epochs=options.lr_drop_epochs,
                                                                 crop_size=crop,
                                                                 dataset=options.dataset,
                                                                 num_thread=options.num_thread,
                                                                 criterion=ScopedFeatureSimilarityLoss,
                                                                 callback=collect_features,
                                                                 optimizer_parameter_picker=common_picker,
                                                                 weight_decay=options.weight_decay)

    # accesses the same resnet model instance
    generator_engine_blueprint = ScopedEpochEngine.describe_default(prefix='GeneratorEpochEngine',
                                                                    net_blueprint=net_blueprint,
                                                                    max_epoch=options.epochs,
                                                                    batch_size=options.batch_size,
                                                                    learning_rate=options.lr * 0.2,
                                                                    lr_decay_ratio=options.lr_decay_ratio,
                                                                    lr_drop_epochs=options.lr_drop_epochs,
                                                                    crop_size=crop,
                                                                    dataset=options.dataset,
                                                                    num_thread=options.num_thread,
                                                                    optimizer_parameter_picker=generator_picker,
                                                                    weight_decay=options.weight_decay)
    c = make_module(common_engine_blueprint)
    g = make_module(generator_engine_blueprint)
    return c, g


def train_with_double_engine(model, options, epochs, crop, n_samples=50000):
    common_engine, generator_engine = create_engine_pair(model, options,
                                                         epochs, crop)

    log(warning, "Network architecture:")
    log(warning, "=====================")
    log(warning, common_engine.net)
    log(warning, "=====================")

    batch = options.batch_size * 17
    repeat = n_samples // batch + 1
    for j in range(common_engine.state['epoch'], options.epochs, 1):
        common_engine.start_epoch()
        generator_engine.start_epoch()

        # train back and forth
        for i in range(repeat):
            common_engine.train_n_batches(batch)
            generator_engine.train_n_batches(batch)

        # test every fourth epoch
        if j % 4 == 3:
            common_engine.end_epoch()
        else:
            common_engine.state['epoch'] += 1
        generator_engine.state['epoch'] += 1

    common_engine.hook('on_end', common_engine.state)
    generator_engine.hook('on_end', generator_engine.state)


def set_default_options_for_single_network(options):
    """Default options for the single network training"""
    options.conv_module = ScopedConv2dDeconv2d
    options.dropout_p = 0.5
    options.drop_p = 0.5
    options.fractal_depth = 4
    options.net = ScopedResNet
    options.callback = all_to_none
    options.criterion = ScopedCrossEntropyLoss
    options.residual = True
    options.group_module = ScopedResGroup
    options.block_module = ScopedResBlock
    options.dense_unit_module = ScopedBottleneckBlock
    options.head_kernel = 3
    options.head_stride = 1
    options.head_padding = 1
    options.head_pool_kernel = 3
    options.head_pool_stride = 2
    options.head_pool_padding = 1
    options.head_modules = ('conv',)
    options.unique = ('bn', 'convdim')
    options.use_tqdm = True
    options.test_every_nth = 1
    options.keep_last_n = 3
    options.load_latest_checkpoint = True
    options.engine_pkl = None
    options.validation_ratio = 0.01
