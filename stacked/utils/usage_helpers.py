# -*- coding: utf-8 -*-
from stacked.models.blueprinted.optimizer import ScopedEpochEngine
from stacked.models.blueprinted.resnet import ScopedResNet
from stacked.models.blueprinted.densenet import ScopedDenseNet
from stacked.models.blueprinted.denseconcatgroup import ScopedDenseConcatGroup
from stacked.models.blueprinted.densefractalgroup import ScopedDenseFractalGroup
from stacked.models.blueprinted.bottleneckblock import ScopedBottleneckBlock
from stacked.models.blueprinted.densesumgroup import ScopedDenseSumGroup
from stacked.models.blueprinted.meta import ScopedMetaMasked
from stacked.modules.scoped_nn import ScopedConv2d, ScopedBatchNorm2d, \
    ScopedFeatureSimilarityLoss, ScopedFeatureConvergenceLoss
from stacked.modules.loss import collect_features, collect_depthwise_features
from stacked.meta.blueprint import make_module, visit_modules, make_blueprint
from stacked.utils.transformer import all_to_none
from stacked.utils import common
import argparse
import json
import os
from stacked.utils.visualize import plot_model
from logging import warning, info
import glob
import pandas as pd


def log(log_func, msg):
    if common.DEBUG_POPULATION:
        log_func("stacked.utils.usage_helpers: %s" % msg)


def make_net_blueprint(options, suffix=''):
    prefix = str(options.net).split('.')[-1]
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
    if not hasattr(options, 'engine_pkl') or options.engine_pkl is None:
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
                                                              weight_decay=options.weight_decay)
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
    if options.mode == 'test':
        options.lr = 0.00001

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
    log(info, f"{engine.state['epoch']}")

    name = '{}_model_dw_{}_{}_bs_{}_decay_{}_lr_{}_{}.pth.tar'.format(
        engine.blueprint['name'],
        type(engine.net),
        options.depth,
        options.width,
        options.batch_size,
        options.weight_decay,
        options.lr,
        options.dataset, )

    filename = os.path.join(options.save_folder, name)

    log(warning, "Network architecture:")
    log(warning, "=====================")
    log(warning, engine.net)
    log(warning, "=====================")

    if options.mode == 'test':
        log(warning, "Test mode is activated")
        engine.start_epoch()
        engine.train_n_samples(options.batch_size)
        engine.end_epoch()
        engine.hook('on_end', engine.state)
        return

    test_every_nth = options.test_every_nth
    keep_last_n = options.keep_last_n
    oldest_kept = 0
    engine.state['maxepoch'] = options.epochs

    if test_every_nth > 0:
        for j in range(engine.state['epoch'], options.epochs, 1):
            engine.start_epoch()
            engine.train_n_samples(options.num_samples)
            if j % test_every_nth == test_every_nth - 1:
                engine.end_epoch()
                ckpt_name = make_checkpoint_path(name, j)
                engine.dump_state(ckpt_name)
                oldest_kept = remove_older_checkpoints(name, j, keep_last_n, oldest_kept)
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

    engine.hook('on_end', engine.state)


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
            bp.make_unique()
            return

        if 'bn' in options.unique:
            if issubclass(bp['type'], ScopedBatchNorm2d):
                bp.make_unique()

        if 'convdim' in options.unique:
            if issubclass(bp['type'], ScopedConv2d):
                if 'kernel_size' in bp['kwargs'] and bp['kwargs']['kernel_size'] == 1:
                    bp.make_unique()

        if 'conv' in options.unique:
            if issubclass(bp['type'], ScopedConv2d):
                if 'kernel_size' in bp['kwargs'] and bp['kwargs']['kernel_size'] == 3:
                    bp.make_unique()

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


def train_population(population, options, default_resnet_shape, default_densenet_shape):
    add_seed_individuals(population, options, default_resnet_shape, default_densenet_shape)

    net_blueprint = None
    for i in range(options.max_iteration):
        log(warning, 'Population generation: %d' % i)
        if i in options.lr_drop_epochs:
            options.lr *= options.lr_decay_ratio
        population.evolve_generation()
        index = population.get_the_best_index()
        net_blueprint = population.genotypes[index]
        best_score = net_blueprint['meta']['score']
        log(warning, "Current top score: {}, id: {}".format(best_score, id(net_blueprint)))

    return net_blueprint


def train_single_network(options, net=None):
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
            common_engine.train_n_samples(batch)
            generator_engine.train_n_samples(batch)

        # test every fourth epoch
        if j % 4 == 3:
            common_engine.end_epoch()
        else:
            common_engine.state['epoch'] += 1
        generator_engine.state['epoch'] += 1

    common_engine.hook('on_end', common_engine.state)
    generator_engine.hook('on_end', generator_engine.state)
