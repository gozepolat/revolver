# -*- coding: utf-8 -*-
from revolver.models.blueprinted.resnet import ScopedResNet
from revolver.models.blueprinted.densenet import ScopedDenseNet
from revolver.models.blueprinted.resgroup import ScopedResGroup
# from revolver.models.blueprinted.tree import ScopedTreeGroup
from revolver.models.blueprinted.resblock import ScopedResBlock
from revolver.models.blueprinted.denseconcatgroup import ScopedDenseConcatGroup
from revolver.models.blueprinted.convdeconv import ScopedConv2dDeconv2d
from revolver.models.blueprinted.bottleneckblock import ScopedBottleneckBlock
from revolver.modules.scoped_nn import ScopedCrossEntropyLoss, ScopedConv2d, ScopedConv2dDeconv2dConcat
from revolver.models.blueprinted.meta import ScopedMetaMasked
from revolver.utils.transformer import all_to_none
from revolver.utils import common
from revolver.utils.usage_helpers import train_single_network, \
    adjust_options, create_single_engine, train_with_single_engine, \
    train_population
from revolver.meta.heuristics.population import generate_net_blueprints, \
    get_phenotype_score, Population
import argparse
import os

import torch.backends.cudnn as cudnn

cudnn.benchmark = True


def parse_args():
    parser = argparse.ArgumentParser(description='In construction..')

    parser.add_argument('--mode', default='single_train', type=str,
                        help="single, or population train mode")
    parser.add_argument('--depth', default=40, type=int)
    parser.add_argument('--sample_size', default=2, type=int)
    parser.add_argument('--population_size', default=100, type=int)
    parser.add_argument('--max_skeleton_width', default=64, type=int)
    parser.add_argument('--max_skeleton_depth', default=8, type=int)
    parser.add_argument('--warmup_x', default=10, type=int)
    parser.add_argument('--skeleton', default='[12,24,48]', type=str,
                        help='json list with epochs to drop lr on')
    parser.add_argument('--block_depth', default=2, type=int)
    parser.add_argument('--width', default=1, type=int)
    parser.add_argument('--dataset', default='CIFAR10', type=str)
    parser.add_argument('--num_thread', default=0, type=int)
    parser.add_argument('--add_seed', default="n", type=str)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--lr', default=0.1, type=float)
    parser.add_argument('--epochs', default=300, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--finetune_epochs', default=10, type=int, metavar='N',
                        help='number of fine tuning epochs to run after evolving')
    parser.add_argument('--finetune_after_evolving', default="y", type=str)
    parser.add_argument('--weight_decay', default=0.0001, type=float)
    parser.add_argument('--genotype_cost', default=32, type=float,
                        help='Coefficient to increase genotype cost.'
                             ' Low test loss reduces the overall cost.')
    parser.add_argument('--lr_drop_epochs', default='[150,225]', type=str,
                        help='json list with epochs to drop lr on')
    parser.add_argument('--min_lr', default=common.MINIMUM_LEARNING_RATE, type=float)
    parser.add_argument('--gradual_lr_drop', default=100, type=int,
                        help='Drop learning rate gradually')
    parser.add_argument('--lr_decay_ratio', default=0.1, type=float)
    parser.add_argument('--p_initialize_with_unique', default=0.5, type=float,
                        help='Probability that a component will be made unique when initialized or mutated')
    parser.add_argument('--single_engine', default=True, type=bool)
    parser.add_argument('--gpu_id', default='0', type=str,
                        help='id(s) for CUDA_VISIBLE_DEVICES')
    parser.add_argument('--save_folder', default='.', type=str,
                        help="path to save the blueprint and engine state")
    parser.add_argument('--load_path', default='', type=str,
                        help="path to load the blueprint and engine state")
    parser.add_argument('--warmup_epoch', default=20, type=int,
                        help="Update population with higher quality genotypes based on genotype fitness")
    parser.add_argument('--save_png_folder', default='', type=str,
                        help="path to save weight visualization output")
    parser.add_argument('--search_mode', default='evolve', type=str,
                        help="evolve: one time population init and genetic operators for search (default)"
                             "random: population init and then randomly generate genotypes to replace the worst"
                             "random_warmup: first random search on genotypes, then continue with evolution"
                             "evolve_warmup: first evolve genotypes without training then with training")

    parsed_args = parser.parse_args()
    return parsed_args


def set_default_options_for_single_network(options):
    """Default options for the single network training"""
    options.conv_module = ScopedConv2dDeconv2d
    options.dropout_p = 0.0
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
    options.keep_last_n = 5
    options.load_latest_checkpoint = True
    options.engine_pkl = None


def set_default_options_for_population(options):
    """Default options for the single network training"""
    set_default_options_for_single_network(options)

    options.net = ScopedDenseNet
    options.conv_module = ScopedConv2d
    # log base for the number of parameters
    options.params_favor_rate = 100

    options.epoch_per_generation = 1

    options.update_score_weight = 0.5
    options.max_iteration = options.epochs

    # default heuristics
    options.generator = generate_net_blueprints
    options.utility = get_phenotype_score
    options.engine_maker = create_single_engine

    # disable engine loading, and tqdm
    options.load_path = ''
    options.use_tqdm = False


if __name__ == '__main__':
    common.BLUEPRINT_GUI = False
    parsed = parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = parsed.gpu_id

    gpu_id = parsed.gpu_id = int(parsed.gpu_id)
    gpu_info = common.get_gpu_memory_info()
    (used, total) = gpu_info[gpu_id]

    print("Overall gpu info: {}".format(gpu_info))
    print("gpu {}, has {} used, {} total".format(gpu_id, used, total))

    adjust_options(parsed)
    print(vars(parsed))

    if parsed.mode == 'population_train':
        assert (used * 10 < total)  # only run with a relatively empty gpu
        set_default_options_for_population(parsed)
        common.POPULATION_GENOTYPE_COST_COEFFICIENT = parsed.genotype_cost

        # population generates new genotypes and estimates their scores with get_genotype_cost
        p = Population(parsed)

        net_blueprint = train_population(p, parsed,
                                         default_resnet_shape=(40, 9),
                                         default_densenet_shape=(190, 40))

        print("Best model blueprint: %s" % net_blueprint['name'])
        print(parsed)
        if parsed.finetune_after_evolving in common.YES_SET:
            set_default_options_for_single_network(parsed)
            parsed.lr = 0.0005
            parsed.lr_decay_ratio = 0.5
            parsed.weight_decay = 0.0
            parsed.epochs = parsed.finetune_epochs
            parsed.lr_drop_epochs = (5, 8)
            parsed.mode = "single_train"
            train_with_single_engine(net_blueprint, parsed)

            parsed.mode = "test"
            print("Best model blueprint: %s" % net_blueprint['name'])
            train_with_single_engine(net_blueprint, parsed)
            print("Best model blueprint: %s" % net_blueprint['name'])

        net_blueprint.dump_pickle(f"../best_{net_blueprint['name']}_"
                                  f"{parsed.genotype_cost}_{parsed.search_mode}_{parsed.dataset}.pkl")
    else:
        set_default_options_for_single_network(parsed)
        train_single_network(parsed)

    # dump all options
    print(parsed)
