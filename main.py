# -*- coding: utf-8 -*-
from stacked.models.blueprinted.resnet import ScopedResNet
from stacked.models.blueprinted.denseconcatgroup import ScopedDenseConcatGroup
from stacked.models.blueprinted.bottleneckblock import ScopedBottleneckBlock
from stacked.modules.scoped_nn import ScopedCrossEntropyLoss
from stacked.models.blueprinted.meta import ScopedMetaMasked
from stacked.utils.transformer import all_to_none
from stacked.utils import common
from stacked.utils.usage_helpers import train_single_network, adjust_options
import argparse
import os

import torch.backends.cudnn as cudnn

cudnn.benchmark = True


def parse_args():
    parser = argparse.ArgumentParser(description='In construction..')

    parser.add_argument('--depth', default=22, type=int)
    parser.add_argument('--skeleton', default='[12,24,48]', type=str,
                        help='json list with epochs to drop lr on')
    parser.add_argument('--block_depth', default=2, type=int)
    parser.add_argument('--width', default=1, type=int)
    parser.add_argument('--dataset', default='CIFAR10', type=str)
    parser.add_argument('--num_thread', default=4, type=int)

    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--lr', default=0.1, type=float)
    parser.add_argument('--epochs', default=300, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--weight_decay', default=0.00005, type=float)
    parser.add_argument('--lr_drop_epochs', default='[150,225]', type=str,
                        help='json list with epochs to drop lr on')
    parser.add_argument('--lr_decay_ratio', default=0.1, type=float)
    parser.add_argument('--single_engine', default=True, type=bool)
    parser.add_argument('--gpu_id', default='0', type=str,
                        help='id(s) for CUDA_VISIBLE_DEVICES')
    parser.add_argument('--save_folder', default='.', type=str,
                        help="path to save the blueprint and engine state")
    parser.add_argument('--load_path', default='', type=str,
                        help="path to load the blueprint and engine state")
    parser.add_argument('--save_png_folder', default='', type=str,
                        help="path to save weight visualization output")

    parsed_args = parser.parse_args()
    return parsed_args


def set_default_options_for_single_network(options):
    """Default options for the single network training"""
    options.conv_module = ScopedMetaMasked
    options.dropout_p = 0.0
    options.drop_p = 0.5
    options.fractal_depth = 4
    options.net = ScopedResNet
    options.callback = all_to_none
    options.criterion = ScopedCrossEntropyLoss
    options.residual = False
    options.group_module = ScopedDenseConcatGroup
    options.block_module = ScopedBottleneckBlock
    options.dense_unit_module = ScopedBottleneckBlock
    options.head_kernel = 3
    options.head_stride = 1
    options.head_padding = 1
    options.head_pool_kernel = 3
    options.head_pool_stride = 2
    options.head_pool_padding = 1
    options.head_modules = ('conv', 'bn')
    options.unique = ('bn',)


if __name__ == '__main__':
    common.BLUEPRINT_GUI = False
    parsed = parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = parsed.gpu_id
    adjust_options(parsed)

    if parsed.mode == 'single_train':
        set_default_options_for_single_network(parsed)
        train_single_network(parsed)

    # TODO next
    #elif parsed.mode == 'population_train':
    #    set_default_options_for_population(parsed)
    #    train_population(parsed)

    # dump all options
    print(parsed)
