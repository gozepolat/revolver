# -*- coding: utf-8 -*-
from stacked.models.blueprinted.optimizer import ScopedEpochEngine
from stacked.models.blueprinted.resnet import ScopedResNet
from stacked.models.blueprinted.meta import ScopedMetaMasked
from stacked.modules.scoped_nn import ScopedConv2d, ScopedBatchNorm2d
from stacked.meta.blueprint import make_module, visit_modules
from stacked.utils import common
import argparse
import json
import os

import torch.backends.cudnn as cudnn


cudnn.benchmark = True


def parse_args():
    parser = argparse.ArgumentParser(description='In construction..')

    parser.add_argument('--depth', default=22, type=int)
    parser.add_argument('--skeleton', default='[16,32,64]', type=str,
                        help='json list with epochs to drop lr on')
    parser.add_argument('--block_depth', default=2, type=int)
    parser.add_argument('--width', default=1, type=int)
    parser.add_argument('--dataset', default='CIFAR10', type=str)
    parser.add_argument('--num_thread', default=4, type=int)

    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--lr', default=0.1, type=float)
    parser.add_argument('--epochs', default=200, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--weight_decay', default=0.0005, type=float)
    parser.add_argument('--lr_drop_epochs', default='[60,120,180]', type=str,
                        help='json list with epochs to drop lr on')
    parser.add_argument('--lr_decay_ratio', default=0.2, type=float)
    parser.add_argument('--single_engine', default=True, type=bool)
    parser.add_argument('--gpu_id', default='0', type=str,
                        help='id(s) for CUDA_VISIBLE_DEVICES')

    parsed_args = parser.parse_args()
    return parsed_args


def create_engine_pair(net_blueprint, options, epochs):
    """engines to train different portions of the given model"""
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
                                                                 lr_drop_epochs=epochs,
                                                                 dataset=options.dataset,
                                                                 num_thread=options.num_thread,
                                                                 optimizer_parameter_picker=common_picker,
                                                                 weight_decay=options.weight_decay)

    # accesses the same resnet model instance
    generator_engine_blueprint = ScopedEpochEngine.describe_default(prefix='GeneratorEpochEngine',
                                                                    net_blueprint=net_blueprint,
                                                                    max_epoch=options.epochs,
                                                                    batch_size=options.batch_size,
                                                                    learning_rate=options.lr * 0.2,
                                                                    lr_decay_ratio=options.lr_decay_ratio,
                                                                    lr_drop_epochs=epochs,
                                                                    dataset=options.dataset,
                                                                    num_thread=options.num_thread,
                                                                    optimizer_parameter_picker=generator_picker,
                                                                    weight_decay=options.weight_decay)
    c = make_module(common_engine_blueprint)
    g = make_module(generator_engine_blueprint)
    return c, g


if __name__ == '__main__':
    parsed = parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = parsed.gpu_id

    num_classes = 10 if parsed.dataset == 'CIFAR10' else 100

    lr_drop_epochs = json.loads(parsed.lr_drop_epochs)
    skeleton = json.loads(parsed.skeleton)

    parsed.skeleton = skeleton
    parsed.num_classes = num_classes
    print(parsed)

    common.BLUEPRINT_GUI = False
    input_shape = (parsed.batch_size, 3, 32, 32)
    resnet = ScopedResNet.describe_default(prefix='ResNet', num_classes=num_classes,
                                           depth=parsed.depth, width=parsed.width,
                                           block_depth=parsed.block_depth,
                                           conv_module=ScopedMetaMasked,
                                           skeleton=skeleton, input_shape=input_shape)

    def make_conv2d_unique(bp, _, __):
        if issubclass(bp['type'], ScopedBatchNorm2d):
            bp.make_unique()
        if issubclass(bp['type'], ScopedConv2d):
            if 'kernel_size' in bp['kwargs'] and bp['kwargs']['kernel_size'] == 1:
                bp.make_unique()


    visit_modules(resnet, None, None, make_conv2d_unique)
    exit(0)
    engine_blueprint = ScopedEpochEngine.describe_default(prefix='EpochEngine',
                                                          net_blueprint=resnet,
                                                          max_epoch=parsed.epochs,
                                                          batch_size=parsed.batch_size,
                                                          learning_rate=parsed.lr,
                                                          lr_decay_ratio=parsed.lr_decay_ratio,
                                                          lr_drop_epochs=lr_drop_epochs,
                                                          dataset=parsed.dataset,
                                                          num_thread=parsed.num_thread,
                                                          use_tqdm=True,
                                                          weight_decay=parsed.weight_decay)

    if parsed.single_engine:
        engine = make_module(engine_blueprint)
        for j in range(parsed.epochs):
            engine.train_one_epoch()
        engine.hook('on_end', engine.state)
    else:
        common_engine, generator_engine = create_engine_pair(resnet, parsed, lr_drop_epochs)
        for j in range(parsed.epochs):
            common_engine.start_epoch()
            generator_engine.start_epoch()

            # train back and forth
            for i in range(23):
                common_engine.train_n_samples(128 * 17)
                generator_engine.train_n_samples(128 * 17)

            # test every fourth epoch
            if j % 4 == 3:
                common_engine.end_epoch()
            else:
                common_engine.state['epoch'] += 1
            generator_engine.state['epoch'] += 1

        common_engine.hook('on_end', common_engine.state)
        generator_engine.hook('on_end', generator_engine.state)
