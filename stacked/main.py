# -*- coding: utf-8 -*-
from stacked.models.blueprinted.optimizer import ScopedEpochEngine
from stacked.models.blueprinted.resnet import ScopedResNet
import argparse
import json

if __name__ == '__main__':
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
    parser.add_argument('--epochs', default=3, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--weight_decay', default=0.0005, type=float)
    parser.add_argument('--lr_drop_epochs', default='[60,120,160]', type=str,
                        help='json list with epochs to drop lr on')
    parser.add_argument('--lr_decay_ratio', default=0.2, type=float)
    parser.add_argument('--gpu_id', default='0', type=str,
                        help='id(s) for CUDA_VISIBLE_DEVICES')

    parsed = parser.parse_args()

    num_classes = 10 if parsed.dataset == 'CIFAR10' else 100
    epoch_step = json.loads(parsed.epoch_step)
    skeleton = json.loads(parsed.skeleton)

    resnet = ScopedResNet.describe_default(prefix='ResNet', num_classes=num_classes,
                                           depth=parsed.depth, width=parsed.width,
                                           block_depth=parsed.block_depth,
                                           skeleton=skeleton)

    engine_blueprint = ScopedEpochEngine.describe_default(prefix='EpochEngine',
                                                          suffix='', parent=None,
                                                          net_blueprint=resnet,
                                                          max_epoch=parsed.epochs,
                                                          batch_size=parsed.batch_size,
                                                          learning_rate=parsed.lr,
                                                          lr_decay_ratio=parsed.lr_decay_ratio,
                                                          lr_drop_epochs=parsed.lr_drop_epochs,
                                                          dataset=parsed.dataset,
                                                          num_thread=parsed.num_thread)