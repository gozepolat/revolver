# -*- coding: utf-8 -*-
import torch.nn.functional as F
from stacked.modules.scoped_nn import ScopedConv2d, \
    ScopedBatchNorm2d, ScopedLinear, ScopedReLU
from stacked.meta.scope import ScopedMeta
from stacked.meta.sequential import Sequential
from stacked.meta.blueprint import Blueprint, make_module
from stacked.modules.conv import get_conv_out_shape
from stacked.models.blueprinted.resgroup import ScopedResGroup
from six import add_metaclass


@add_metaclass(ScopedMeta)
class ScopedResNet(Sequential):
    """WRN inspired implementation of ResNet

    Args:
        scope (string): Scope for the self (ScopedResBlock instance)
        blueprint: Description of the scopes and member module types
    """
    def __init__(self, scope, blueprint):
        super(ScopedResNet, self).__init__(blueprint)
        self.scope = scope

        self.act = make_module(blueprint['act'])
        self.conv = make_module(blueprint['conv'])
        self.bn = make_module(blueprint['bn'])
        self.linear = make_module(blueprint['linear'])

    def forward(self, x):
        return self.function(self.conv, self.container,
                             self.bn, self.act, self.linear, x)

    @staticmethod
    def function(conv, container, bn, act, linear, x):
        x = conv(x)
        for group in container:
            x = group(x)
        o = act(bn(x))
        o = F.avg_pool2d(o, o.size()[2], 1, 0)
        o = o.view(o.size(0), -1)
        o = linear(o)
        return o

    @staticmethod
    def get_num_blocks_per_group(depth, num_groups, block_depth):
        return (depth - 4) // (num_groups * block_depth)

    @staticmethod
    def __set_default_items(prefix, default, shape, ni, no, kernel_size, num_classes,
                            bn_module, act_module, conv_module, linear_module):
        """Set blueprint items that are not Sequential type"""

        default['input_shape'] = shape
        default['bn'] = Blueprint('%s/bn' % prefix, '%d' % no, default, False,
                                  bn_module, kwargs={'num_features': no})
        default['act'] = Blueprint('%s/act' % prefix, '%d' % no, default, False,
                                   act_module, kwargs={'inplace': True})
        # describe conv
        kwargs = {'in_channels': shape[1], 'out_channels': ni,
                  'kernel_size': kernel_size, 'stride': 1, 'padding': 1}
        suffix = '%d_%d_%d_%d_%d' % (shape[1], ni, kernel_size, 1, 1)
        default['conv'] = Blueprint('%s/conv' % prefix, suffix,
                                    default, False, conv_module, kwargs=kwargs)
        default['conv']['input_shape'] = shape
        shape = get_conv_out_shape(shape, ni, kernel_size, 1, 1)
        default['conv']['output_shape'] = shape

        # describe linear, (shapes will be set after children)
        kwargs = {'in_features': no, 'out_features': num_classes}
        default['linear'] = Blueprint('%s/linear' % prefix,
                                      '%d_%d' % (no, num_classes),
                                      default, False, linear_module,
                                      kwargs=kwargs)
        return shape

    @staticmethod
    def __set_default_children(prefix, default, ni, widths, group_depth,
                               block_depth, conv_module, bn_module, act_module,
                               kernel_size, stride, padding, shape):
        """Sequentially set children blueprints"""
        children = []
        for width in widths:
            no = width
            suffix = '%d_%d_%d_%d_%d' % (ni, no, kernel_size, stride, padding)
            block = ScopedResGroup.describe_default('%s/group' % prefix, suffix,
                                                    default, group_depth, block_depth,
                                                    conv_module, bn_module, act_module,
                                                    ni, no, kernel_size, stride, padding,
                                                    shape)
            shape = block['output_shape']
            children.append(block)
            padding = 1
            stride = 2
            ni = no
        default['children'] = children
        return shape

    @staticmethod
    def __get_default(prefix, suffix, parent, shape, ni, no, kernel_size,
                      num_classes, bn_module, act_module, conv_module, linear_module,
                      widths, group_depth, block_depth, stride, padding):
        """Set the items and the children of the default blueprint object"""
        default = Blueprint(prefix, suffix, parent, False, ScopedResNet)
        shape = ScopedResNet.__set_default_items(prefix, default, shape, ni, no,
                                                 kernel_size, num_classes, bn_module,
                                                 act_module, conv_module, linear_module)

        shape = ScopedResNet.__set_default_children(prefix, default, ni, widths,
                                                    group_depth, block_depth, conv_module,
                                                    bn_module, act_module, kernel_size,
                                                    stride, padding, shape)

        default['linear']['input_shape'] = (shape[0], shape[1])
        default['linear']['output_shape'] = (shape[0], num_classes)
        default['output_shape'] = (shape[0], num_classes)
        default['kwargs'] = {'blueprint': default}
        return default

    @staticmethod
    def describe_default(prefix='ResNet', suffix='', parent=None, skeleton=(16, 32, 64),
                         num_classes=10, depth=28, width=1,
                         block_depth=2, conv_module=ScopedConv2d,
                         bn_module=ScopedBatchNorm2d, linear_module=ScopedLinear,
                         act_module=ScopedReLU, kernel_size=3, padding=1,
                         input_shape=None):
        """Create a default ResBlock blueprint

        Args:
            prefix (str): Prefix from which the member scopes will be created
            suffix (str): Suffix to append the name of the scoped object
            parent (Blueprint): None or the instance of the parent blueprint
            skeleton (iterable): Smallest possible widths per group
            num_classes (int): Number of categories for supervised learning
            depth (int): Overall depth of the network
            width (int): Scalar to get the scaled width per group
            block_depth (int): Number of [conv/act/bn] units in the block
            conv_module (type): CNN module to use in forward. e.g. ScopedConv2d
            bn_module (type): Batch normalization module. e.g. ScopedBatchNorm2d
            linear_module (type): Linear module for classification e.g. ScopedLinear
            act_module (type): Activation module e.g ScopedReLU
            kernel_size (int or tuple): Size of the convolving kernel.
            padding (int or tuple, optional): Padding for the first convolution
            input_shape (tuple): (N, C_{in}, H_{in}, W_{in})
        """
        if input_shape is None:
            # assume batch_size = 1, in_channels: 3, h: 32, and w : 32
            input_shape = (1, 3, 32, 32)

        widths = [i * width for i in skeleton]
        stride = 1
        num_groups = len(skeleton)
        group_depth = ScopedResNet.get_num_blocks_per_group(depth, num_groups, block_depth)
        ni = skeleton[0]
        no = widths[-1]

        default = ScopedResNet.__get_default(prefix, suffix, parent, input_shape, ni, no,
                                             kernel_size, num_classes, bn_module, act_module,
                                             conv_module, linear_module, widths, group_depth,
                                             block_depth, stride, padding)
        return default
