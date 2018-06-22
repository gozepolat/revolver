# -*- coding: utf-8 -*-
import torch.nn.functional as F
from stacked.modules.scoped_nn import ScopedConv2d, \
    ScopedBatchNorm2d, ScopedLinear, ScopedReLU
from stacked.meta.scope import ScopedMeta
from stacked.meta.sequential import Sequential
from stacked.meta.blueprint import Blueprint, make_module
from stacked.models.blueprinted.resgroup import ScopedResGroup
from stacked.utils.transformer import all_to_none
from six import add_metaclass


@add_metaclass(ScopedMeta)
class ScopedResNet(Sequential):
    """WRN inspired implementation of ResNet

    Args:
        scope (string): Scope for the self (ScopedResBlock instance)
        blueprint: Description of the scopes and member module types
    """
    def __init__(self, scope, blueprint, *_, **__):
        super(ScopedResNet, self).__init__(blueprint)
        self.scope = scope
        self.blueprint = blueprint

        self.act = make_module(blueprint['act'])
        self.conv = make_module(blueprint['conv'])
        self.bn = make_module(blueprint['bn'])
        self.linear = make_module(blueprint['linear'])
        self.callback = blueprint['callback']

    def forward(self, x):
        return self.function(self.conv, self.container,
                             self.bn, self.act, self.linear,
                             self.callback, self.scope, id(self), x)

    @staticmethod
    def function(conv, container, bn, act, linear,
                 callback, scope, module_id, x):
        x = conv(x)
        for group in container:
            x = group(x)

        o = act(bn(x))
        o = F.avg_pool2d(o, o.size()[2], 1, 0)
        o = o.view(o.size(0), -1)
        o = linear(o)
        callback(scope, module_id, o)
        return o

    @staticmethod
    def get_num_blocks_per_group(depth, num_groups, block_depth):
        return (depth - 4) // (num_groups * block_depth)

    @staticmethod
    def __set_default_items(prefix, default, shape, ni, no, kernel_size, num_classes,
                            bn_module, act_module, conv_module, linear_module,
                            dilation=1, groups=1, bias=False, callback=all_to_none,
                            conv_kwargs=None, bn_kwargs=None, act_kwargs=None):
        """Set blueprint items that are not Sequential type"""

        default['input_shape'] = shape
        default['callback'] = callback

        if bn_kwargs is None:
            bn_kwargs = {'num_features': no}
        default['bn'] = Blueprint('%s/bn' % prefix, '%d' % no, default, False,
                                  bn_module, kwargs=bn_kwargs)
        if act_kwargs is None:
            if issubclass(act_module, ScopedReLU):
                act_kwargs = {'inplace': True}
        default['act'] = Blueprint('%s/act' % prefix, '%d' % no, default, False,
                                   act_module, kwargs=act_kwargs)
        # describe conv
        suffix = '%d_%d_%d_%d_%d_%d_%d_%d' % (shape[1], ni, kernel_size, 1,
                                              1, dilation, groups, bias)

        default['conv'] = conv_module.describe_default('%s/conv' % prefix, suffix,
                                                       default, shape,
                                                       in_channels=shape[1],
                                                       out_channels=ni,
                                                       kernel_size=kernel_size,
                                                       stride=1, padding=1,
                                                       dilation=dilation,
                                                       groups=groups, bias=bias,
                                                       callback=callback,
                                                       conv_kwargs=conv_kwargs)

        # describe linear, (shapes will be set after children)
        kwargs = {'in_features': no, 'out_features': num_classes}
        default['linear'] = Blueprint('%s/linear' % prefix,
                                      '%d_%d' % (no, num_classes),
                                      default, False, linear_module,
                                      kwargs=kwargs)

        # return input shape for __set_default_children
        return default['conv']['output_shape']

    @staticmethod
    def __set_default_children(prefix, default, ni, widths, group_depths,
                               block_depth, conv_module, bn_module, act_module,
                               kernel_size, stride, padding, shape,
                               dilation=1, groups=1, bias=False,
                               callback=all_to_none,
                               drop_p=0.0, dropout_p=0.0, conv_kwargs=None,
                               bn_kwargs=None, act_kwargs=None):
        """Sequentially set children blueprints"""
        children = []
        for width, group_depth in zip(widths, group_depths):
            no = width
            suffix = '%d_%d_%d_%d_%d_%d_%d_%d' % (ni, no, kernel_size, stride,
                                                  padding, dilation, groups, bias)

            block = ScopedResGroup.describe_default('%s/group' % prefix, suffix,
                                                    default, shape,
                                                    ni, no, kernel_size, stride, padding,
                                                    dilation, groups, bias,
                                                    act_module, bn_module, conv_module,
                                                    group_depth, block_depth,
                                                    callback, drop_p, dropout_p,
                                                    conv_kwargs, bn_kwargs, act_kwargs)
            shape = block['output_shape']
            children.append(block)
            stride = 2
            ni = no

        default['children'] = children
        default['depth'] = len(children)
        return shape

    @staticmethod
    def __get_default(prefix, suffix, parent, shape, ni, no, kernel_size,
                      num_classes, bn_module, act_module, conv_module, linear_module,
                      widths, group_depths, block_depth, stride, padding,
                      dilation=1, groups=1, bias=False, callback=all_to_none,
                      drop_p=0.0, dropout_p=0.0, conv_kwargs=None,
                      bn_kwargs=None, act_kwargs=None):
        """Set the items and the children of the default blueprint object"""
        default = Blueprint(prefix, suffix, parent, False, ScopedResNet)
        shape = ScopedResNet.__set_default_items(prefix, default, shape, ni, no,
                                                 kernel_size, num_classes, bn_module,
                                                 act_module, conv_module, linear_module,
                                                 dilation, groups, bias, callback,
                                                 conv_kwargs, bn_kwargs, act_kwargs)

        shape = ScopedResNet.__set_default_children(prefix, default, ni, widths,
                                                    group_depths, block_depth, conv_module,
                                                    bn_module, act_module, kernel_size,
                                                    stride, padding, shape, dilation,
                                                    groups, bias, callback,
                                                    drop_p, dropout_p, conv_kwargs,
                                                    bn_kwargs, act_kwargs)

        default['linear']['input_shape'] = (shape[0], shape[1])
        default['linear']['output_shape'] = (shape[0], num_classes)
        default['output_shape'] = (shape[0], num_classes)
        default['kwargs'] = {'blueprint': default, 'kernel_size': kernel_size,
                             'stride': stride, 'padding': padding, 'dilation': dilation,
                             'groups': groups, 'bias': bias}
        return default

    @staticmethod
    def describe_default(prefix='ResNet', suffix='', parent=None,
                         skeleton=(16, 32, 64), group_depths=None,
                         num_classes=10, depth=28, width=1,
                         block_depth=2, conv_module=ScopedConv2d,
                         bn_module=ScopedBatchNorm2d, linear_module=ScopedLinear,
                         act_module=ScopedReLU, kernel_size=3, padding=1,
                         input_shape=None, dilation=1, groups=1, bias=False,
                         callback=all_to_none, drop_p=0.0, dropout_p=0.0,
                         conv_kwargs=None, bn_kwargs=None, act_kwargs=None):
        """Create a default ResBlock blueprint

        Args:
            prefix (str): Prefix from which the member scopes will be created
            suffix (str): Suffix to append the name of the scoped object
            parent (Blueprint): None or the instance of the parent blueprint
            skeleton (iterable): Smallest possible widths per group
            group_depths (iterable): Finer grained group depth description
            num_classes (int): Number of categories for supervised learning
            depth (int): Overall depth of the network
            width (int): Scalar to get the scaled width per group
            block_depth (int): Number of [conv/act/bn] units in the block
            conv_module (type): CNN module to use in forward. e.g. ScopedConv2d
            bn_module: Batch normalization module. e.g. ScopedBatchNorm2d
            linear_module (type): Linear module for classification e.g. ScopedLinear
            act_module (type): Activation module e.g ScopedReLU
            kernel_size (int or tuple): Size of the convolving kernel.
            padding (int or tuple, optional): Padding for the first convolution
            input_shape (tuple): (N, C_{in}, H_{in}, W_{in})
            dilation (int): Spacing between kernel elements.
            groups: Number of blocked connections from input to output channels.
            bias: Add a learnable bias if True
            callback: function to call after the output in forward is calculated
            drop_p: Probability of vertical drop
            dropout_p: Probability of dropout in the blocks
            conv_kwargs: extra conv arguments to be used in children
            bn_kwargs: extra bn args, if bn module requires other than 'num_features'
            act_kwargs: extra act args, if act module requires other than defaults
        """
        if input_shape is None:
            # assume batch_size = 1, in_channels: 3, h: 32, and w : 32
            input_shape = (1, 3, 32, 32)

        widths = [i * width for i in skeleton]
        stride = 1
        num_groups = len(skeleton)

        if group_depths is None:
            group_depth = ScopedResNet.get_num_blocks_per_group(depth, num_groups,
                                                                block_depth)
            group_depths = []
            for _ in widths:
                group_depths.append(group_depth)

        ni = skeleton[0]
        no = widths[-1]

        default = ScopedResNet.__get_default(prefix, suffix, parent,
                                             input_shape, ni, no,
                                             kernel_size, num_classes,
                                             bn_module, act_module,
                                             conv_module, linear_module,
                                             widths, group_depths,
                                             block_depth, stride, padding,
                                             dilation, groups, bias,
                                             callback, drop_p, dropout_p,
                                             conv_kwargs, bn_kwargs, act_kwargs)
        return default
