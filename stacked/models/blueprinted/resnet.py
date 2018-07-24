# -*- coding: utf-8 -*-
import torch
from torch.nn import ModuleList
import torch.nn.functional as F
from stacked.modules.scoped_nn import ScopedConv2d, \
    ScopedBatchNorm2d, ScopedLinear, ScopedReLU
from stacked.meta.scope import ScopedMeta
from stacked.meta.sequential import Sequential
from stacked.meta.blueprint import Blueprint, make_module
from stacked.models.blueprinted.resgroup import ScopedResGroup
from stacked.models.blueprinted.resblock import ScopedResBlock
from stacked.models.blueprinted.conv_unit import ScopedConvUnit
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

        # readjust linear layer and bns in case there has been mutation / crossover
        self.shortcut_index = ScopedResNet.__readjust_tail(blueprint['prefix'],
                                                           blueprint,
                                                           blueprint['shortcut_index'])
        self.bns = ModuleList()
        for bp in blueprint['bns']:
            self.bns.append(make_module(bp))

        assert(len(self.bns) + self.shortcut_index == len(self.container))

        self.linear = make_module(blueprint['linear'])
        self.callback = blueprint['callback']

    def forward(self, x):
        return self.function(self.conv, self.container, self.bns,
                             self.act, self.linear, self.shortcut_index,
                             self.callback, self.scope, id(self), x)

    @staticmethod
    def function(conv, container, bns, act, linear, shortcut_index,
                 callback, scope, module_id, x):
        x = conv(x)
        group_outputs = []
        for i, group in enumerate(container):
            x = group(x)
            if i >= shortcut_index:
                group_outputs.append(x)

        o = None
        for i, x in enumerate(group_outputs):
            g = act(bns[i](x))
            g = F.avg_pool2d(g, g.size()[2], 1, 0)

            if o is None:
                o = g
            else:
                o = torch.cat((o, g), 1)

        o = o.view(o.size(0), -1)
        o = linear(o)
        callback(scope, module_id, o)
        return o

    @staticmethod
    def get_num_blocks_per_group(depth, num_groups, block_depth):
        return (depth - 4) // (num_groups * block_depth)

    @staticmethod
    def __set_default_items(prefix, default, shape, ni, no, kernel_size,
                            act_module, conv_module, dilation=1,
                            groups=1, bias=False, callback=all_to_none,
                            conv_kwargs=None, act_kwargs=None):
        """Set blueprint items that are not Sequential type"""

        default['input_shape'] = shape
        default['callback'] = callback

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

        # return input shape for __set_children
        return default['conv']['output_shape']

    @staticmethod
    def __readjust_tail(prefix, default, shortcut_index=-1,
                        linear_module=ScopedLinear,
                        bn_module=ScopedBatchNorm2d):
        default['shortcut_index'] = shortcut_index
        batch_size = default['input_shape'][0]
        num_classes = default['output_shape'][1]

        cut_index = 0
        children = default['children']
        out_shape = children[-1]['output_shape']

        for i, c in enumerate(children):
            # ignore the rest of the children when depth is reached
            cut_index = i + 1
            if i >= default['depth'] and out_shape == c['output_shape']:
                break

        default['depth'] = cut_index

        widths = [c['output_shape'][1] for c in children[:cut_index]]

        default['bns'] = []
        default['shortcut_index'] = shortcut_index

        if shortcut_index < 0:
            shortcut_index = cut_index + shortcut_index

        for i, w in enumerate(widths):
            if i >= shortcut_index:
                default['bns'].append(Blueprint('%s/bn' % prefix, '%d' % w, default, True,
                                                bn_module, kwargs={'num_features': w}))

        in_features = sum(widths[shortcut_index:cut_index])

        kwargs = {'in_features': in_features, 'out_features': num_classes}
        default['linear'] = Blueprint('%s/linear' % prefix, '%d_%d' % (in_features, num_classes),
                                      default, True, linear_module, kwargs=kwargs)
        default['linear']['input_shape'] = (batch_size, in_features)
        default['linear']['output_shape'] = (batch_size, num_classes)
        return shortcut_index

    @staticmethod
    def __set_children(prefix, default, ni, widths, group_depths,
                       block_depth, block_module, conv_module,
                       bn_module, act_module, kernel_size, stride,
                       padding, shape, dilation=1, groups=1, bias=False,
                       callback=all_to_none, drop_p=0.0, dropout_p=0.0,
                       residual=True, conv_kwargs=None,
                       bn_kwargs=None, act_kwargs=None,
                       unit_module=ScopedConvUnit, group_module=ScopedResGroup,
                       fractal_depth=1, dense_unit_module=ScopedConvUnit):
        """Sequentially set children and bn blueprints"""
        children = []
        default['bns'] = []

        for i, (width, group_depth) in enumerate(zip(widths, group_depths)):
            no = width
            suffix = '%d_%d_%d_%d_%d_%d_%d_%d' % (ni, no, kernel_size, stride,
                                                  padding, dilation, groups, bias)

            block = group_module.describe_default('%s/group' % prefix, suffix,
                                                  default, shape, ni, no,
                                                  kernel_size, stride, padding,
                                                  dilation, groups, bias,
                                                  act_module, bn_module, conv_module,
                                                  callback, conv_kwargs,
                                                  bn_kwargs, act_kwargs, unit_module,
                                                  block_depth, dropout_p, residual,
                                                  block_module=block_module,
                                                  group_depth=group_depth, drop_p=drop_p,
                                                  fractal_depth=fractal_depth,
                                                  dense_unit_module=dense_unit_module)
            shape = block['output_shape']
            children.append(block)
            stride = 2
            ni = shape[1]

        default['children'] = children
        default['depth'] = len(children)
        return shape

    @staticmethod
    def __get_default(prefix, suffix, parent, shape, ni, no, kernel_size,
                      num_classes, block_module, bn_module, act_module,
                      conv_module, widths, group_depths,
                      block_depth, stride, padding, dilation=1, groups=1,
                      bias=False, callback=all_to_none, drop_p=0.0,
                      dropout_p=0.0, residual=True, conv_kwargs=None,
                      bn_kwargs=None, act_kwargs=None,
                      unit_module=ScopedConvUnit, group_module=ScopedResGroup,
                      fractal_depth=1, dense_unit_module=ScopedConvUnit):
        """Set the items and the children of the default blueprint object"""
        default = Blueprint(prefix, suffix, parent, False, ScopedResNet)

        shape = ScopedResNet.__set_default_items(prefix, default, shape, ni, no,
                                                 kernel_size, act_module, conv_module,
                                                 dilation, groups, bias, callback,
                                                 conv_kwargs, act_kwargs)

        shape = ScopedResNet.__set_children(prefix, default, ni, widths,
                                            group_depths, block_depth,
                                            block_module, conv_module,
                                            bn_module, act_module,
                                            kernel_size, stride, padding, shape,
                                            dilation, groups, bias, callback,
                                            drop_p, dropout_p, residual,
                                            conv_kwargs, bn_kwargs, act_kwargs,
                                            unit_module, group_module,
                                            fractal_depth, dense_unit_module)

        default['output_shape'] = (shape[0], num_classes)
        default['kwargs'] = {'blueprint': default, 'kernel_size': kernel_size,
                             'stride': stride, 'padding': padding, 'dilation': dilation,
                             'groups': groups, 'bias': bias}
        return default

    @staticmethod
    def describe_default(prefix='ResNet', suffix='', parent=None,
                         skeleton=(16, 32, 64), group_depths=None,
                         num_classes=10, depth=28,
                         width=1, block_depth=2,
                         block_module=ScopedResBlock, conv_module=ScopedConv2d,
                         bn_module=ScopedBatchNorm2d, linear_module=ScopedLinear,
                         act_module=ScopedReLU,
                         kernel_size=3, padding=1, input_shape=None,
                         dilation=1, groups=1, bias=False, callback=all_to_none,
                         drop_p=0.0, dropout_p=0.0, residual=True,
                         conv_kwargs=None, bn_kwargs=None, act_kwargs=None,
                         unit_module=ScopedConvUnit, group_module=ScopedResGroup,
                         fractal_depth=1, shortcut_index=-1,
                         dense_unit_module=ScopedConvUnit, *_, **__):
        """Create a default ResNet blueprint

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
            block_module: Children modules used as block modules
            conv_module: CNN module to use in forward. e.g. ScopedConv2d
            bn_module: Batch normalization module. e.g. ScopedBatchNorm2d
            linear_module: Linear module for classification e.g. ScopedLinear
            act_module: Activation module e.g ScopedReLU
            kernel_size (int or tuple): Size of the convolving kernel.
            padding (int or tuple, optional): Padding for the first convolution
            input_shape (tuple): (N, C_{in}, H_{in}, W_{in})
            dilation (int): Spacing between kernel elements.
            groups (int): Number of blocked connections from input to output channels.
            bias (bool): Add a learnable bias if True
            callback: Function to call after the output in forward is calculated
            drop_p (float): Probability of vertical drop
            dropout_p (float): Probability of dropout in the blocks
            residual (bool): True if a shortcut connection will be used
            conv_kwargs: Extra conv arguments to be used in children
            bn_kwargs: Extra bn args, if bn module requires other than 'num_features'
            act_kwargs: Extra act args, if act module requires other than defaults
            unit_module: Basic building unit of resblock
            group_module: Basic building group of resnet
            fractal_depth (int): Recursion depth for fractal group module
            shortcut_index (int): Starting index for groups shortcuts to the linear layer
            dense_unit_module: Children modules that will be used in dense connections
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
                                             block_module, bn_module,
                                             act_module, conv_module,
                                             widths, group_depths,
                                             block_depth, stride, padding,
                                             dilation, groups, bias,
                                             callback, drop_p, dropout_p, residual,
                                             conv_kwargs, bn_kwargs, act_kwargs,
                                             unit_module, group_module,
                                             fractal_depth, dense_unit_module)

        ScopedResNet.__readjust_tail(prefix, default,
                                     shortcut_index=shortcut_index,
                                     linear_module=linear_module,
                                     bn_module=bn_module)
        return default
