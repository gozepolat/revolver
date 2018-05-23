# -*- coding: utf-8 -*-
from stacked.modules.scoped_nn import ScopedBatchNorm2d, \
    ScopedReLU, ScopedTanh, ScopedConv2d
from stacked.meta.scope import ScopedMeta
from stacked.meta.sequential import Sequential
from stacked.meta.blueprint import Blueprint, make_module
from stacked.utils.transformer import all_to_none
from stacked.models.blueprinted.conv_unit import ScopedConvUnit
from six import add_metaclass


@add_metaclass(ScopedMeta)
class ScopedResBlock(Sequential):
    """Pre-ResNet block

    Args:
        scope: Scope for the self (ScopedResBlock instance)
        blueprint: Description of inner scopes, member modules, and args
    """

    def __init__(self, scope, blueprint, *_, **__):
        super(ScopedResBlock, self).__init__(blueprint)
        self.scope = scope
        self.blueprint = blueprint

        self.depth = len(blueprint['children'])
        self.bn = make_module(blueprint['bn'])
        self.act = make_module(blueprint['act'])
        self.conv = make_module(blueprint['conv'])
        self.callback = blueprint['callback']

        # 1x1 conv to correct the number of channels for summation
        self.convdim = make_module(blueprint['convdim'])

    def forward(self, x):
        return self.function(self.bn, self.act,
                             self.conv, self.container,
                             self.convdim, self.callback,
                             self.scope, x)

    @staticmethod
    def function(bn, act, conv, container, convdim,
                 callback, scope, x):
        if bn is not None:
            x = bn(x)

        o = act(x)
        z = conv(o)

        for unit in container:
            z = unit(z)

        if convdim is not None:
            z = z + convdim(o)
        else:
            z = z + x

        callback(scope, z)
        return z

    @staticmethod
    def __set_default_items(prefix, default, input_shape, ni, no, kernel_size,
                            stride, padding, conv_module, act_module, bn_module,
                            dilation=1, groups=1, bias=True,
                            callback=all_to_none, conv3d_args=None):
        default['input_shape'] = input_shape
        default['callback'] = callback

        # bn, act, conv
        ScopedConvUnit.set_unit_description(default, prefix, input_shape, ni, no,
                                            kernel_size, stride, padding,
                                            conv_module, act_module, bn_module,
                                            dilation, groups, bias,
                                            callback, conv3d_args)
        # convdim
        suffix = '%d_%d_%d_%d_%d_%d_%d_%d' % (ni, no, 1, stride,
                                              0, dilation, groups, bias)
        default['convdim'] = conv_module.describe_default('%s/convdim' % prefix,
                                                          suffix, default,
                                                          input_shape, ni, no, 1,
                                                          stride, 0, dilation,
                                                          groups, bias)
        default['convdim']['type'] = conv_module if ni != no else all_to_none

        return default['conv']['output_shape']

    @staticmethod
    def __set_default_children(prefix, default, shape, ni, no, kernel_size, stride,
                               padding, conv_module, act_module, bn_module, depth,
                               dilation=1, groups=1, bias=True,
                               callback=all_to_none, conv3d_args=None):
        children = []
        for i in range(depth):
            unit_prefix = '%s/unit' % prefix
            suffix = '%d_%d_%d_%d_%d_%d_%d_%d' % (ni, no, kernel_size, stride,
                                                  padding, dilation, groups, bias)
            assert(shape[1] == ni)
            unit = ScopedConvUnit.describe_default(unit_prefix, suffix, default, shape,
                                                   ni, no, kernel_size, stride, padding,
                                                   dilation, groups, bias, act_module,
                                                   bn_module, conv_module,
                                                   callback, conv3d_args)
            shape = unit['output_shape']
            children.append(unit)

        default['children'] = children
        default['depth'] = len(children)
        return shape

    @staticmethod
    def describe_default(prefix, suffix, parent, input_shape, in_channels,
                         out_channels, kernel_size, stride, padding,
                         dilation=1, groups=1, bias=True,
                         act_module=ScopedReLU, bn_module=all_to_none,
                         conv_module=ScopedConv2d, block_depth=2,
                         callback=all_to_none, conv3d_args=None, *_, **__):
        """Create a default ScopedResBlock blueprint

        Args:
            prefix (str): Prefix from which the member scopes will be created
            suffix (str): Suffix to append the name of the scoped object
            parent (Blueprint): None or the instance of the parent blueprint
            block_depth: Number of (bn, act, conv) units in the block
            conv_module (type): CNN module to use in forward. e.g. ScopedConv2d
            bn_module (type): Batch normalization module. e.g. ScopedBatchNorm2d
            act_module (type): Activation module e.g ScopedReLU
            in_channels (int): Number of channels in the input
            out_channels (int): Number of channels produced by the block
            kernel_size (int or tuple): Size of the convolving kernel. Default: 3
            stride (int or tuple, optional): Stride for the first convolution
            padding (int or tuple, optional): Padding for the first convolution
            input_shape (tuple): (N, C_{in}, H_{in}, W_{in})
            dilation (int): Spacing between kernel elements.
            groups: Number of blocked connections from input to output channels.
            bias: Add a learnable bias if True
            callback: function to call after the output in forward is calculated
            conv3d_args: extra conv arguments to be used in self.conv and children
        """
        default = Blueprint(prefix, suffix, parent, False, ScopedResBlock)

        input_shape = ScopedResBlock.__set_default_items(prefix, default, input_shape,
                                                         in_channels, out_channels,
                                                         kernel_size, stride,
                                                         padding, conv_module, act_module,
                                                         bn_module, dilation, groups, bias,
                                                         callback, conv3d_args)

        input_shape = ScopedResBlock.__set_default_children(prefix, default, input_shape,
                                                            out_channels, out_channels,
                                                            kernel_size, 1,
                                                            padding, conv_module, ScopedTanh,
                                                            bn_module, block_depth, dilation,
                                                            groups, bias, callback, conv3d_args)
        default['output_shape'] = input_shape
        default['kwargs'] = {'blueprint': default, 'kernel_size': kernel_size,
                             'stride': stride, 'padding': padding, 'dilation': dilation,
                             'groups': groups, 'bias': bias}
        return default

    @staticmethod
    def describe_from_blueprint(prefix, suffix, blueprint, parent, depth,
                                kernel_size=None, stride=None, padding=None,
                                dilation=None, groups=None, bias=None,
                                callback=all_to_none):

        kwargs = blueprint['kwargs']
        input_shape = blueprint['input_shape']
        output_shape = blueprint['output_shape']

        def override_default(key, default):
            return kwargs[key] if default is None else default

        _kernel = override_default('kernel_size', kernel_size)
        _stride = override_default('stride', stride)
        _padding = override_default('padding', padding)
        _dilation = override_default('dilation', dilation)
        _groups = override_default('groups', groups)
        _bias = override_default('bias', bias)

        return ScopedResBlock.describe_default(prefix, suffix, parent,
                                               block_depth=depth,
                                               conv_module=blueprint['type'],
                                               bn_module=ScopedBatchNorm2d,
                                               act_module=ScopedReLU,
                                               in_channels=input_shape[1],
                                               out_channels=output_shape[1],
                                               kernel_size=_kernel,
                                               stride=_stride,
                                               padding=_padding,
                                               input_shape=input_shape,
                                               dilation=_dilation,
                                               groups=_groups,
                                               bias=_bias,
                                               callback=callback)
