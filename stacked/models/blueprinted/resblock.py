# -*- coding: utf-8 -*-
from stacked.modules.scoped_nn import ScopedBatchNorm2d, ScopedReLU
from stacked.meta.scope import ScopedMeta
from stacked.meta.sequential import Sequential
from stacked.meta.blueprint import Blueprint, make_module
from stacked.utils.transformer import all_to_none
from stacked.modules.conv import get_conv_out_shape
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
        self.depth = len(blueprint['children'])

        self.bn = make_module(blueprint['bn'])
        self.act = make_module(blueprint['act'])
        self.conv = make_module(blueprint['conv'])

        # 1x1 conv to correct the number of channels for summation
        self.convdim = make_module(blueprint['convdim'])

    def forward(self, x):
        return self.function(self.bn, self.act,
                             self.conv, self.container,
                             self.convdim, x)

    @staticmethod
    def function(bn, act, conv, container, convdim, x):
        o = act(bn(x))
        z = conv(o)

        for unit in container:
            z = unit(z)

        if convdim is not None:
            return z + convdim(o)
        else:
            return z + x

    @staticmethod
    def __set_default_items(prefix, default, input_shape, ni, no, kernel_size,
                            stride, padding, conv_module, act_module, bn_module):
        default['input_shape'] = input_shape

        # describe bn, act, conv
        ScopedConvUnit.set_unit_description(default, prefix, input_shape, ni, no,
                                            kernel_size, stride, padding,
                                            conv_module, act_module, bn_module)
        # describe convdim
        convdim_type = conv_module if ni != no else all_to_none
        kwargs = {'in_channels': ni, 'out_channels': no,
                  'kernel_size': 1, 'stride': stride, 'padding': 0}
        convdim = Blueprint('%s/convdim' % prefix,
                            '%d_%d_%d_%d_%d' % (ni, no, 1, stride, 0),
                            default, False, convdim_type, kwargs=kwargs)
        convdim['input_shape'] = input_shape
        convdim['output_shape'] = get_conv_out_shape(input_shape, no, 1, stride, 0)
        default['convdim'] = convdim
        return default['conv']['output_shape']

    @staticmethod
    def __set_default_children(prefix, default, shape, ni, no, kernel_size, stride,
                               padding, conv_module, act_module, bn_module, depth):
        children = []
        for i in range(depth):
            unit_prefix = '%s/unit' % prefix
            suffix = '%d_%d_%d_%d_%d' % (ni, no, kernel_size, stride, padding)
            unit = ScopedConvUnit.describe_default(unit_prefix, suffix, default, shape,
                                                   ni, no, kernel_size, stride, padding,
                                                   act_module, bn_module, conv_module)
            shape = unit['output_shape']
            children.append(unit)

        default['children'] = children
        return shape

    @staticmethod
    def describe_default(prefix, suffix, parent, depth, conv_module, bn_module,
                         act_module, ni, no, kernel_size, stride, padding, input_shape):
        """Create a default ScopedResBlock blueprint

        Args:
            prefix (str): Prefix from which the member scopes will be created
            suffix (str): Suffix to append the name of the scoped object
            parent (Blueprint): None or the instance of the parent blueprint
            depth: Number of (bn, act, conv) units in the block
            conv_module (type): CNN module to use in forward. e.g. ScopedConv2d
            bn_module (type): Batch normalization module. e.g. ScopedBatchNorm2d
            act_module (type): Activation module e.g ScopedReLU
            ni (int): Number of channels in the input
            no (int): Number of channels produced by the block
            kernel_size (int or tuple): Size of the convolving kernel. Default: 3
            stride (int or tuple, optional): Stride for the first convolution
            padding (int or tuple, optional): Padding for the first convolution
            input_shape (tuple): (N, C_{in}, H_{in}, W_{in})
        """
        default = Blueprint(prefix, suffix, parent, False, ScopedResBlock)

        input_shape = ScopedResBlock.__set_default_items(prefix, default, input_shape,
                                                         ni, no, kernel_size, stride,
                                                         padding, conv_module, act_module,
                                                         bn_module)
        # in_channels = no, and stride = 1 for children
        input_shape = ScopedResBlock.__set_default_children(prefix, default, input_shape,
                                                            no, no, kernel_size, 1,
                                                            padding, conv_module, act_module,
                                                            bn_module, depth)
        default['output_shape'] = input_shape
        default['kwargs'] = {'blueprint': default, 'kernel_size':kernel_size,
                             'stride': stride, 'padding': padding}
        return default

    @staticmethod
    def describe_from_blueprint(prefix, suffix, blueprint, parent, depth):
        kwargs = blueprint['kwargs']
        input_shape = blueprint['input_shape']
        output_shape = blueprint['output_shape']
        return ScopedResBlock.describe_default(prefix, suffix, parent,
                                               depth=depth,
                                               conv_module=blueprint['type'],
                                               bn_module=ScopedBatchNorm2d,
                                               act_module=ScopedReLU,
                                               ni=input_shape[1],
                                               no=output_shape[1],
                                               kernel_size=kwargs['kernel_size'],
                                               stride=kwargs['stride'],
                                               padding=kwargs['padding'],
                                               input_shape=input_shape)
