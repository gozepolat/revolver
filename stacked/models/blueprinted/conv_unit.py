# -*- coding: utf-8 -*-
from torch.nn import Module
from stacked.meta.scope import ScopedMeta
from stacked.meta.blueprint import Blueprint, make_module
from stacked.modules.scoped_nn import ScopedReLU, \
    ScopedConv2d, ScopedBatchNorm2d
from six import add_metaclass


@add_metaclass(ScopedMeta)
class ScopedConvUnit(Module):
    """BN-act-conv unit

    Args:
        scope: Scope for the self (ScopedConvUnit instance)
        blueprint: Description of bn, act, and conv
        Ignores the rest of the args
    """
    def __init__(self, scope, blueprint, *_, **__):
        super(ScopedConvUnit, self).__init__()
        self.scope = scope

        self.bn = make_module(blueprint['bn'])
        self.act = make_module(blueprint['act'])
        self.conv = make_module(blueprint['conv'])

    def forward(self, x):
        return self.function(self.bn, self.act, self.conv, x)

    @staticmethod
    def function(bn, act, conv, x):
        x = act(bn(x))
        x = conv(x)
        return x

    @staticmethod
    def set_unit_description(default, prefix, input_shape, ni, no, kernel_size,
                             stride, padding, conv_module, act_module, bn_module,
                             dilation=1, groups=1, bias=True, conv3d_args=None):
        """Set descriptions for act, bn, and conv"""
        suffix = '%d_%d_%d_%d_%d_%d_%d_%d' % (ni, no, kernel_size, stride,
                                              padding, dilation, groups, bias)
        kwargs = None
        if issubclass(act_module, ScopedReLU):
            kwargs = {'inplace': True}

        default['act'] = Blueprint('%s/act' % prefix, suffix, default,
                                   False, act_module, kwargs=kwargs)

        default['bn'] = Blueprint('%s/bn' % prefix, suffix, default,
                                  False, bn_module, kwargs={'num_features': ni})

        if conv3d_args is None:
            conv3d_args = dict()
        else:
            conv3d_args = conv3d_args.copy()

        if 'in_channels' not in conv3d_args:
            conv3d_args['in_channels'] = ni
        if 'out_channels' not in conv3d_args:
            conv3d_args['out_channels'] = no
        if 'kernel_size' not in conv3d_args:
            conv3d_args['kernel_size'] = kernel_size
        if 'stride' not in conv3d_args:
            conv3d_args['stride'] = stride
        if 'padding' not in conv3d_args:
            conv3d_args['padding'] = padding
        if 'dilation' not in conv3d_args:
            conv3d_args['dilation'] = dilation
        if 'groups' not in conv3d_args:
            conv3d_args['groups'] = groups
        if 'bias' not in conv3d_args:
            conv3d_args['bias'] = bias

        default['conv'] = conv_module.describe_default('%s/conv' % prefix, suffix,
                                                       default, input_shape,
                                                       conv3d_args['in_channels'],
                                                       conv3d_args['out_channels'],
                                                       conv3d_args['kernel_size'],
                                                       conv3d_args['stride'],
                                                       conv3d_args['padding'],
                                                       conv3d_args['dilation'],
                                                       conv3d_args['groups'],
                                                       conv3d_args['bias'])

    @staticmethod
    def describe_default(prefix, suffix, parent, input_shape,
                         in_channels, out_channels, kernel_size,
                         stride, padding, dilation=1, groups=1, bias=True,
                         act_module=ScopedReLU, bn_module=ScopedBatchNorm2d,
                         conv_module=ScopedConv2d, conv_args=None, *_, **__):
        """Create a default ScopedConvUnit blueprint

        Args:
            prefix (str): Prefix from which the member scopes will be created
            suffix (str): Suffix to append the name of the scoped object
            parent (Blueprint): None or the instance of the parent blueprint
            in_channels (int): Number of channels in the input
            out_channels (int): Number of channels produced by the block
            kernel_size (int or tuple): Size of the convolving kernel. Default: 3
            stride (int or tuple, optional): Stride for the first convolution
            padding (int or tuple, optional): Padding for the first convolution
            input_shape (tuple): (N, C_{in}, H_{in}, W_{in})
            dilation: Spacing between kernel elements.
            groups: Number of blocked connections from input to output channels.
            bias: Add a learnable bias if True
            conv_module (type): CNN module to use in forward. e.g. ScopedConv2d
            bn_module (type): Batch normalization module. e.g. ScopedBatchNorm2d
            act_module (type): Activation module e.g ScopedReLU
            conv_args: extra conv arguments to be used in children
        """
        default = Blueprint(prefix, suffix, parent, False, ScopedConvUnit)
        default['input_shape'] = input_shape

        ScopedConvUnit.set_unit_description(default, prefix, input_shape,
                                            in_channels, out_channels,
                                            kernel_size, stride, padding,
                                            conv_module, act_module, bn_module,
                                            dilation, groups, bias, conv_args)

        default['output_shape'] = default['conv']['output_shape']
        default['kwargs'] = {'blueprint': default, 'kernel_size': kernel_size,
                             'stride': stride, 'padding': padding,
                             'dilation': dilation, 'groups': groups, 'bias': bias}
        return default
