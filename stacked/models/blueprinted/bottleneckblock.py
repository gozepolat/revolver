# -*- coding: utf-8 -*-
from stacked.modules.scoped_nn import ScopedBatchNorm2d, \
    ScopedReLU, ScopedConv2d, ScopedAvgPool2d
from stacked.meta.scope import ScopedMeta
from stacked.meta.sequential import Sequential
from stacked.meta.blueprint import Blueprint, make_module
from stacked.utils.transformer import all_to_none
from stacked.models.blueprinted.conv_unit import ScopedConvUnit
from six import add_metaclass
from torch.nn.functional import dropout


@add_metaclass(ScopedMeta)
class ScopedBottleneckBlock(Sequential):
    """Pre-ResNet BottleNeck block

    Args:
        scope: Scope for the self (ScopedBottleneckBlock instance)
        blueprint: Description of inner scopes, member modules, and args
    """

    def __init__(self, scope, blueprint, *_, **__):
        super(ScopedBottleneckBlock, self).__init__(blueprint)
        self.scope = scope
        self.blueprint = blueprint

        self.depth = len(blueprint['children'])
        self.bn = make_module(blueprint['bn'])
        self.act = make_module(blueprint['act'])
        self.conv = make_module(blueprint['conv'])
        self.pool = make_module(blueprint['pool'])
        self.callback = blueprint['callback']
        self.dropout_p = blueprint['dropout_p']
        self.residual = blueprint['residual']

        self.convdim = None
        # 1x1 conv to correct the number of channels for summation
        if self.residual:
            self.convdim = make_module(blueprint['convdim'])

    def forward(self, x, *_):
        return self.function(self.bn, self.act,
                             self.conv, self.pool, self.container,
                             self.convdim, self.callback,
                             self.scope, self.dropout_p,
                             self.residual, self.training,
                             id(self), x)

    @staticmethod
    def function(bn, act, conv, pool, container, convdim,
                 callback, scope, dropout_p,
                 residual, training, module_id, x):
        o = x
        if bn is not None:
            o = bn(o)

        o = act(o)

        if dropout_p > 0:
            o = dropout(o, training=training, p=dropout_p)

        z = conv(o)

        if pool is not None:
            z = pool(z)

        for unit in container:
            z = unit(z)

        if not residual:
            pass
        elif convdim is not None:
            z = z + convdim(o)
        else:
            z = z + x

        callback(scope, module_id, z)
        return z

    @staticmethod
    def __set_pooling(prefix, default, stride):
        pool_module = all_to_none
        if stride > 1:
            pool_module = ScopedAvgPool2d
        pool_kwargs = {'kernel_size': 2, 'stride': stride}
        default['pool'] = Blueprint("%s/pool" % prefix, "2_%d" % stride,
                                    default, False, pool_module, kwargs=pool_kwargs)

        shape = default['pool']['input_shape'] = default['conv']['output_shape']
        default['pool']['output_shape'] = (shape[0], shape[1],
                                           shape[2] // stride, shape[3] // stride)

    @staticmethod
    def __set_default_items(prefix, default, input_shape, ni, no,
                            stride, conv_module, act_module, bn_module,
                            dilation=1, groups=1, bias=True,
                            callback=all_to_none, dropout_p=0.0, conv_kwargs=None,
                            bn_kwargs=None, act_kwargs=None):
        default['input_shape'] = input_shape
        default['callback'] = callback
        default['dropout_p'] = dropout_p

        # bn, act, conv
        ScopedConvUnit.set_unit_description(default, prefix, input_shape, ni, no,
                                            1, 1, 0, conv_module, act_module,
                                            bn_module, dilation, groups, bias,
                                            callback, conv_kwargs,
                                            bn_kwargs, act_kwargs)

        ScopedBottleneckBlock.__set_pooling(prefix, default, stride)

        # convdim
        suffix = '%d_%d_%d_%d_%d_%d_%d_%d' % (ni, no, 1, stride,
                                              0, dilation, groups, bias)

        default['convdim'] = conv_module.describe_default('%s/convdim' % prefix,
                                                          suffix, default,
                                                          input_shape, ni, no, 1,
                                                          stride, 0, dilation,
                                                          groups, bias)
        default['convdim']['type'] = conv_module if ni != no or stride > 1 else all_to_none

        return default['pool']['output_shape']

    @staticmethod
    def __set_default_children(prefix, default, shape, ni, no, kernel_size, stride,
                               padding, unit_module, conv_module, act_module,
                               bn_module, depth, dilation=1, groups=1, bias=True,
                               callback=all_to_none, conv_kwargs=None,
                               bn_kwargs=None, act_kwargs=None):
        children = []
        for i in range(depth-1):
            out, kernel, pad = (no, kernel_size, padding) if i == depth - 2 else (ni, 1, 0)
            unit_prefix = '%s/unit' % prefix
            suffix = '%d_%d_%d_%d_%d_%d_%d_%d' % (ni, out, kernel, stride,
                                                  pad, dilation, groups, bias)
            assert(shape[1] == ni)
            unit = unit_module.describe_default(unit_prefix, suffix, default, shape,
                                                ni, out, kernel, stride, pad,
                                                dilation, groups, bias, act_module,
                                                bn_module, conv_module,
                                                callback, conv_kwargs,
                                                bn_kwargs, act_kwargs)

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
                         conv_module=ScopedConv2d, callback=all_to_none,
                         conv_kwargs=None, bn_kwargs=None, act_kwargs=None,
                         unit_module=ScopedConvUnit, block_depth=2,
                         dropout_p=0.0, residual=True,
                         hidden_channels=0, hidden_scale=4, *_, **__):
        """Create a default ScopedBottleneckBlock blueprint

        Args:
            prefix (str): Prefix from which the member scopes will be created
            suffix (str): Suffix to append the name of the scoped object
            parent (Blueprint): None or the instance of the parent blueprint
            unit_module: basic building unit of resblock
            conv_module: CNN module to use in block_module
            bn_module: Batch normalization module. e.g. ScopedBatchNorm2d
            act_module: Activation module e.g ScopedReLU
            in_channels (int): Number of channels in the input
            out_channels (int): Number of channels produced by the block
            kernel_size (int or tuple): Size of the convolving kernel. Default: 3
            stride (int or tuple, optional): Stride for the first convolution
            padding (int or tuple, optional): Padding for the first convolution
            input_shape (tuple): (N, C_{in}, H_{in}, W_{in})
            dilation (int): Spacing between kernel elements.
            groups (int): Number of blocked connections from input to output channels.
            bias (bool): Add a learnable bias if True
            callback: function to call after the output in forward is calculated
            dropout_p (float): Probability of dropout
            residual (bool): True if a shortcut connection will be used
            conv_kwargs: extra conv arguments to be used in self.conv and children
            bn_kwargs: extra bn args, if bn module requires other than 'num_features'
            act_kwargs: extra act args, if act module requires other than defaults
            block_depth: Number of (bn, act, conv) units in the block
            hidden_channels: Number of hidden channels as the output of the first layer
            hidden_scale: Default multiplier for the hidden_channels
        """
        default = Blueprint(prefix, suffix, parent, False, ScopedBottleneckBlock)

        if hidden_channels == 0:
            hidden_channels = out_channels * hidden_scale

        input_shape = ScopedBottleneckBlock.__set_default_items(prefix, default, input_shape,
                                                                in_channels, hidden_channels,
                                                                stride, conv_module, act_module,
                                                                bn_module, dilation, groups, bias,
                                                                callback, dropout_p, conv_kwargs,
                                                                bn_kwargs, act_kwargs)

        output_shape = ScopedBottleneckBlock.__set_default_children(prefix, default, input_shape,
                                                                    hidden_channels, out_channels,
                                                                    kernel_size, 1, padding,
                                                                    unit_module, conv_module, act_module,
                                                                    bn_module, block_depth, dilation,
                                                                    groups, bias, callback, conv_kwargs,
                                                                    bn_kwargs, act_kwargs)
        default['output_shape'] = output_shape
        default['residual'] = residual
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

        return ScopedBottleneckBlock.describe_default(prefix, suffix, parent,
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
                                                      callback=callback,
                                                      hidden_channels=output_shape[1] * 4)
