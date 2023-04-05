# -*- coding: utf-8 -*-
from stacked.modules.scoped_nn import ScopedBatchNorm2d, \
    ScopedReLU, ScopedConv2d
from stacked.meta.scope import ScopedMeta
from stacked.meta.sequential import Sequential
from stacked.meta.blueprint import Blueprint, make_module
from stacked.utils.transformer import all_to_none
from stacked.models.blueprinted.convunit import ScopedConvUnit, is_conv_simple
from stacked.models.blueprinted.unit import set_convdim
from six import add_metaclass
from torch.nn.functional import dropout


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

        self.depth = None
        self.bn = None
        self.act = None
        self.conv = None
        self.convdim = None
        self.callback = None
        self.dropout_p = None
        self.residual = None
        self.update(True)

    def update(self, init=False):
        if not init:
            super(ScopedResBlock, self).update()
        blueprint = self.blueprint
        self.depth = len(blueprint['children'])

        if 'bn' in blueprint:
            self.bn = make_module(blueprint['bn'])
        if 'act' in blueprint:
            self.act = make_module(blueprint['act'])
        if 'conv' in blueprint:
            self.conv = make_module(blueprint['conv'])

        self.callback = blueprint['callback']
        self.dropout_p = blueprint['dropout_p']
        self.residual = blueprint['residual']

        # 1x1 conv to correct the number of channels for summation
        if self.residual:
            self.convdim = make_module(blueprint['convdim'])

    def forward(self, x, *_):
        return self.function(self.bn, self.act,
                             self.conv, self.container,
                             self.convdim, self.callback,
                             self.scope, self.dropout_p,
                             self.residual, self.training,
                             id(self), x)

    @staticmethod
    def function(bn, act, conv, container, convdim,
                 callback, scope, dropout_p,
                 residual, training, module_id, x):
        o = x
        if bn is not None:
            o = bn(o)

        if act is not None:
            o = act(o)

        if dropout_p > 0:
            o = dropout(o, training=training, p=dropout_p)

        z = conv(o)

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
    def __set_default_items(prefix, default, input_shape, ni, no, kernel_size,
                            stride, padding, conv_module, act_module, bn_module,
                            dilation=1, groups=1, bias=True,
                            callback=all_to_none, dropout_p=0.0, conv_kwargs=None,
                            bn_kwargs=None, act_kwargs=None, residual=False):
        default['input_shape'] = input_shape
        default['callback'] = callback
        default['dropout_p'] = dropout_p
        assert(ni == input_shape[1])

        if is_conv_simple(conv_module):
            module_order = ["bn", "act", "conv"]
        else:
            # conv corresponds to a complex component where bn, act are inside
            module_order = ["conv"]

        ScopedConvUnit.set_unit_description(default, prefix, input_shape, ni, no,
                                            kernel_size, stride, padding,
                                            conv_module, act_module, bn_module,
                                            dilation, groups, bias,
                                            callback, conv_kwargs,
                                            bn_kwargs, act_kwargs, dropout_p,
                                            module_order=module_order)

        set_convdim(default, prefix, input_shape, ni, no, stride, dilation, groups,
                    bias, conv_module, residual)

        return default['conv']['output_shape']

    @staticmethod
    def __set_children(prefix, default, shape, ni, no, kernel_size, stride,
                       padding, unit_module, conv_module, act_module,
                       bn_module, depth, dilation=1, groups=1, bias=True,
                       callback=all_to_none, conv_kwargs=None,
                       bn_kwargs=None, act_kwargs=None, dropout_p=0.0):
        children = []
        for i in range(depth - 1):
            unit_prefix = '%s/unit' % prefix
            suffix = '_'.join([str(s) for s in (ni, no, kernel_size, stride,
                                                padding, dilation, groups, bias)])
            assert(shape[1] == ni)
            unit = unit_module.describe_default(unit_prefix, suffix, default, shape,
                                                ni, no, kernel_size, stride, padding,
                                                dilation, groups, bias, act_module,
                                                bn_module, conv_module,
                                                callback, conv_kwargs,
                                                bn_kwargs, act_kwargs, dropout_p,
                                                module_order=["bn", "act", "conv"])
            shape = unit['output_shape']
            children.append(unit)
            stride = 1

        default['children'] = children
        default['depth'] = len(children)
        return shape

    @staticmethod
    def describe_default(prefix, suffix, parent, input_shape, in_channels,
                         out_channels, kernel_size, stride, padding,
                         dilation=1, groups=1, bias=True,
                         act_module=ScopedReLU, bn_module=ScopedBatchNorm2d,
                         conv_module=ScopedConv2d, callback=all_to_none,
                         conv_kwargs=None, bn_kwargs=None, act_kwargs=None,
                         unit_module=ScopedConvUnit, block_depth=2,
                         dropout_p=0.0, residual=True, mutation_p=0.8,
                         toggle_p=0.02, *_, **__):
        """Create a default ScopedResBlock blueprint

        Args:
            prefix (str): Prefix from which the member scopes will be created
            suffix (str): Suffix to append the name of the scoped object
            parent (Blueprint): None or the instance of the parent blueprint
            unit_module (type): basic building unit of resblock
            conv_module (type): CNN module to use in block_module
            bn_module: Batch normalization module. e.g. ScopedBatchNorm2d
            act_module (type): Activation module e.g ScopedReLU
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
        """
        default = Blueprint(prefix, suffix, parent, False, ScopedResBlock)

        if conv_module == ScopedResBlock:
            conv_module = ScopedConv2d

        input_shape = ScopedResBlock.__set_default_items(prefix, default, input_shape,
                                                         in_channels, out_channels,
                                                         kernel_size, stride,
                                                         padding, conv_module, act_module,
                                                         bn_module, dilation, groups, bias,
                                                         callback, dropout_p, conv_kwargs,
                                                         bn_kwargs, act_kwargs, residual)

        input_shape = ScopedResBlock.__set_children(prefix, default, input_shape,
                                                    out_channels, out_channels,
                                                    kernel_size, 1, padding,
                                                    unit_module, conv_module, act_module,
                                                    bn_module, block_depth, dilation,
                                                    groups, bias, callback, conv_kwargs,
                                                    bn_kwargs, act_kwargs, dropout_p)
        default['output_shape'] = input_shape
        default['residual'] = residual
        default['kwargs'] = {'blueprint': default, 'kernel_size': kernel_size,
                             'stride': stride, 'padding': padding, 'dilation': dilation,
                             'groups': groups, 'bias': bias}
        default['mutation_p'] = mutation_p
        default['toggle_p'] = toggle_p
        return default

    @staticmethod
    def describe_from_blueprint(prefix, suffix, blueprint, parent, block_depth,
                                kernel_size=None, stride=None, padding=None,
                                dilation=None, groups=None, bias=None,
                                act_module=ScopedReLU,
                                bn_module=ScopedBatchNorm2d,
                                callback=all_to_none, conv_kwargs=None,
                                bn_kwargs=None, act_kwargs=None,
                                unit_module=ScopedConvUnit,
                                dropout_p=0.0, residual=True,
                                mutation_p=0.8, toggle_p=0.02):

        kwargs = blueprint['kwargs']
        input_shape = blueprint['input_shape']
        output_shape = blueprint['output_shape']

        args = ScopedConv2d.adjust_args(kwargs, kernel_size, stride,
                                        padding, dilation, groups, bias)

        _kernel, _stride, _padding, _dilation, _groups, _bias = args
        return ScopedResBlock.describe_default(prefix, suffix, parent,
                                               block_depth=block_depth,
                                               conv_module=blueprint['type'],
                                               bn_module=bn_module,
                                               act_module=act_module,
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
                                               conv_kwargs=conv_kwargs,
                                               bn_kwargs=bn_kwargs,
                                               act_kwargs=act_kwargs,
                                               unit_module=unit_module,
                                               dropout_p=dropout_p,
                                               residual=residual,
                                               mutation_p=mutation_p,
                                               toggle_p=toggle_p)
