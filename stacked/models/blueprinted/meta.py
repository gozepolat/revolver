# -*- coding: utf-8 -*-
import torch
from torch.nn import Module, Parameter
from torch.nn.init import kaiming_normal
from stacked.modules.scoped_nn import ScopedConv2d, \
    ScopedReLU, ScopedConv3d2d, ScopedTanh, ScopedSigmoid
from stacked.meta.scope import ScopedMeta
from stacked.models.blueprinted.resblock import ScopedResBlock
from stacked.models.blueprinted.conv_unit import ScopedConvUnit
from stacked.meta.blueprint import Blueprint, make_module
from stacked.utils.transformer import get_cuda, all_to_none
from stacked.modules.fakes import MaskMultiplied, \
    MaskScalarMultipliedSummed
from six import add_metaclass


@add_metaclass(ScopedMeta)
class PreConvMask(Module):
    def __init__(self, scope, blueprint, *_, **__):
        super(PreConvMask, self).__init__()
        self.scope = scope
        self.blueprint = blueprint

        self.scalar = Parameter(torch.FloatTensor([blueprint['scalar']]),
                                requires_grad=True)
        self.function = make_module(blueprint['function'])
        self.act = make_module(blueprint['act'])

    def forward(self, module_out, mask):
        scalar = self.act(self.scalar)
        return self.function(module_out, mask,
                             scalar.expand_as(module_out))

    @staticmethod
    def describe_default(prefix='pre_conv', suffix='', parent=None,
                         scalar=0.01, act_module=ScopedSigmoid,
                         function_module=MaskScalarMultipliedSummed):
        bp = Blueprint(prefix, suffix, parent, True, PreConvMask)
        bp['scalar'] = scalar
        bp['act'] = Blueprint('%s/act' % prefix, suffix, parent, False, act_module)
        bp['function'] = Blueprint('%s/function' % prefix, suffix, bp, False,
                                   function_module)
        bp['kwargs'] = {'blueprint': bp}
        return bp


@add_metaclass(ScopedMeta)
class ScopedMetaMaskGenerator(Module):
    def __init__(self, scope, blueprint, *_, **__):
        super(ScopedMetaMaskGenerator, self).__init__()
        self.scope = scope
        self.blueprint = blueprint

        self.conv = make_module(blueprint['conv'])
        self.mask = get_cuda(
            kaiming_normal(torch.ones(blueprint['input_shape'][1:])))
        self.pre_conv = make_module(blueprint['pre_conv'])
        self.callback = blueprint['callback']

    def forward(self, x):
        mask = self.function(x, self.conv, self.pre_conv, self.mask)
        self.mask = torch.mean(mask, dim=0).data
        self.mask += self.mask.data.new(self.mask.size()).normal_(0, 0.01)

        self.callback(self.scope, id(self), mask)
        return mask

    @staticmethod
    def function(x, conv, pre_conv, mask):
        return conv(pre_conv(x, mask.expand_as(x)))

    @staticmethod
    def describe_default(prefix='gen', suffix='', parent=None,
                         shape=None, in_channels=3, out_channels=3,
                         kernel_size=3, stride=1,
                         padding=1, dilation=1, groups=1,
                         bias=True, bn_module=all_to_none,
                         act_module=ScopedReLU,
                         conv_module=ScopedConv3d2d,
                         gen_module=ScopedResBlock,
                         pre_conv=PreConvMask,
                         callback=all_to_none,
                         conv3d_args=None, **__):
        bp = Blueprint(prefix, suffix, parent, True, ScopedMetaMaskGenerator)

        depth = 2
        assert (in_channels == out_channels)
        bp['conv'] = gen_module.describe_default(prefix='%s/conv' % prefix,
                                                 suffix=suffix, parent=bp,
                                                 input_shape=shape,
                                                 in_channels=in_channels,
                                                 out_channels=out_channels,
                                                 kernel_size=kernel_size,
                                                 stride=stride, padding=padding,
                                                 dilation=dilation, groups=groups,
                                                 bias=bias, act_module=act_module,
                                                 bn_module=bn_module,
                                                 conv_module=conv_module, depth=depth,
                                                 callback=callback,
                                                 conv3d_args=conv3d_args)
        bp['callback'] = callback
        bp['pre_conv'] = pre_conv.describe_default(prefix='pre_conv', suffix='',
                                                   parent=bp, scalar=0.2)
        bp['input_shape'] = shape
        bp['output_shape'] = shape
        assert (bp['conv']['output_shape'] == shape)
        bp['kwargs'] = {'blueprint': bp}
        return bp


@add_metaclass(ScopedMeta)
class ScopedMetaMasked(Module):
    def __init__(self, scope, blueprint, *_, **__):
        super(ScopedMetaMasked, self).__init__()
        self.scope = scope
        self.blueprint = blueprint

        self.generator = make_module(blueprint['generator'])
        self.conv = make_module(blueprint['conv'])
        self.convdim = make_module(blueprint['convdim'])
        self.mask_fn = make_module(blueprint['mask_fn'])
        self.callback = blueprint['callback']

    def forward(self, x):
        return self.function(self.mask_fn,
                             self.generator, self.conv,
                             self.convdim, self.callback,
                             self.scope, id(self), x)

    @staticmethod
    def function(mask_fn, generator, conv, convdim, callback,
                 scope, module_id, x):
        out = conv(x)
        out = mask_fn(out, generator(out), x)
        out = convdim(out)
        callback(scope, module_id, out)
        return out

    @staticmethod
    def describe_default(prefix='meta_layer', suffix='', parent=None,
                         shape=None, in_channels=3, out_channels=3,
                         kernel_size=3, stride=1, padding=1,
                         dilation=1, groups=1, bias=True,
                         conv_module=ScopedConv2d,
                         generator=ScopedMetaMaskGenerator,
                         mask_fn=MaskMultiplied,
                         gen_bn_module=all_to_none,
                         gen_act_module=ScopedTanh,
                         gen_conv=ScopedConv2d,
                         gen_module=ScopedConvUnit,
                         gen_in_channels=32, gen_out_channels=32,
                         gen_kernel_size=9, gen_stride=1,
                         gen_dilation=1, gen_groups=1, gen_bias=True,
                         gen_pre_conv=PreConvMask,
                         callback=all_to_none, conv3d_args=None,
                         depthwise=True,
                         **__):
        """Meta masks to model local rules"""
        kwargs = {'in_channels': in_channels,
                  'out_channels': out_channels,
                  'kernel_size': kernel_size, 'stride': stride,
                  'padding': padding, 'dilation': dilation,
                  'groups': groups, 'bias': bias}

        suffix = "%s_%d_%d_%d_%d_%d_%d_%d_%d" % (suffix, in_channels,
                                                 out_channels,
                                                 kernel_size, stride,
                                                 kernel_size, dilation,
                                                 groups, bias)

        bp = Blueprint(prefix, suffix, parent, True,
                       ScopedMetaMasked, kwargs=kwargs)

        bp['mask_fn'] = Blueprint('%s/mask_fn' % prefix, suffix, bp,
                                  False, mask_fn)
        bp['callback'] = callback

        # groups = in_channels for depth-wise filtering
        filtering_groups = in_channels
        if out_channels % in_channels != 0:
            filtering_groups = groups

        bp['conv'] = conv_module.describe_default('%s/conv' % prefix, suffix,
                                                  bp, shape, in_channels,
                                                  out_channels, kernel_size,
                                                  stride, padding, dilation,
                                                  filtering_groups, bias)

        out_shape = bp['conv']['output_shape']
        bp['kwargs'] = {'blueprint': bp, 'kernel_size': kernel_size,
                        'stride': stride, 'padding': padding,
                        'dilation': dilation, 'groups': groups, 'bias': bias}

        # pointwise 1x1 conv
        bp['convdim'] = conv_module.describe_default('%s/convdim' % prefix, suffix,
                                                     bp, out_shape, out_channels,
                                                     out_channels, 1,
                                                     1, 0, dilation,
                                                     groups, bias)

        # in case the generator uses conv3d adjust conv3d_arguments accordingly
        kwargs = {'in_channels': gen_in_channels, 'out_channels': gen_out_channels,
                  'kernel_size': gen_kernel_size, 'stride': gen_stride,
                  'padding': gen_kernel_size // 2, 'dilation': gen_dilation,
                  'groups': gen_groups, 'bias': gen_bias}

        conv3d_args = ScopedConv3d2d.adjust_args(conv3d_args, gen_conv, **kwargs)

        if depthwise:
            groups = min(gen_out_channels, out_channels)

        bp['generator'] = generator.describe_default('%s/gen' % prefix, suffix,
                                                     bp, out_shape, out_channels,
                                                     out_channels, kernel_size,
                                                     1, padding, dilation,
                                                     groups, bias, gen_bn_module,
                                                     gen_act_module, gen_conv,
                                                     gen_module, gen_pre_conv,
                                                     callback=callback,
                                                     conv3d_args=conv3d_args)
        assert (shape is not None)
        bp['input_shape'] = shape
        bp['output_shape'] = out_shape

        return bp

    @staticmethod
    def describe_from_blueprint(prefix='meta_layer', suffix='',
                                blueprint=None, parent=None,
                                generator=ScopedMetaMaskGenerator,
                                mask_fn=MaskMultiplied,
                                gen_bn_module=all_to_none,
                                gen_act_module=ScopedReLU,
                                gen_conv=ScopedConv3d2d,
                                gen_module=ScopedResBlock,
                                gen_in_channels=16, gen_out_channels=16,
                                gen_kernel_size=9, gen_stride=1,
                                gen_dilation=1, gen_groups=1, gen_bias=True,
                                gen_pre_conv=PreConvMask, **__):
        input_shape = blueprint['input_shape']
        output_shape = blueprint['output_shape']
        kwargs = blueprint['kwargs']
        if parent is None:
            parent = blueprint['parent']
        bp = ScopedMetaMasked.describe_default(prefix, suffix, parent,
                                               input_shape, input_shape[1],
                                               output_shape[1],
                                               kwargs['kernel_size'],
                                               kwargs['stride'],
                                               kwargs['padding'],
                                               kwargs['dilation'],
                                               kwargs['groups'],
                                               kwargs['bias'],
                                               blueprint['type'],
                                               generator,
                                               mask_fn,
                                               gen_bn_module,
                                               gen_act_module,
                                               gen_conv,
                                               gen_module,
                                               gen_in_channels,
                                               gen_out_channels,
                                               gen_kernel_size,
                                               gen_stride,
                                               gen_dilation,
                                               gen_groups, gen_bias,
                                               gen_pre_conv)
        return bp