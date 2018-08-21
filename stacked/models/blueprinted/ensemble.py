# -*- coding: utf-8 -*-
from torch.nn import ModuleList
from stacked.modules.scoped_nn import ScopedConv2d
from stacked.meta.scope import ScopedMeta
from stacked.meta.sequential import Sequential
from stacked.meta.blueprint import Blueprint, make_module
from six import add_metaclass
import copy


@add_metaclass(ScopedMeta)
class ScopedEnsemble(Sequential):
    """Average ensemble of modules with the same input / output shape"""

    def __init__(self, scope, blueprint, *_, **__):
        self.scope = scope
        super(ScopedEnsemble, self).__init__(blueprint)

    def forward(self, x):
        return self.function(x, self.container)

    @staticmethod
    def function(x, container):
        out = 0.0
        for module in container:
            out = module(x) + out
        out = out / len(container)
        return out

    @staticmethod
    def describe_from_blueprint(prefix, suffix, blueprint,
                                parent=None, ensemble_size=3):

        input_shape = blueprint['input_shape']
        output_shape = blueprint['output_shape']
        kwargs = blueprint['kwargs']

        suffix = "%s_%d_%d_%d_%d_%d_%d_%d_%d" % (suffix, input_shape[1],
                                                 output_shape[1],
                                                 kwargs['kernel_size'],
                                                 kwargs['stride'],
                                                 kwargs['padding'],
                                                 kwargs['dilation'],
                                                 kwargs['groups'],
                                                 kwargs['bias'],)
        if parent is None:
            parent = blueprint['parent']
        blueprint = copy.deepcopy(blueprint)
        blueprint['prefix'] = '%s/conv' % prefix
        blueprint.refresh_name()

        ensemble = Blueprint(prefix, suffix, parent, False, ScopedEnsemble)
        blueprint['parent'] = ensemble

        ensemble['children'] = [copy.deepcopy(blueprint) for _ in range(ensemble_size)]
        ensemble['depth'] = len(ensemble['children'])

        ensemble['kwargs'] = {'blueprint': ensemble,
                              'kernel_size': kwargs['kernel_size'],
                              'stride': kwargs['stride'],
                              'padding': kwargs['padding'],
                              'dilation': kwargs['dilation'],
                              'groups': kwargs['groups'],
                              'bias': kwargs['bias']}
        ensemble['input_shape'] = input_shape
        ensemble['output_shape'] = output_shape
        return ensemble

    @staticmethod
    def describe_default(prefix='conv', suffix='', parent=None,
                         shape=None, in_channels=3, out_channels=3,
                         kernel_size=3, stride=1, padding=1,
                         dilation=1, groups=1, bias=True,
                         conv_module=ScopedConv2d,
                         **__):
        input_shape = shape
        bp = conv_module.describe_default(prefix,
                                          suffix,
                                          parent, input_shape,
                                          in_channels,
                                          out_channels,
                                          kernel_size,
                                          stride,
                                          padding,
                                          dilation,
                                          groups,
                                          bias)
        return ScopedEnsemble.describe_from_blueprint(prefix, suffix,
                                                      bp, parent, 3)
