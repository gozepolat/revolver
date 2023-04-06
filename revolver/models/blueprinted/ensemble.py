# -*- coding: utf-8 -*-
from revolver.modules.scoped_nn import ScopedConv2d
from revolver.meta.scope import ScopedMeta
from revolver.meta.sequential import Sequential
from revolver.meta.blueprint import Blueprint
from six import add_metaclass
import copy
import torch


@add_metaclass(ScopedMeta)
class ScopedEnsembleMean(Sequential):
    """Average ensemble of modules with the same input / output shape"""

    def __init__(self, scope, blueprint, *_, **__):
        self.scope = scope
        super(ScopedEnsembleMean, self).__init__(blueprint)

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

        suffix = '_'.join([str(s) for s in (suffix, input_shape[1],
                                            output_shape[1],
                                            kwargs['kernel_size'],
                                            kwargs['stride'],
                                            kwargs['padding'],
                                            kwargs['dilation'],
                                            kwargs['groups'],
                                            kwargs['bias'],)])
        if parent is None:
            parent = blueprint['parent']
        blueprint = copy.deepcopy(blueprint)
        blueprint['prefix'] = '%s/conv' % prefix
        blueprint.refresh_name()

        ensemble = Blueprint(prefix, suffix, parent, False, ScopedEnsembleMean)
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
        ensemble.refresh_unique_suffixes()
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
        return ScopedEnsembleMean.describe_from_blueprint(prefix, suffix,
                                                          bp, parent, 3)


@add_metaclass(ScopedMeta)
class ScopedEnsembleConcat(Sequential):
    """Concat ensemble of modules with the same input / output shape"""

    def __init__(self, scope, blueprint, *_, **__):
        super(ScopedEnsembleConcat, self).__init__(blueprint)
        self.scope = scope

    def forward(self, x):
        return self.function(x, self.container)

    @staticmethod
    def function(x, container):
        head = container[0]
        out = head(x)
        for module in container[1:]:
            o = module(x)
            out = torch.cat((out, o), dim=1)
        return out

    @staticmethod
    def describe_default(prefix='conv', suffix='', parent=None, children=None):
        assert (children is not None and len(children) > 0)
        size = len(children)
        block = children[0]
        input_shape = block['input_shape']
        output_shape = [j if i != 1 else j * size for i, j in enumerate(input_shape)]

        ensemble = Blueprint(prefix, suffix, parent, False, ScopedEnsembleConcat)
        ensemble['kwargs'] = block['kwargs'].copy()
        ensemble['kwargs']['blueprint'] = ensemble
        ensemble['kwargs']['out_channels'] = output_shape[1]
        ensemble['input_shape'] = input_shape
        ensemble['output_shape'] = output_shape
        ensemble['depth'] = len(children)

        for c in children:
            c['parent'] = ensemble

        ensemble['children'] = children
        return ensemble
