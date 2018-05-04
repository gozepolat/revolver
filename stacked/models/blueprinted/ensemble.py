# -*- coding: utf-8 -*-
from torch.nn import ModuleList
from stacked.modules.scoped_nn import ParameterModule
from stacked.meta.scope import ScopedMeta
from stacked.meta.sequential import Sequential
from stacked.meta.blueprint import Blueprint, make_module
from six import add_metaclass


@add_metaclass(ScopedMeta)
class ScopedEnsemble(Sequential):
    """Ensemble of modules with the same input / output shape"""
    def __init__(self, scope, blueprint, *_, **__):
        self.scope = scope
        super(ScopedEnsemble, self).__init__(blueprint)

        self.masks = ModuleList()
        self.mask_fn = make_module(blueprint['mask_fn'])

        masks = blueprint['masks']['children']
        for mask in masks:
            self.masks.append(make_module(mask))

    def forward(self, x):
        return self.function(x, self.container,
                             self.masks, self.mask_fn)

    @staticmethod
    def function(x, container, masks, mask_fn):
        out = 0.0
        for module, mask in zip(container, masks):
            out = mask_fn(out, module(x), mask)
        return out

    @staticmethod
    def describe_from_blueprint(prefix, suffix, blueprint,
                                parent=None, ensemble_size=3):

        input_shape = blueprint['input_shape']
        output_shape = blueprint['output_shape']
        kwargs = blueprint['kwargs']

        prefix = "%s/%s" % (blueprint['prefix'], prefix)
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
        blueprint = blueprint.clone()
        blueprint['prefix'] = '%s/conv' % prefix
        blueprint.refresh_name()

        class Mask:
            def __init__(self, *_, **__):
                return

            def __call__(self, out, module_out, mask):
                return out + module_out * mask()

        ensemble = Blueprint(prefix, suffix, parent, False, ScopedEnsemble)
        blueprint['parent'] = ensemble
        ensemble['mask_fn'] = Blueprint('%s/mask_fn' % prefix, suffix, ensemble,
                                        False, Mask)
        ensemble['masks'] = Blueprint('%s/masks' % prefix, suffix, ensemble,
                                      False, ModuleList)
        masks = ensemble['masks']
        masks['children'] = [Blueprint('%s/masks/mask' % prefix, suffix,
                                       masks, True, ParameterModule,
                                       kwargs={'value': 1.0 / ensemble_size,
                                               'size': output_shape})
                             for _ in range(ensemble_size)]

        ensemble['children'] = [blueprint.clone() for _ in range(ensemble_size)]
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
