# -*- coding: utf-8 -*-
import torch.nn.functional as F
from torch.nn import Module, ModuleList
from stacked.modules.scoped_nn import ScopedConv2d, \
    ScopedBatchNorm2d, ScopedLinear, ScopedReLU, ParameterModule
from stacked.meta.scope import ScopedMeta
from stacked.meta.sequential import Sequential
from stacked.meta.blueprint import Blueprint, make_module
from stacked.utils.transformer import all_to_none
from stacked.modules.conv import get_conv_out_shape
from six import add_metaclass


@add_metaclass(ScopedMeta)
class ScopedMetaLayer(Sequential):
    def __init__(self, scope, blueprint, *_, **__):
        self.generator = make_module(blueprint['generator'])
        self.layer_fn = make_module(blueprint['layer_fn'])
        self.layer_args = make_module(blueprint['layer_args'])

    def forward(self, x):
        pass
