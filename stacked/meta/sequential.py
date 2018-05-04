# -*- coding: utf-8 -*-
from torch.nn import Module, ModuleList
from stacked.meta.blueprint import make_module


class Sequential(Module):
    """Sequential blueprinted base module"""
    def __init__(self, blueprint, *_, **__):
        super(Sequential, self).__init__()
        self.container = ModuleList()
        for bp in blueprint['children']:
            self.container.append(make_module(bp))

    def forward(self, x):
        raise NotImplementedError("Sequential forward")
