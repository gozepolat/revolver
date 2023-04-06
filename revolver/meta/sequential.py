# -*- coding: utf-8 -*-
from torch.nn import Module, ModuleList
from revolver.meta.blueprint import make_module


class Sequential(Module):
    """Sequential blueprinted base module"""
    def __init__(self, blueprint, *_, **__):
        super(Sequential, self).__init__()
        self.blueprint = blueprint
        self.container = None
        self.update()

    def update(self):
        blueprint = self.blueprint

        self.container = ModuleList()
        depth = blueprint['depth']
        children = blueprint['children']

        for i, bp in enumerate(children):
            self.container.append(make_module(bp))
            if (i >= depth and
                    children[-1]['output_shape'] == bp['output_shape']):
                break

    def forward(self, x):
        raise NotImplementedError("Sequential forward")
