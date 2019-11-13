# -*- coding: utf-8 -*-
from stacked.meta.scope import ScopedMeta
from stacked.meta.sequential import Sequential
from six import add_metaclass


@add_metaclass(ScopedMeta)
class ScopedTreeGroup(Sequential):
    def __init__(self, scope, blueprint, *_, **__):
        """Group of trees that have the same output shape"""
        pass

    def update(self):
        pass

    def forward(self):
        pass

    @staticmethod
    def function():
        pass

    @staticmethod
    def describe_default():
        pass

