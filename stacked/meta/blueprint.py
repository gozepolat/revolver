# -*- coding: utf-8 -*-
from stacked.meta.scoped import generate_random_scope
from stacked.utils.transformer import all_to_none
from torch.nn import Module
from tkinter import ttk, StringVar
from stacked.utils import common
from logging import warning


def log(log_func, msg):
    if common.DEBUG_BLUEPRINT:
        log_func(msg)


class Blueprint(dict):
    r"""Dictionary like description of a scoped module

    Args:
        prefix (str): The root scope name of the module
        unique (bool): Indication of uniqueness, if True
                       append a random string to the scope name
        module_type (type): Type that will be used as constructor;
                            if not given, the constructor will return None
        args (iterable): Arguments that will be used in the constructor
        children (iterable): Member module descriptions
        description (dict): Dictionary form of the whole description
    """

    def __init__(self, prefix='None', parent=None, unique=False,
                 module_type=all_to_none, args=[],
                 children=[], description={}, kwargs={}):
        super(Blueprint, self).__init__(description)

        # set from args if not in description
        if 'name' not in self:
            self['name'] = prefix
        if 'args' not in self:
            self['args'] = args
        if 'kwargs' not in self:
            self['kwargs'] = kwargs
        if 'type' not in self:
            self['type'] = module_type
        if 'unique' not in self:
            self['unique'] = unique
        if 'prefix' not in self:
            self['prefix'] = prefix
        if 'parent' not in self:
            self['parent'] = parent
        if 'children' not in self:
            self['children'] = children

        if self['unique']:
            self.make_unique()

        if common.BLUEPRINT_GUI:
            # gui related
            self.button_text = StringVar()
            self.button_text.set(self['name'])

    def has_unique_elements(self):
        if self['unique']:
            return True

        for k, v in self.items():
            if type(v) == Blueprint and k != 'parent':
                if v.has_unique_elements():
                    return True

        for b in self['children']:
            if type(b) == Blueprint:
                if b.has_unique_elements():
                    return True
            else:
                for i in b:
                    if i.has_unique_elements():
                        return True
        return False

    def make_common(self):
        """Revert back from uniqueness"""
        self['unique'] = False
        if self.has_unique_elements():
            log(warning, "Can't make common, %s has unique elements." % self['name'])
            self['unique'] = True
            return

        index = self['name'].find('~')
        if index > 0:
            self['name'] = self['name'][0:index]
            if common.BLUEPRINT_GUI:
                self.button_text.set(self['name'])

    def make_unique(self):
        """Make the blueprint and all parents unique"""
        self['unique'] = True
        if '~' not in self['name']:
            self['name'] = generate_random_scope(self['prefix'])
            if common.BLUEPRINT_GUI:
                self.button_text.set(self['name'])

        if self['parent'] is not None:
            self['parent'].make_unique()

    def get_child(self, index):
        b = self['children']
        if type(index) == int:
            b = b[index]
            if type(b) == Blueprint:
                return b
        else:
            for i in index:
                b = b[i]
            assert(type(b) == Blueprint)
            return b

    def get_scope_button(self, master):
        blueprint = self

        def callback():
            if blueprint['unique']:
                blueprint.make_common()
            else:
                blueprint.make_unique()

        return ttk.Button(master, textvariable=self.button_text, command=callback)


def make_module(blueprint):
    r"""Construct named (or scoped) object given the blueprint"""
    return blueprint['type'](blueprint['name'], *blueprint['args'],
                             **blueprint['kwargs'])


def summarize(blueprint):
    r"""Print the names of objects in the blueprint"""
    print(blueprint['name'])
    for b in blueprint['children']:
        if type(b) == Blueprint:
            summarize(b)
        else:
            for i in b:
                summarize(i)


def visualize(blueprint):
    master = common.GUI
    buttons = []
    visit_modules(blueprint, master, buttons)
    for b in buttons:
        b.pack()

    master.mainloop()


def visit_modules(blueprint, master, elements=[],
                  fn=lambda x, k: x.get_scope_button(k)):
    r"""Recursively add named buttons to the module set"""
    if issubclass(blueprint['type'], Module):
        elements.append(fn(blueprint, master))

    for b in blueprint['children']:
        if type(b) == Blueprint:
            visit_modules(b, master, elements)
        else:
            for i in b:
                visit_modules(i, master, elements)


def get_module_names(blueprint, module_set=set()):
    r"""Recursively add names (or scopes) to the module set"""
    if issubclass(blueprint['type'], Module):
        module_set.add(blueprint['name'])
    for b in blueprint['children']:
        if type(b) == Blueprint:
            get_module_names(b, module_set)
        else:
            for i in b:
                get_module_names(i, module_set)
