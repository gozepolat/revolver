# -*- coding: utf-8 -*-
from six import string_types
from stacked.meta.scoped import generate_random_scope
from stacked.utils.transformer import all_to_none
from torch.nn import Module
import tkinter as tk
from stacked.utils import common
from logging import warning
import json


def log(log_func, msg):
    if common.DEBUG_BLUEPRINT:
        log_func(msg)


class Blueprint(dict):
    r"""Dictionary like description of a scoped module

    Args:
        prefix (str): The root scope name of the module,
                      it will be included in the scope of the submodules
        suffix(str): Suffix to append to the name, (only the main module)
        unique (bool): Indication of uniqueness, if True
                       append a random string to the scope name
        module_type (type): Type that will be used as constructor;
                            if not given, the constructor will return None
        args (iterable): Arguments that will be used in the constructor
        kwargs (dict): Key, value arguments for the constructor
        mutables (dict): (Key, Domain) for elements that can mutate
        children (iterable): Member module descriptions
        description (dict): Dictionary form of the whole description
        meta (dict): Container for information such as utility score etc.
        input_shape (tuple): (N, C_{in}, H_{in}, W_{in})
        output_shape (tuple): (N, C_{out}, H_{out}, W_{out})
    """

    def __init__(self, prefix='None', suffix='', parent=None, unique=False,
                 module_type=all_to_none, args=[],
                 children=[], description={}, kwargs={}, mutables={},
                 meta={}, freeze=False, input_shape=None, output_shape=None):
        super(Blueprint, self).__init__(description)

        # set from args if not in description
        if 'name' not in self:
            self['name'] = '%s%s' % (prefix, suffix)
        if 'args' not in self:
            self['args'] = args
        if 'kwargs' not in self:
            self['kwargs'] = kwargs
        if 'type' not in self:
            self['type'] = module_type
        if 'unique' not in self:
            self['unique'] = unique
        if 'freeze' not in self:
            self['freeze'] = freeze
        if 'prefix' not in self:
            self['prefix'] = prefix
        if 'parent' not in self:
            self['parent'] = parent
        if 'children' not in self:
            self['children'] = children
        if 'mutables' not in self:
            self['mutables'] = mutables
        if 'meta' not in self:
            self['meta'] = meta
        if 'input_shape' not in self:
            self['input_shape'] = input_shape
        if 'output_shape' not in self:
            self['output_shape'] = output_shape

        if self['unique']:
            self.make_unique()

        # gui related
        if common.BLUEPRINT_GUI:
            self.button_text = tk.StringVar()
            self.button_text_color = tk.StringVar()
            self.button_text.set(self['name'])
            self.button = None
            if self['unique']:
                self.button_text_color.set('#FFAAAA')
            else:
                self.button_text_color.set('#BBDDBB')

    def __eq__(self, other):
        return other == self.get_acyclic_dict()

    def __ne__(self, other):
        return not (self == other)

    def get_acyclic_dict(self):
        """Dictionary representation without cycles"""
        acyclic = {}
        for k, v in self.items():
            if k != 'children':
                if not issubclass(type(v), Blueprint):
                    v = common.replace_key(v, 'blueprint', 'self')
                    acyclic[k] = str(v)
                elif k != 'parent':
                    acyclic[k] = v.get_acyclic_dict()

        acyclic['parent'] = None
        if self['parent'] is not None:
            acyclic['parent'] = self['parent']['name']

        children = []
        for c in self['children']:
            if issubclass(type(c), Blueprint):
                children.append(c.get_acyclic_dict())
            else:
                for i in c:
                    children.append(i.get_acyclic_dict())

        acyclic['children'] = children
        return acyclic

    def __str__(self):
        return json.dumps(self.get_acyclic_dict(), indent=4)

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
                self.button_text_color.set('#BBDDBB')
                if self.button is not None:
                    self.button.configure(bg=self.button_text_color.get())
        if self['parent'] is not None:
            self['parent'].make_common()

    def make_unique(self):
        """Make the blueprint and all parents unique"""
        self['unique'] = True
        if '~' not in self['name']:
            self['name'] = generate_random_scope(self['name'])
            if self['parent'] is not None:
                self['parent'].make_unique()
            if common.BLUEPRINT_GUI:
                self.button_text_color.set('#FFAAAA')
                self.button_text.set(self['name'])
                if self.button is not None:
                    self.button.configure(bg=self.button_text_color.get())

    def get_element(self, index):
        if isinstance(index, string_types):
            return self[index]

        if type(index) == int:
            index = [index]
        b = self
        for i in index:
            if 'children' in b:
                b = b['children']
            # otherwise b is just an iterable
            b = b[i]

        assert(type(b) == Blueprint)
        return b

    def get_scope_button(self, master, info):
        if not common.BLUEPRINT_GUI:
            return None

        blueprint = self

        def callback():
            if blueprint['unique']:
                blueprint.make_common()
            else:
                blueprint.make_unique()

        self.button = tk.Button(master,
                                textvariable=self.button_text,
                                command=callback,
                                bg=self.button_text_color.get())

        def enter(_):
            info.delete('1.0', "end")
            info.insert("end", "{}".format(blueprint))

        self.button.bind("<Enter>", enter)
        return self.button


def make_module(blueprint):
    """Construct named (or scoped) object given the blueprint"""
    return blueprint['type'](blueprint['name'], *blueprint['args'],
                             **blueprint['kwargs'])


def visualize(blueprint):
    """Visualize module names, and toggle uniqueness"""
    master = common.GUI
    t = tk.Toplevel(master)
    t.wm_title("Info")
    info = tk.Text(t, wrap="none")
    info.tag_configure("center", justify=tk.CENTER)
    info.pack(fill="both", expand=True)
    buttons = []

    frame = master
    text = tk.Text(frame, wrap="none")
    vsb = tk.Scrollbar(orient="vertical", command=text.yview)
    text.configure(yscrollcommand=vsb.set)
    vsb.pack(side="right", fill="y")
    text.tag_configure("center", justify=tk.CENTER)
    text.pack(fill="both", expand=True)

    visit_modules(blueprint, [master, info], buttons)
    for b in buttons:
        # insert tabs before the button for hierarchical display
        text.insert("end", '\t' * b.config('text')[-1].count('/'))
        # add the button
        text.window_create("end", window=b)
        # next line
        text.insert("end", "\n")

    text.configure(state="disabled")
    master.mainloop()


def visit_modules(blueprint, main_input, outputs=[],
                  fn=lambda bp, inp, o: o.append(bp.get_scope_button(*inp))):
    """Recursively apply a function to all modules

    e.g.add named buttons to the outputs"""
    if issubclass(blueprint['type'], Module):
        fn(blueprint, main_input, outputs)

    for k, v in blueprint.items():
        if type(v) == Blueprint and k != 'parent':
            # type all_to_none -> ignore the module
            if v['type'] != all_to_none:
                visit_modules(v, main_input, outputs, fn)

    for b in blueprint['children']:
        if type(b) == Blueprint:
            visit_modules(b, main_input, outputs, fn)
        else:
            for i in b:
                visit_modules(i, main_input, outputs, fn)


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
