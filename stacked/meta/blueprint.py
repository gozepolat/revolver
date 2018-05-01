# -*- coding: utf-8 -*-
from six import string_types
from stacked.meta.scope import generate_random_scope
from stacked.utils.transformer import all_to_none
from torch.nn import Module
import tkinter as tk
from stacked.utils import common
from logging import warning, error
from collections import Iterable
import json
import copy


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
        self.uuid = generate_random_scope()
        # set from args if not in description
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
        if 'suffix' not in self:
            self['suffix'] = suffix
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
        if 'name' not in self:
            self.refresh_name()

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

    def refresh_name(self):
        self['name'] = '%s%s' % (self['prefix'], self['suffix'])
        if self['unique']:
            self.make_unique()

    def get_parents(self, uuids=set()):
        """List of parents until root"""
        parents = []
        if self.uuid in uuids:
            log(error, "get_parents: Blueprint %s has cycles!!"
                % self['name'])
            raw_input("continue")
            return parents

        p = self['parent']
        if p is not None:
            parents = [p] + p.get_parents(uuids | set([self.uuid]))
        return parents

    def get_acyclic_dict(self, uuids=set()):
        """Dictionary representation without cycles"""
        acyclic = {}
        if self.uuid in uuids:
            log(error, "get_acyclic_dict: Blueprint %s has cycles!!"
                % self['name'])
            raw_input("continue")
            return acyclic

        for k, v in self.items():
            if k != 'children':
                if not isinstance(v, Blueprint):
                    v = common.replace_key(v, 'blueprint', 'self')
                    acyclic[k] = str(v)
                elif k != 'parent':
                    acyclic[k] = v.get_acyclic_dict(uuids | set([self.uuid]))

        acyclic['parent'] = None
        if self['parent'] is not None:
            acyclic['parent'] = self['parent']['name']

        children = []
        for c in self['children']:
            children.append(c.get_acyclic_dict(uuids | set([self.uuid])))

        acyclic['children'] = children
        return acyclic

    def __str__(self):
        return json.dumps(self.get_acyclic_dict(), indent=4)

    def copy(self):
        """Copy self, without changing the references to the items"""
        return Blueprint(description=self)

    def clone(self):
        """Fast deep copy self, removing mutable elements"""
        description = {'children': [c.clone() for c in self['children']],
                       'parent': self['parent'],
                       'kwargs': self['kwargs'],
                       'mutables': {}}

        for k, v in self.items():
            if k not in description:
                description[k] = copy.deepcopy(v)

        return Blueprint(description=description)

    def has_unique_elements(self):
        if self['unique']:
            return True

        for k, v in self.items():
            if isinstance(v, Blueprint) and k != 'parent':
                if v.has_unique_elements():
                    return True

        for b in self['children']:
            print(b['name'])
            if b.has_unique_elements():
                return True

        return False

    def get_index_from_root(self):
        if self['parent'] is None:
            return []

        indices = self['parent'].get_index_from_root()
        index = self['parent'].get_element_index(self)
        assert(index is not None)
        indices += index

        return indices

    def get_element_index(self, element):
        for k, v in self.items():
            if element == v:
                return [k]

        for i, c in enumerate(self['children']):
            if c == element:
                return [i]

        return None

    def make_button_common(self):
        if common.BLUEPRINT_GUI:
            log(warning, "Call button common")
            self.button_text.set(self['name'])
            self.button_text_color.set('#BBDDBB')
            if self.button is not None:
                self.button.configure(bg=self.button_text_color.get())

    def make_common(self):
        """Revert back from uniqueness"""
        self['unique'] = False
        log(warning, "Calling make common")
        if self.has_unique_elements():
            log(warning, "Can't make common, %s has unique elements."
                % self['name'])
            self['unique'] = True
            return
        log(warning, "Can make common, %s has no unique elements." % self['name'])
        index = self['name'].find('~')
        if index > 0:
            self['name'] = self['name'][0:index]
            self.make_button_common()

        if self['parent'] is not None:
            self['parent'].make_common()

    def make_button_unique(self):
        if common.BLUEPRINT_GUI:
            self.button_text_color.set('#FFAAAA')
            self.button_text.set(self['name'])
            if self.button is not None:
                self.button.configure(bg=self.button_text_color.get())

    def make_unique(self):
        """Make the blueprint and all parents unique"""
        self['unique'] = True
        if '~' not in self['name']:
            self['name'] = generate_random_scope(self['name'])
            self.make_button_unique()

        if self['parent'] is not None:
            self['parent'].make_unique()

    def get_element(self, index):
        if not isinstance(index, Iterable):
            index = [index]

        b = self
        for i in index:
            if isinstance(i, string_types):
                b = b[i]
                continue
            # sugar for easier access to sub elements
            elif 'children' in b:
                b = b['children']
            b = b[i]

        return b

    def set_element(self, index, value):
        if not isinstance(index, Iterable):
            index = [index]

        last = index[-1]
        b = self.get_element(index[:-1])
        b[last] = value

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

    def collect(bp, _, out):
        out.append(bp)

    module_list = []
    visit_modules(blueprint, None, module_list, collect)
    for module in module_list:
        b = module.get_scope_button(master, info)
        # insert tabs before the button for hierarchical display
        text.insert("end", '\t' * len(module.get_parents()))
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
        if isinstance(v, Blueprint) and k != 'parent':
            # type all_to_none -> ignore the module
            if v['type'] != all_to_none:
                visit_modules(v, main_input, outputs, fn)

    for b in blueprint['children']:
        visit_modules(b, main_input, outputs, fn)

