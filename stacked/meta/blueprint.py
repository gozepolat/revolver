# -*- coding: utf-8 -*-
import torch
from six import string_types
from six.moves import cPickle as pickle
from stacked.meta.scope import generate_random_scope, get_common_scope_name_length, UNIQUE_SUFFIX_DELIMITER
from stacked.utils.transformer import all_to_none
from torch.nn import Module, BatchNorm2d
import tkinter as tk
from stacked.utils import common
from logging import warning, error, exception
from collections.abc import Iterable
from collections import defaultdict
import numpy as np
import json
import copy
import inspect


def log(log_func, msg):
    if common.DEBUG_BLUEPRINT:
        log_func("stacked.meta.blueprint: %s" % msg)


def with_gui():
    return common.BLUEPRINT_GUI and (common.GUI is not None)


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
                 module_type=all_to_none, args=None,
                 children=None, description=None, kwargs=None, mutables=None,
                 meta=None, freeze=False, input_shape=None, output_shape=None):
        if description is None:
            description = {}
        super(Blueprint, self).__init__(description)

        # set from args if not in description
        if 'args' not in self:
            if args is None:
                args = []
            self['args'] = args
        if 'kwargs' not in self:
            if kwargs is None:
                kwargs = {}
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
            self['parent'] = parent if id(parent) != id(self) else None
        if 'children' not in self:
            if children is None:
                children = []
            self['children'] = children
        if 'mutables' not in self:
            if mutables is None:
                mutables = {}
            self['mutables'] = mutables
        if 'meta' not in self:
            if meta is None:
                meta = {}
            self['meta'] = meta
        if 'input_shape' not in self:
            self['input_shape'] = input_shape
        if 'output_shape' not in self:
            self['output_shape'] = output_shape
        if 'name' not in self:
            self.refresh_name(refresh_unique_suffixes=False)

        # gui related
        if with_gui():
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

    def add_gui_elements(self):
        self.button_text = tk.StringVar()
        self.button_text_color = tk.StringVar()
        self.button_text.set(self['name'])
        self.button = None
        if self['unique']:
            self.button_text_color.set('#FFAAAA')
        else:
            self.button_text_color.set('#BBDDBB')

        for bp in collect_modules(self):
            if bp['name'] != self['name']:
                bp.add_gui_elements()

    def refresh_name(self, refresh_unique_suffixes=True):
        self.reset_name_without_unique_suffix()
        if self['unique']:
            self.make_unique(refresh_unique_suffixes)

    def get_parents(self, id_set=None):
        """List of parents until root"""
        if id_set is None:
            id_set = set()

        parents = []
        if id(self) in id_set:
            log(error, "get_parents: Blueprint %s has cycles!!"
                % self['name'])
            return parents

        p = self['parent']
        if p is not None:
            parents = [p] + p.get_parents(id_set | {id(self)})
        return parents

    @staticmethod
    def load_pickle(filename):
        """Return a blueprint loaded from a pickle file"""
        with open(filename, 'rb') as f:
            blueprint = pickle.load(f)
        return blueprint

    def dump_pickle(self, filename=None):
        """Save the original blueprint to a pickle file"""
        if filename is None:
            filename = "%s.pkl" % self['name']

        with open(filename, 'wb') as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

    def dump_json(self, filename=None):
        """Save the blueprint as acyclic dictionary to a json file"""
        if filename is None:
            filename = "%s.json" % self['name']

        with open(filename, 'w') as f:
            json.dump(self, f)

    def detect_cycles(self, id_set):
        """After mixing many blueprints """
        if id_set is None:
            id_set = set()

        if id(self) in id_set:
            raise ValueError("Blueprint has cycles")

        for k, v in self.items():
            if k != "parent" and hasattr(v, 'detect_cycles'):
                v.detect_cycles(id_set | {id(self)})

        for c in self['children']:
            c.detect_cycles(id_set | {id(self)})

    def get_acyclic_dict(self, id_set=None):
        """Dictionary representation without cycles"""
        if id_set is None:
            id_set = set()

        acyclic = {}
        if id(self) in id_set:
            log(error, "get_acyclic_dict: Blueprint %s has cycles!!"
                % self['name'])
            return acyclic

        for k, v in self.items():
            if k != 'children' and k != 'bns':
                if not isinstance(v, Blueprint):
                    v = common.replace_key(v, 'blueprint', 'self')
                    acyclic[k] = str(v)
                elif k != 'parent':
                    acyclic[k] = v.get_acyclic_dict(id_set | {id(self)})

        acyclic['parent'] = None
        if self['parent'] is not None:
            acyclic['parent'] = self['parent']['name']

        children = []
        for c in self['children']:
            children.append(c.get_acyclic_dict(id_set | {id(self)}))
        acyclic['children'] = children

        if 'bns' in self:
            bns = []
            for bn in self['bns']:
                bns.append(bn.get_acyclic_dict(id_set | {id(self)}))
            acyclic['bns'] = bns

        return acyclic

    def __str__(self):
        return json.dumps(self.get_acyclic_dict(), indent=4)

    def copy(self):
        """Copy self, without changing the references to the items"""
        bp = Blueprint(description=self)
        if 'blueprint' in bp['kwargs']:
            bp['kwargs']['blueprint'] = bp
        return copy

    def __deepcopy__(self, memo=None):
        # log(warning, f"Deep copying {self['name']}")
        if memo is None:
            memo = {}

        if id(self) in memo:
            return memo[id(self)]

        copied = Blueprint(self['prefix'], self['suffix'])
        memo[id(self)] = copied

        for (k, v) in self.items():
            k_id = id(self[k])

            if k_id in memo:
                copied[k] = memo[k_id]
                continue

            if k == 'mutables':
                copied[k] = {}
            elif k == 'kwargs':
                copied[k] = {key: value for key, value in self[k].items() if key != 'blueprint'}
                if 'blueprint' in self[k]:
                    copied[k]['blueprint'] = copied
            elif k == 'parent':
                # Handle it outside, e.g. deepcopy children, and assign the higher level obj as parent
                copied[k] = None
            elif k == 'children':
                copied[k] = [None] * len(self[k])
                for i, c in enumerate(self[k]):
                    copied[k][i] = copy.deepcopy(c, memo)
                    # Handled here
                    copied[k][i]["parent"] = copied
            else:
                copied[k] = copy.deepcopy(v, memo)
                if isinstance(copied[k], Blueprint):
                    # Handle here
                    copied[k]["parent"] = copied

            memo[k_id] = copied[k]

        return copied

    def has_unique_elements(self):
        if self['unique']:
            return True

        for k, v in self.items():
            if isinstance(v, Blueprint) and k != 'parent':
                if v.has_unique_elements():
                    return True

        for b in self['children']:
            if b.has_unique_elements():
                return True

        return False

    def get_index_from_root(self):
        if self['parent'] is None:
            return []

        indices = self['parent'].get_index_from_root()
        index = self['parent'].get_element_index(self)
        assert (index is not None)
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
        if with_gui():
            log(warning, "Call button common")
            self.button_text.set(self['name'])
            self.button_text_color.set('#BBDDBB')
            if self.button is not None:
                self.button.configure(bg=self.button_text_color.get())

    def make_common(self, id_set=None):
        """Revert back from uniqueness"""
        if inspect.isclass(self['type']) and issubclass(self['type'], BatchNorm2d):
            log(warning, "Can't make batch norm common, %s must be kept unique."
                % self['name'])
            return

        if id_set is None:
            id_set = set()

        if id(self) in id_set:
            return

        id_set.add(id(self))

        self['unique'] = False
        log(warning, "Calling make common")
        if self.has_unique_elements():
            log(warning, "Can't make top level common, %s has unique elements."
                % self['name'])
            self['unique'] = True
            for child in self['children']:
                child.make_common(id_set)
            for key, value in self.items():
                if isinstance(value, Blueprint) and key not in {"parent", "blueprint"} and value.has_unique_elements():
                    value.make_common(id_set)
            return

        log(warning,
            "Can make common, %s has no unique elements." % self['name'])

        index = get_common_scope_name_length(self['name'])

        if index > 0:
            self['name'] = self['name'][0:index]
            self.make_button_common()

        if self['parent'] is not None:
            self['parent'].make_common(id_set)

    def make_button_unique(self):
        if with_gui() and hasattr(self, 'button_text'):
            self.button_text_color.set('#FFAAAA')
            self.button_text.set(self['name'])
            if self.button is not None:
                self.button.configure(bg=self.button_text_color.get())

    def reset_name_without_unique_suffix(self):
        self['name'] = f"{self['prefix']}{self['suffix']}"

    def refresh_unique_suffixes(self):
        """Refresh unique suffixes of all under self"""
        for child in self['children']:
            if child.has_unique_elements():
                child.refresh_unique_suffixes()
                if child['unique']:
                    child.make_unique_name()

        for key, value in self.items():
            if isinstance(value, Blueprint) and value.has_unique_elements() and key != "parent":
                value.refresh_unique_suffixes()
                value.make_unique_name()

    def make_unique_name(self):
        """Make the current blueprint name unique and refresh it if it was already unique"""
        self.reset_name_without_unique_suffix()
        self['unique'] = True
        if UNIQUE_SUFFIX_DELIMITER not in self['name']:
            self['name'] = generate_random_scope(self['name'])
            self.make_button_unique()

    def make_unique(self, refresh_unique_suffixes=True):
        """Make the blueprint and all parents unique
        Refresh the unique suffixes of all children"""
        self.make_unique_name()

        if self['parent'] is not None:
            self['parent'].make_unique(refresh_unique_suffixes)
        elif refresh_unique_suffixes:
            # from the top level as all under it will be refreshed
            self.refresh_unique_suffixes()

    def get_element(self, index):
        if (isinstance(index, string_types)
                or not isinstance(index, Iterable)):
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
        if not with_gui():
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


def make_blueprint(json_dict):
    """From acyclic dictionary make a blueprint"""
    bp = Blueprint(description=json_dict)
    for k, v in json_dict.items():
        if k == 'parent':
            bp[k] = bp
        elif isinstance(v, Blueprint):
            bp[k] = make_blueprint(v)
        elif k == 'children' or k == 'bns':
            bp[k] = []
            for c in v:
                bp[k].append(make_blueprint(c))
        else:
            common.replace_key(v, 'blueprint', v)

    return bp


def get_duplicates(io_indices, one_out=1):
    """Return indices of non-unique shaped elements"""
    duplicates = []
    for k, v in io_indices.items():
        size = len(v)
        if size > one_out:
            cs = np.random.choice(v, size - one_out, replace=False)
            duplicates.extend(cs)

    return duplicates


def get_io_shape_indices(children):
    """Return the indices dict of children with the same io shapes"""

    def get_key(child):
        return str(child['input_shape']) + str(child['output_shape'])

    indices = defaultdict(list)

    for i, c in enumerate(children):
        if c['input_shape'] is not None and c['output_shape'] is not None:
            indices[get_key(c)].append(i)

    return indices


def toggle_uniqueness(blueprint, key, favor_common=0.5):
    if isinstance(blueprint[key], Blueprint):
        if blueprint[key]['unique']:
            blueprint[key].make_common()
        elif np.random.random() > favor_common:
            blueprint[key].make_unique()


def make_module(blueprint):
    """Construct named (or scoped) object given the blueprint"""
    assert (blueprint is not None)
    try:
        module = blueprint['type'](blueprint['name'], *blueprint['args'],
                                   **blueprint['kwargs'])
    except TypeError:
        log(warning, "make_module: Different typed objects, same scope!")
        blueprint.make_unique()
        module = blueprint['type'](blueprint['name'], *blueprint['args'],
                                   **blueprint['kwargs'])
    if module is not None:
        module.blueprint = blueprint

    return module


def visualize(blueprint):
    """Visualize module names, and toggle uniqueness"""
    master = common.GUI
    t = tk.Toplevel(master)
    t.wm_title("Info")
    info = tk.Text(t, wrap="none")
    info.tag_configure("center", justify=tk.CENTER)
    info.pack(fill="both", expand=True)

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


def visit_modules(blueprint, main_input, outputs=None,
                  fn=lambda bp, inp, o: o.append(bp.get_scope_button(*inp))):
    """Recursively apply a function to all modules

    e.g.add named buttons to the outputs"""
    if outputs is None:
        outputs = []

    if inspect.isclass(blueprint['type']) and issubclass(blueprint['type'], Module):
        fn(blueprint, main_input, outputs)

    for k, v in blueprint.items():
        if isinstance(v, Blueprint) and k != 'parent':
            # type all_to_none -> ignore the module
            if v['type'] != all_to_none:
                visit_modules(v, main_input, outputs, fn)

    for b in blueprint['children']:
        visit_modules(b, main_input, outputs, fn)


def collect_modules(blueprint,
                    collect=lambda bp, _, out: out.append(bp)):
    out = []
    visit_modules(blueprint, None, out, collect)
    return out


def collect_keys(blueprint, key,
                 collect=lambda bp, key, out:
                 out.append(bp[key]) if key in bp else None):
    out = []
    visit_modules(blueprint, key, out, collect)
    return out


def model_diagnostics(blueprint, input_tensor=None):
    if not common.DEBUG_MODEL_DIAGNOSTIC:
        return

    flattened = collect_modules(blueprint)
    for bp in flattened:
        kwargs = {k: v for k, v in bp.get('kwargs', {}).items() if k != 'parent'}
        print(f"{bp['name']}, in: {bp.get('input_shape', None)}, "
              f"out: {bp.get('output_shape', None)}, kwargs: {kwargs}")

    def hook(module, inp, outp):
        print(inp.shape if not isinstance(inp, tuple) else [i.shape for i in inp],
              "\33[91m%s\033[0m" % str(outp.shape),
              f"{type(module)}: {module.blueprint['name']}" if hasattr(module, "blueprint") else f"{type(module)}",
              flush=True)

    net = make_module(blueprint)
    for m in net.modules():
        m.register_forward_hook(hook)
    try:
        print("==== model shapes")
        if input_tensor is None:
            input_tensor = torch.rand(*blueprint['input_shape'])

        net(input_tensor)
        print("end of diagnostics")
    except:
        exception("Could not run the full thing due to exception")
