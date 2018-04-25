from stacked.meta.blueprint import Blueprint, visit_modules
from stacked.utils import common
from collections import Iterable
from logging import warning
import numpy as np


def log(log_func, msg):
    if common.DEBUG_HEURISTICS:
        log_func(msg)


def mutate_sub(blueprint, key, diameter, p, p_decay):
    """Mutate a sub-element

     Mutate in blueprint[key], where the sub-element is randomly picked
    """
    element = blueprint[key]
    if issubclass(type(element), Blueprint) and len(element['mutables']) > 0:
        random_key = np.random.choice(element['mutables'].keys())
        mutate(element, random_key, diameter, p, p_decay)


def mutate_current(blueprint, key, diameter):
    """Mutate the current blueprint"""
    domain = blueprint['mutables'][key]
    float_index = domain.get_normalized_index(blueprint[key])
    if float_index >= 0.0:
        new_index, value = domain.pick_random_neighbor(float_index, diameter)
    else:
        new_index, value = domain.pick_random()
    blueprint[key] = value


def mutate(blueprint, key=None, diameter=1.0, p=0.05, p_decay=1.0):
    """Mutate the blueprint element given the key

    Args:
        blueprint: Blueprint instance to mutate
        key: name of the item to mutate
        diameter: [0.0, 1.0] if small close by values will be picked
        p: probability with which mutate will operate
        p_decay: Multiplier for p, when a component is being mutated instead
    """
    domains = blueprint['mutables']
    if key is None:
        if len(domains) == 0:
            log(warning, "Blueprint {} has no mutable".format(blueprint['name']))
            return
        # can be extended with score based choice
        key = np.random.choice(domains.keys())

    assert (key in blueprint)
    if key not in domains or np.random.random() > p:
        mutate_sub(blueprint, key, diameter, p * p_decay, p_decay)
        return

    mutate_current(blueprint, key, diameter)


def search_elements(blueprint, main_input, add_if):
    """Accumulate elements with matching criterion

    Args:
        blueprint: Module description
        main_input: Input to be used in the append_fn
        add_if (Fn): that adds the blueprint itself, or its elements
    """
    blueprints = []
    visit_modules(blueprint, main_input, blueprints, add_if)
    return blueprints


def get_entry(blueprint, shape, append_fn):
    """Randomly pick an element with the given input shape"""
    blueprints = search_elements(blueprint, shape, append_fn)
    if len(blueprints) > 0:
        # can be extended with score based choice
        return np.random.choice(blueprints)
    return None


def search_matching_point(blueprint1, blueprint2, add_if, point):
    bp1 = get_entry(blueprint1, blueprint2[point], add_if)
    if bp1 is None:
        bp1 = blueprint1
        bp2 = get_entry(blueprint2, blueprint1[point], add_if)
        if bp2 is None:
            bp1 = blueprint1['name']
            bp2 = blueprint2['name']
            log(warning, "No matching point for {}, {}".format(bp1, bp2))
            return None, None
    else:
        bp2 = blueprint2
    return bp1, bp2


def add_if_in(bp, inp, outputs):
    if bp['input_shape'] == inp:
        outputs.append(bp)


def add_if_out(bp, output, outputs):
    if bp['output_shape'] == output:
        outputs.append(bp)


def get_random_index(blueprint, p_stop=0.5):
    if ('children' in blueprint and len(blueprint['children']) > 0
            and np.random.random() < p_stop):
        index = [np.random.randint(len(blueprint['children']))]
        blueprint = blueprint.get_element(index)
        assert(isinstance(blueprint, Blueprint))
        tail = get_random_index(blueprint, p_stop)
        if tail is not None:
            return index + tail
        return index
    elif 'linear' in blueprint or 'conv0' in blueprint:
        return [['linear'], ['conv0']][np.random.randint(2)]
    elif 'convdim' in blueprint:
        return ['convdim']
    return None


def search_cross_items(blueprint1, blueprint2):

    if blueprint1['input_shape'] != blueprint2['input_shape']:
        bps = search_matching_point(blueprint1, blueprint2, add_if_in, 'input_shape')
        blueprint1, blueprint2 = bps
        if blueprint2 is None or blueprint1 is None:
            return None, None

    out1 = blueprint1
    out2 = blueprint2
    if out1['output_shape'] != out2['output_shape']:
        bps = search_matching_point(out1, out2, add_if_out, 'output_shape')
        out1, out2 = bps
        if out2 is None or out1 is None:
            return None, None

    return blueprint1, blueprint2, out1, out2


def crossover(blueprint1, blueprint2, p_items=0.05,
              p_children=0.1, p_grandchildren=0.1):
    # try crossing random items
    if np.random.random() < p_items:
        items = search_cross_items(blueprint1, blueprint2)
        if None not in items:
            bp1, bp2, out1, out2 = items
            indices = []
            for i in items:
                index = i.get_index_from_root()
                if len(index) == 0:
                    pass
                # indices.append()

    # try crossing immediate children:
    if 'children' in blueprint1 and 'children' in blueprint2:
        # pick range of children
        # switch selected children
        pass

        # try crossing grand children
        pass





