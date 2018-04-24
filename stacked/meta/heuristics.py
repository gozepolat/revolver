from stacked.meta.blueprint import Blueprint, visit_modules
from stacked.utils import common
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


def search_cross_items(blueprint1, blueprint2):

    def add_if_in(bp, inp, outputs):
        if bp['input_shape'] == inp:
            outputs.append(bp)

    def add_if_out(bp, outp, outputs):
        if bp['output_shape'] == outp:
            outputs.append(bp)

    if blueprint2['input_shape'] != blueprint2['input_shape']:
        bps = search_matching_point(blueprint1, blueprint2, add_if_in, 'input_shape')
        blueprint1, blueprint2 = bps

    if blueprint2['output_shape'] != blueprint2['output_shape']:
        bps = search_matching_point(blueprint1, blueprint2, add_if_out, 'output_shape')
        blueprint1, blueprint2 = bps

    return blueprint1, blueprint2


def crossover(blueprint1, blueprint2, p_items=0.05,
              p_children=0.1, p_grandchildren=0.1):
    # try crossing immediate items
    if np.random.random() < p_items:
        bp1, bp2 = search_cross_items()
        pass

    # try crossing immediate children:
    if 'children' in blueprint1 and 'children' in blueprint2:
        # pick range of children
        # switch selected children
        pass

        # try crossing grand children
        pass





