from stacked.meta.blueprint import Blueprint
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
        # can be extended with score based choice
        key = np.random.choice(domains.keys())

    assert (key in blueprint)
    if key not in domains or np.random.random() > p:
        mutate_sub(blueprint, key, diameter, p * p_decay, p_decay)
        return

    mutate_current(blueprint, key, diameter)


def swap_consecutive(container1, container2, index1, index2):
    """Swap all the elements after the entry point (index1, index2)"""
    tmp = [k for k in container1]
    container1[index1:] = container2[index2:]
    container2[index2:] = tmp[index1:]


def reset_parent(children, parent):
    for c in children:
        c['parent'] = parent


def cross_elements(iterable1, iterable2, index1, index2):
    parent1 = iterable1[0]['parent']
    parent2 = iterable2[0]['parent']
    swap_consecutive(iterable1, iterable2, index1, index2)
    reset_parent(iterable1, parent1)
    reset_parent(iterable2, parent2)


def pick_key_dict(iterable1, iterable2,
                  key='input_shape', ix1=0, ix2=0):
    """Pick index list dictionaries with the same key"""
    dict1 = common.get_same_value_indices(iterable1, key, ix1)
    dict2 = common.get_same_value_indices(iterable2, key, ix2)

    intersection = set(dict1.keys()).intersection(set(dict2.keys()))
    keys = list(intersection)
    return dict1, dict2, keys


def pick_random_cross_indices(shapes1, shapes2, keys):
    if len(keys) > 0:
        key = np.random.choice(keys)
        assert(len(shapes1[key]) > 0 and len(shapes2[key]) > 0)
        index1 = np.random.choice(shapes1[key])
        index2 = np.random.choice(shapes2[key])
        return index1, index2, key

    return None, None, None


def search_exit(iterable1, iterable2, index1, index2):
    """Search for a crossover exit point, given entry (index1, index2)

    No cross operation is done on children[1,2] yet
    (index1, index2) is a hypothetical entry point
    """
    # shape1, shape2 switched due to cross_elements no being done
    shapes2, shapes1, keys = pick_key_dict(iterable1, iterable2,
                                           'output_shape', index1, index2)

    # search for a random exit point as if cross_elements was done
    ix1, ix2, key = pick_random_cross_indices(shapes1, shapes2, keys)
    len2 = len1 = None

    if ix1 is not None:
        ix1 = ix1 - index2 + index1 + 1
        len1 = index1 + len(iterable2) - ix2
    if ix2 is not None:
        ix2 = ix2 - index1 + index2 + 1
        len2 = index2 + len(iterable1) - ix1

    return ix1, ix2, len1, len2


def cross_children(iterable1, iterable2):
    """Crossover on children

    Arguments should contain the keys input_shape, and output_shape
    """
    shapes1, shapes2, keys = pick_key_dict(iterable1, iterable2)
    index1, index2, key = pick_random_cross_indices(shapes1, shapes2, keys)
    if index1 is None or index2 is None:
        log(warning, "Children don't have matching input shape!")
        return False

    ix1, ix2, len1, len2 = search_exit(iterable1, iterable2, index1, index2)
    if (ix1 is None or ix2 is None or
            ix1 == len1 or ix2 == len2):
        if iterable1[-1]['output_shape'] == iterable2[-1]['output_shape']:
            cross_elements(iterable1, iterable2, index1, index2)
            return True
        return False

    cross_elements(iterable1, iterable2, index1, index2)
    cross_elements(iterable1, iterable2, ix1, ix2)
    return True


def crossover_on_children(blueprint1, blueprint2, p_items, p_children):
    if 'children' in blueprint1 and 'children' in blueprint2:
        children1 = blueprint1['children']
        children2 = blueprint2['children']

        # try crossing immediate children:
        if np.random.random() < p_children:
            if cross_children(children1, children2):
                return True

        # try crossing grandchildren, or children x grandchildren
        if np.random.random() < p_children:
            c1 = children1 + [blueprint1]
            c2 = children2 + [blueprint2]
            bp1 = np.random.choice(c1)
            bp2 = np.random.choice(c2)
            if crossover(bp1, bp2, p_items, p_children):
                return True
    return False


def crossover_on_item(blueprint1, blueprint2, key):
    if key in blueprint1 and key in blueprint2:
        key1 = blueprint1[key]
        key2 = blueprint2[key]

        if (key1['input_shape'] == key2['input_shape']
                and key1['output_shape'] == key2['output_shape']):
            common.swap_elements(blueprint1, blueprint2, key, key)
            return True

    return False


def crossover(blueprint1, blueprint2, p_items=0.5, p_children=0.9):
    """In place, crossover operation on conv, convdim, linear or children"""
    if np.random.random() < p_items:
        if crossover_on_item(blueprint1, blueprint2, 'conv'):
            return True
        if crossover_on_item(blueprint1, blueprint2, 'convdim'):
            return True
        if crossover_on_item(blueprint1, blueprint2, 'linear'):
            return True

    return crossover_on_children(blueprint1, blueprint2, p_items, p_children)
