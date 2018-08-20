from stacked.meta.blueprint import Blueprint, \
    get_io_shape_indices, toggle_uniqueness, get_duplicates
from stacked.utils import common
from logging import warning, info
import numpy as np
import copy


def log(log_func, msg):
    if common.DEBUG_HEURISTICS:
        log_func("stacked.meta.heuristics.operators: %s" % msg)


def mutate_sub(blueprint, key, diameter, p, p_decay):
    """Mutate a sub-element

     Mutate in blueprint[key], where the sub-element is randomly picked
    """
    element = blueprint[key]

    if (issubclass(type(element), Blueprint)
            and len(element['mutables']) > 0):
        random_key = np.random.choice(element['mutables'].keys())
        return mutate(element, random_key, diameter, p, p_decay)

    return False


def adjust_mutation(blueprint, key):
    """Post mutation adjustments to the blueprint"""
    if key == 'depth':
        # shrink children size to the given depth
        new_children = []
        children = blueprint['children']
        new_depth = blueprint[key]
        remove_count = len(children) - new_depth

        io_indices = get_io_shape_indices(children)
        duplicates = get_duplicates(io_indices)
        np.random.shuffle(duplicates)
        remove_set = set(duplicates[:remove_count])

        for i, c in enumerate(children):
            if i not in remove_set:
                new_children.append(c)

        blueprint['children'] = new_children
        log(info, 'Adjusted mutation: %s'
            % blueprint['children'])


def mutate_current(blueprint, key, diameter, p):
    """Mutate the current blueprint, or change uniqueness"""
    domain = blueprint['mutables'][key]

    def compare(bp1, bp2):
        return bp1 == bp2

    float_index = domain.get_normalized_index(blueprint[key], compare)

    if float_index >= 0.0:
        new_index, value = domain.pick_random_neighbor(float_index, diameter)
    else:
        new_index, value = domain.pick_random()

    blueprint[key] = value
    log(warning, 'Mutated %s, at key: %s with %s'
        % (blueprint['name'], key, value))

    adjust_mutation(blueprint, key)

    if np.random.random() < p:
        toggle_uniqueness(blueprint, key)

    return True


def mutate(blueprint, key=None, diameter=1.0, p=0.05, p_decay=1.0,
           choice_fn=lambda bp: np.random.choice(bp['mutables'].keys())):
    """Mutate the blueprint element given the key

    Args:
        blueprint: Blueprint instance to mutate
        key: name of the item to mutate
        diameter: [0.0, 1.0] if small close by values will be picked
        p: probability with which mutate will operate
        p_decay: Multiplier for p, when a component is being mutated instead
        choice_fn: Function that picks component key to mutate
    """
    domains = blueprint['mutables']

    if key is None:
        if len(domains) == 0:
            log(warning, "Blueprint {} not mutable".format(blueprint['name']))
            return False

        key = choice_fn(blueprint)

    if key not in domains or np.random.random() > p:
        return mutate_sub(blueprint, key, diameter, p * p_decay, p_decay)

    return mutate_current(blueprint, key, diameter, p)


def get_parent_ids(blueprint):
    return set([id(p) for p in blueprint.get_parents()])


def child_in_parents(blueprint1, blueprint2):
    parents = get_parent_ids(blueprint2)
    return id(blueprint1) in parents


def children_in_parents(children, parents):
    parents = set([id(p) for p in parents])

    for c in children:
        if id(c) in parents:
            return True

    return False


def readjust_uniqueness(blueprint):
    if blueprint['unique']:
        blueprint.make_unique()


def readjust_child(blueprint, parent):
    """Set the new parent and adjust uniqueness again"""
    blueprint['parent'] = parent
    readjust_uniqueness(blueprint)


def readjust_children(children, parent):
    for c in children:
        readjust_child(c, parent)


def swap_child(children1, children2, key1, key2):
    """Swap child in children1 with another in children2"""
    tmp = children1[key1]
    parent1 = tmp['parent']
    parent2 = children2[key2]['parent']

    # prevent cycles
    if (child_in_parents(tmp, children2[key2])
            or child_in_parents(children2[key2], tmp)):
        return False

    # swap
    children1[key1] = children2[key2]
    children2[key2] = tmp

    readjust_child(children1[key1], parent1)
    readjust_child(children2[key2], parent2)

    return True


def override_child(children1, children2, key1, key2):
    """Override child in children2 with one from children1"""
    parent = children2[key2]['parent']
    blueprint = copy.deepcopy(children1[key1])

    if child_in_parents(blueprint, children2[key2]):
        return False

    children2[key2] = blueprint

    readjust_child(blueprint, parent)

    return True


def cross_children(children1, children2,
                   index1, index2, ix1=None, ix2=None):
    parent1 = children1[0]['parent']
    parent2 = children2[0]['parent']
    parents1 = [parent1] + parent1.get_parents()
    parents2 = [parent2] + parent2.get_parents()

    if children_in_parents(children1[index1:ix1], parents2):
        return False

    if children_in_parents(children1[index2:ix2], parents1):
        return False

    common.swap_consecutive(children1, children2,
                            index1, index2, ix1, ix2)
    readjust_children(children1, parent1)
    readjust_children(children2, parent2)


def override_children(children1, children2,
                      index1, index2, ix1=None, ix2=None):
    """Override some of elements in children2 with ones from children1"""
    parent2 = children2[0]['parent']
    parents = [parent2] + parent2.get_parents()

    if children_in_parents(children1[index1:ix1], parents):
        return False

    children2[index2:ix2] = copy.deepcopy(children1[index1:ix1])

    readjust_children(children2, parent2)


def pick_key_dict(iterable1, iterable2,
                  key1='input_shape', key2='input_shape', ix1=0, ix2=0):
    """Pick index list dictionaries with the same key"""
    dict1 = common.get_same_value_indices(iterable1, key1, ix1)
    dict2 = common.get_same_value_indices(iterable2, key2, ix2)

    intersection = set(dict1.keys()).intersection(set(dict2.keys()))
    keys = list(intersection)
    return dict1, dict2, keys


def pick_random_cross_indices(shapes1, shapes2, keys):
    """Randomly pick indices, and matching key (shape) in shapes"""
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
                                           'output_shape', 'output_shape',
                                           index1, index2)

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


def crossover_children(iterable1, iterable2):
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
            cross_children(iterable1, iterable2, index1, index2)
            return True
        return False

    cross_children(iterable1, iterable2, index1, index2)
    cross_children(iterable1, iterable2, ix1, ix2)

    return True


def copy_children(children1, children2):
    """Override children

    Arguments should contain the keys input_shape, and output_shape
    """
    shapes1, shapes2, keys = pick_key_dict(children1, children2)
    index1, index2, key = pick_random_cross_indices(shapes1, shapes2, keys)

    if index1 is None or index2 is None:
        log(warning, "Children don't have matching input shape!")
        return False

    shapes1, shapes2, keys = pick_key_dict(children1, children2, 'output_shape',
                                           'output_shape', index1, index2)
    ix1, ix2, key = pick_random_cross_indices(shapes1, shapes2, keys)

    if (ix1 is None or ix2 is None or
            ix1 == len(children1) or ix2 == len(children2)):
        if children1[-1]['output_shape'] == children2[-1]['output_shape']:
            override_children(children1, children2, index1, index2)
            return True
        return False

    override_children(children1, children2, index1, index2, ix1 + 1, ix2 + 1)

    return True


def op_over_children(blueprint1, blueprint2, p_items, p_children,
                     fn_over, fn1=swap_child, fn=crossover_children):
    """Crossover or copy over children (fn_over)"""
    if 'children' in blueprint1 and 'children' in blueprint2:
        children1 = blueprint1['children']
        children2 = blueprint2['children']

        # try crossover or copy over immediate children:
        if np.random.random() < p_children:
            if fn(children1, children2):
                return True

        # try over grandchildren, or children x grandchildren
        if np.random.random() < p_children:
            c1 = children1 + [blueprint1]
            c2 = children2 + [blueprint2]
            bp1 = np.random.choice(c1)
            bp2 = np.random.choice(c2)
            if fn_over(bp1, bp2, p_items, p_children, fn1, fn):
                return True

    return False


def op_over_item(blueprint1, blueprint2, key, fn=swap_child):
    if key in blueprint1 and key in blueprint2:
        key1 = blueprint1[key]
        key2 = blueprint2[key]
        if (key1 is not None and key2 is not None
                and key1['input_shape'] == key2['input_shape']
                and key1['input_shape'] is not None
                and key1['output_shape'] is not None
                and key1['output_shape'] == key2['output_shape']):
            return fn(blueprint1, blueprint2, key, key)

    return False


def op_over(blueprint1, blueprint2, p_items=0.5, p_children=0.9,
            fn1=swap_child, fn2=crossover_children):
    """In place, crossover like operation on conv, convdim, linear or children"""
    if np.random.random() < p_items:
        if op_over_item(blueprint1, blueprint2, 'conv', fn1):
            return True
        if op_over_item(blueprint1, blueprint2, 'convdim', fn1):
            return True
        if op_over_item(blueprint1, blueprint2, 'linear', fn1):
            return True

    return op_over_children(blueprint1, blueprint2,
                            p_items, p_children, op_over, fn1, fn2)


def crossover(blueprint1, blueprint2, p_items=0.5, p_children=0.9):
    return op_over(blueprint1, blueprint2, p_items, p_children)


def copyover(blueprint1, blueprint2, p_items=0.5, p_children=0.9):
    return op_over(blueprint1, blueprint2, p_items, p_children,
                   fn1=override_child, fn2=copy_children)
