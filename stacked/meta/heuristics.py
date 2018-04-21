from stacked.meta.blueprint import Blueprint
import numpy as np


def mutate(blueprint, key, diameter, p=0.05, p_decay=1.0):
    """Mutate the blueprint element given the key

    Args:
        blueprint: Blueprint instance to mutate
        key: name of the item to mutate
        diameter: [0.0, 1.0] if small close by values will be picked
        p: probability with which mutate will operate
        p_decay: Multiplier for p, when a component is being mutated instead
    """
    assert (key in blueprint)
    domains = blueprint['evolvables']
    if key not in domains or np.random.random() > p:
        element = blueprint[key]
        # try mutating a component instead
        if issubclass(type(element), Blueprint) and len(element['evolvables']) > 0:
            random_key = np.random.choice(element['evolvables'].keys())
            mutate(element, random_key, diameter, p)
        return
    domain = domains[key]
    float_index = domain.get_normalized_index(blueprint[key])
    if float_index >= 0.0:
        new_index, value = domain.pick_random_neighbor(float_index, diameter)
    else:
        new_index, value = domain.pick_random()
    blueprint[key] = value







