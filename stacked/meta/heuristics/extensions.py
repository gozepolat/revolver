from stacked.models.blueprinted import ScopedEnsemble, ScopedResBlock
from stacked.utils.domain import ClosedList


def append_mutables(elements, mutables):
    """Append mutables to the elements, if they are not in elements"""
    new_mutables = []
    names = set([e['names'] for e in elements])
    for bp in mutables:
        if bp['name'] not in names:
            new_mutables.append(bp)
    return elements + new_mutables


def extend_conv_mutables(blueprint, ensemble_size=5, block_depth=2):
    """Create mutation options for the conv of the given blueprint

    Supports the existing convolution op, ensemble, and res_block
    """
    prefix = blueprint['conv']['prefix']
    conv = blueprint['conv']
    parent = conv['parent']
    bp = ScopedEnsemble.describe_from_blueprint(prefix, '',
                                                conv, parent,
                                                ensemble_size)

    res_block = ScopedResBlock.describe_from_blueprint(prefix, "_block",
                                                       conv, parent,
                                                       block_depth)
    mutables = [conv, res_block, bp]
    if 'conv' in blueprint['mutables']:
        elements = blueprint['mutables']['conv'].elements
        mutables = append_mutables(elements, mutables)

    blueprint['mutables'] = {
        'conv': ClosedList(mutables)
    }
