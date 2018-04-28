from stacked.models.blueprinted import ScopedEnsemble
from stacked.utils.domain import ClosedList


def extend_mutables(blueprint, parent, ensemble_size=5):
    prefix = blueprint['conv']['prefix']
    conv = blueprint['conv']
    bp = ScopedEnsemble.describe_ensemble_from_blueprint(prefix, "_ensemble",
                                                         conv, parent,
                                                         ensemble_size)
    mutables = [bp, conv]
    if 'conv' in blueprint['mutables']:
        mutables = mutables + blueprint['mutables']['conv'].elements

    blueprint['mutables'] = {
        'conv': ClosedList(mutables)
    }
