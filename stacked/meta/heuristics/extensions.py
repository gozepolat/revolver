from stacked.models.blueprinted.ensemble import ScopedEnsembleMean
from stacked.models.blueprinted.resblock import ScopedResBlock
from stacked.models.blueprinted.resbottleneckblock import ScopedResBottleneckBlock
from stacked.models.blueprinted.meta import ScopedMetaMasked
from stacked.models.blueprinted.convdeconv import ScopedConv2dDeconv2d
from stacked.models.blueprinted.separable import ScopedDepthwiseSeparable
from stacked.modules.scoped_nn import ScopedConv3d2d
from stacked.utils.domain import ClosedList, ClosedInterval
from stacked.utils import common


def append_mutables(elements, mutables):
    """Append mutables to the elements, if they are not in elements"""
    new_mutables = []
    mutable_ids = set([id(e) for e in elements])
    for bp in mutables:
        if id(bp) not in mutable_ids:
            new_mutables.append(bp)
    return elements + new_mutables


def extend_conv_mutables(blueprint, ensemble_size=5, block_depth=2):
    """Create mutation options for the conv of the given blueprint

    Supports the existing convolution op, conv3d2d, ensemble, and res_block
    """
    if 'conv' not in blueprint:
        return

    prefix = blueprint['conv']['prefix']
    conv = blueprint['conv']
    parent = conv['parent']
    ensemble = ScopedEnsembleMean.describe_from_blueprint(prefix, '_ensemble',
                                                          conv, parent, ensemble_size)

    res_block = ScopedResBlock.describe_from_blueprint(prefix, "_block",
                                                       conv, parent, block_depth)

    separable = ScopedDepthwiseSeparable.describe_from_blueprint(prefix, '_separable',
                                                                 conv, parent)

    res_bottleneck = ScopedResBottleneckBlock.describe_from_blueprint(prefix, "_block", conv,
                                                                      parent, block_depth)

    meta = ScopedMetaMasked.describe_from_blueprint(prefix, '_meta', conv, parent)

    deconv = ScopedConv2dDeconv2d.describe_from_blueprint(prefix, '_deconv', conv, parent)

    mutables = [conv, res_block, ensemble, meta, separable, res_bottleneck, deconv]

    in_channels = blueprint['conv']['input_shape'][1]
    out_channels = blueprint['conv']['output_shape'][1]

    gcd = common.gcd(out_channels, in_channels)
    if gcd > 1:
        kwargs = {'in_channels': in_channels // gcd,
                  'out_channels': out_channels // gcd}
        conv3d2d = ScopedConv3d2d.describe_from_blueprint(prefix, '_conv3d2d', conv, parent, kwargs)
        mutables.append(conv3d2d)

    if 'conv' in blueprint['mutables']:
        elements = blueprint['mutables']['conv'].elements
        mutables = append_mutables(elements, mutables)

    blueprint['mutables']['conv'] = ClosedList(mutables)


def extend_conv_kernel_size_mutables(blueprint, min_kernel_size=3, max_kernel_size=15):
    """Create mutation options for the conv of the given blueprint

    Supports the existing convolution op, ensemble, and res_block
    """
    blueprint['mutables']['conv_kernel'] = ClosedList(list(range(min_kernel_size, max_kernel_size, 2)))


def extend_depth_mutables(blueprint, min_depth=2):
    if 'depth' not in blueprint or len(blueprint['children']) == 0:
        return

    max_depth = len(blueprint['children'])

    if max_depth < min_depth:
        min_depth = max_depth

    blueprint['mutables']['depth'] = ClosedInterval(min_depth, max_depth)
