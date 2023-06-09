from revolver.models.blueprinted.bottleneckblock import ScopedBottleneckBlock
from revolver.models.blueprinted.ensemble import ScopedEnsembleMean
from revolver.models.blueprinted.resblock import ScopedResBlock
from revolver.models.blueprinted.meta import ScopedMetaMasked
from revolver.models.blueprinted.convdeconv import ScopedConv2dDeconv2d
from revolver.models.blueprinted.separable import ScopedDepthwiseSeparable
from revolver.modules.scoped_nn import ScopedConv3d2d, ScopedBatchNorm2d, ScopedConv2d
from revolver.utils.domain import ClosedList, ClosedInterval
from revolver.utils import common
import numpy as np
from torch.nn import Conv2d, BatchNorm2d
import inspect


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

    bottleneck = ScopedBottleneckBlock.describe_from_blueprint(prefix, "_block", conv,
                                                                      parent, block_depth)

    # meta = ScopedMetaMasked.describe_from_blueprint(prefix, '_meta', conv, parent)

    deconv = ScopedConv2dDeconv2d.describe_from_blueprint(prefix, '_deconv', conv, parent)

    mutables = [conv, res_block, ensemble, separable, bottleneck, deconv]

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


def extend_conv_kernel_size_mutables(blueprint, min_kernel_size=1, max_kernel_size=4):
    """Create mutation options for the conv of the given blueprint

    Supports the existing convolution op, ensemble, and res_block
    """

    if not inspect.isclass(blueprint['type']):
        return

    classes = [ScopedConv2d, ScopedConv2dDeconv2d, ScopedEnsembleMean,
               ScopedResBlock, ScopedDepthwiseSeparable, ScopedBottleneckBlock]

    if not np.any([issubclass(blueprint['type'], classType) for classType in classes]):
        return

    def new_kernel_size(value):
        new_kwargs = blueprint['kwargs'].copy()
        new_kwargs['kernel_size'] = value
        return new_kwargs

    kwargs_list = [new_kernel_size(k) for k in range(min_kernel_size, max_kernel_size, 2)]
    blueprint['mutables']['kwargs'] = ClosedList(kwargs_list)


def extend_depth_mutables(blueprint, min_depth=2):
    if 'depth' not in blueprint or len(blueprint['children']) == 0:
        return

    # if 'bns' in blueprint and not issubclass()

    max_depth = len(blueprint['children'])

    if max_depth < min_depth:
        min_depth = max_depth

    blueprint['mutables']['depth'] = ClosedInterval(int(min_depth), int(max_depth), element_type=int)


def extend_bn_mutables(bp, min_momentum=0.05, max_momentum=0.99):
    if not (inspect.isclass(bp['type']) and issubclass(bp['type'], BatchNorm2d)):
        return

    def new_momentum(value):
        new_kwargs = bp['kwargs'].copy()
        new_kwargs['momentum'] = value
        return new_kwargs

    kwargs_list = [new_momentum(m) for m in np.arange(min_momentum, max_momentum, 0.05)]
    bp['mutables']['kwargs'] = ClosedList(kwargs_list)


def extend_mutation_mutables(blueprint, min_mutation_p=0.001, max_mutation_p=1.0,
                             min_toggle_p=0.001, max_toggle_p=0.5):
    if 'mutation_p' in blueprint:
        blueprint['mutables']['mutation_p'] = ClosedInterval(min_mutation_p, max_mutation_p)
    if 'toggle_p' in blueprint:
        blueprint['mutables']['toggle_p'] = ClosedInterval(min_toggle_p, max_toggle_p)
