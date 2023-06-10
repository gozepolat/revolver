# -*- coding: utf-8 -*-
import torch
import torch.utils.data as data
from torch.utils.data import DataLoader
from torch.autograd import Variable
from revolver.models.blueprinted.resnet import ScopedResNet
from revolver.meta.scope import ScopedMeta
from revolver.meta.blueprint import Blueprint, make_module
from revolver.modules.scoped_nn import ScopedBatchNorm2d, \
    ScopedReLU, ScopedConv2d, ScopedLinear, ScopedCrossEntropyLoss
from revolver.models.blueprinted.resblock import ScopedResBlock
from revolver.models.blueprinted.resgroup import ScopedResGroup
from revolver.models.blueprinted.convunit import ScopedConvUnit
from revolver.utils.engine import EpochEngine, EngineEventHooks
from revolver.utils.dataset import create_dataset
from revolver.utils.transformer import all_to_none
from revolver.utils import common
from logging import warning
from six import add_metaclass, string_types
from torch.optim import SGD
from tqdm import tqdm
import pandas as pd


def log(log_func, msg):
    if common.DEBUG_OPTIMIZER:
        log_func("revolver.meta.optimizer: %s" % msg)


def get_all_parameters(model):
    return model.parameters()


@add_metaclass(ScopedMeta)
class ScopedOptimizerMaker:
    def __init__(self, scope, blueprint, *_, **__):
        self.scope = scope
        self.optimizer_type = blueprint['optimizer_type']
        self.optimizer_parameter_picker = blueprint['optimizer_parameter_picker']
        self.momentum = blueprint['momentum']
        self.weight_decay = blueprint['weight_decay']

    def __call__(self, model, lr, *args, **kwargs):
        params = self.optimizer_parameter_picker(model)
        log(warning, "Making new optimizer!! with lr %f" % lr)
        return self.optimizer_type(params, lr, momentum=self.momentum,
                                   weight_decay=self.weight_decay, *args, **kwargs)

    @staticmethod
    def describe_default(prefix, suffix, parent,
                         optimizer_type, optimizer_parameter_picker,
                         momentum, weight_decay):
        default = Blueprint(prefix, suffix, parent, False, ScopedOptimizerMaker)

        default['optimizer_type'] = optimizer_type

        default['momentum'] = momentum
        default['weight_decay'] = weight_decay

        if optimizer_parameter_picker is None:
            optimizer_parameter_picker = get_all_parameters

        default['optimizer_parameter_picker'] = optimizer_parameter_picker

        default['kwargs'] = {'blueprint': default}
        return default


@add_metaclass(ScopedMeta)
class ScopedDataLoader(DataLoader):
    def __init__(self, scope, blueprint, *_, **__):
        dataset = create_dataset(blueprint['dataset'],
                                 '.', blueprint['train_mode'],
                                 blueprint['is_validation'],
                                 blueprint['crop_size'])

        if blueprint['train_mode']:
            dataset, _ = split_dataset(dataset, blueprint['validation_ratio'])
        elif blueprint['is_validation']:
            _, dataset = split_dataset(dataset, blueprint['validation_ratio'])

        super(ScopedDataLoader, self).__init__(dataset,
                                               batch_size=blueprint['batch_size'],
                                               shuffle=blueprint['train_mode'],
                                               num_workers=blueprint['num_thread'],
                                               pin_memory=torch.cuda.is_available())
        self.scope = scope

    @staticmethod
    def describe_default(prefix, suffix, parent,
                         dataset, train_mode, batch_size, num_thread, crop_size,
                         is_validation=False, validation_ratio=.1, is_unique=False):
        default = Blueprint(prefix, suffix, parent, is_unique, ScopedDataLoader)
        if is_unique:
            default.refresh_name()

        default['dataset'] = dataset
        default['batch_size'] = batch_size
        default['train_mode'] = train_mode
        default['num_thread'] = num_thread
        default['crop_size'] = crop_size
        default['is_validation'] = is_validation
        default['validation_ratio'] = validation_ratio

        default['kwargs'] = {'blueprint': default}
        return default


@add_metaclass(ScopedMeta)
class ScopedCriterion:
    def __init__(self, scope, blueprint, *_, **__):
        self.scope = scope

        self.criterion = make_module(blueprint['criterion'])

    def __call__(self, out, target, *args, **kwargs):
        return self.criterion(out, target)

    @staticmethod
    def describe_default(prefix, suffix, parent,
                         criterion=ScopedCrossEntropyLoss):
        default = Blueprint(prefix, suffix, parent, False, ScopedCriterion)

        default['criterion'] = Blueprint('%s/criterion' % prefix, suffix,
                                         default, False, criterion)

        default['kwargs'] = {'blueprint': default}
        return default


@add_metaclass(ScopedMeta)
class ScopedNetRunner:
    def __init__(self, scope, blueprint, *_, **__):
        self.scope = scope
        self.loss_func = make_module(blueprint['loss_func'])
        self.net = None

    def set_model(self, engine, model):
        self.net = model
        self.loss_func.criterion.engine = engine

    def __call__(self, sample):
        x_input = Variable(getattr(sample[0].cuda() if torch.cuda.is_available() else sample[0], 'float')())
        y_targets = Variable(getattr(sample[1].cuda() if torch.cuda.is_available() else sample[1], 'long')())
        y_out = self.net(x_input)
        return self.loss_func(y_out, y_targets), y_out

    @staticmethod
    def describe_default(prefix, suffix, parent,
                         criterion=ScopedCrossEntropyLoss,
                         loss_func=ScopedCriterion):
        default = Blueprint(prefix, suffix, parent, True, ScopedNetRunner)
        default['loss_func'] = loss_func.describe_default("%s/loss_func" % prefix,
                                                          suffix, default, criterion)
        default['kwargs'] = {'blueprint': default}
        return default


def split_dataset(train_set, validation_ratio=.1, seed=None):
    valid_set_size = int(len(train_set) * validation_ratio)
    train_set_size = len(train_set) - valid_set_size
    if seed is None:
        seed = common.SEED
        common.SEED += 1
    seed = torch.Generator().manual_seed(seed)
    train, validation = data.random_split(train_set,
                                          [train_set_size, valid_set_size],
                                          generator=seed)
    return train, validation


@add_metaclass(ScopedMeta)
class ScopedEpochEngine(EpochEngine):
    """Training engine with blueprint"""

    def __init__(self, scope, blueprint, *_, **__):
        super(ScopedEpochEngine, self).__init__()
        self.scope = scope
        self.blueprint = blueprint

        self.retain_graph = blueprint['retain_graph']
        test_mode = blueprint["test_mode"]
        engine = self

        test_loader = self.test_loader = make_module(blueprint['test_loader'])
        train_loader = self.train_loader = make_module(blueprint['train_loader'])
        validation_loader = make_module(blueprint['validation_loader'])

        log(warning, f"Make module {blueprint['net']['name']}")
        net = make_module(blueprint['net'])
        if torch.cuda.is_available():
            net.cuda()
        # TODO remove
        # torch.autograd.set_detect_anomaly(True)
        self.net = net
        net_runner = make_module(blueprint['net_runner'])
        net_runner.set_model(engine, net)

        optimizer_maker = make_module(blueprint['optimizer_maker'])

        logger = make_module(blueprint['logger'])

        lr = blueprint['learning_rate']
        lr_decay_ratio = blueprint['lr_decay_ratio']
        lr_drop_epochs = blueprint['lr_drop_epochs']

        train_id = id(net.blueprint)
        hooks = EngineEventHooks(engine, train_loader, validation_loader=validation_loader, test_loader=test_loader,
                                 net=net, net_runner=net_runner, make_optimizer=optimizer_maker, learning_rate=lr,
                                 lr_decay_ratio=lr_decay_ratio, lr_drop_epochs=lr_drop_epochs, logger=logger,
                                 train_id=train_id,
                                 use_tqdm=blueprint['use_tqdm'])

        self.hooks['on_sample'] = hooks.on_sample
        self.hooks['on_forward'] = hooks.on_forward
        self.hooks['on_start_epoch'] = hooks.on_start_epoch
        self.hooks['on_end_epoch'] = hooks.on_end_epoch
        self.hooks['on_start'] = hooks.on_start
        self.hooks['on_end'] = hooks.on_end

        if test_mode:
            return

        self.set_state(net_runner, train_loader,
                       blueprint['max_epoch'], optimizer_maker(self.net, lr),
                       epoch=0, t=0, train=True)

        self.hook('on_start', self.state)

    def load_state_dict(self, state):
        """Load the state of the engine, optimizer, as well as the network"""
        if isinstance(state, string_types):
            log(Warning, "State is not a dictionary, attempting to load as a file")
            state = torch.load(state)

        if 'blueprint' in state:
            log(warning, 'Overriding the blueprint, using the last saved state')
            self.blueprint = pd.read_pickle(state['blueprint'])

        net = self.state['network'].net
        net.load_state_dict(state['network'])
        self.state['optimizer'].load_state_dict(state['optimizer'])

        for k, v in state.items():
            if k not in ['network', 'optimizer']:
                self.state[k] = v

        if self.blueprint['use_tqdm']:
            self.state['iterator'] = tqdm(self.state['iterator'])

    def dump_state(self, filename=None):
        """Save the state of the engine, optimizer, as well as the network"""
        if filename is None:
            filename = '{}_epoch_{}.pth.tar'.format(self.blueprint['net']['name'],
                                                    self.state['epoch'])

        log(Warning, "dump_state: to %s" % filename)
        d = self.state.copy()
        d['network'] = self.net.state_dict()
        d['optimizer'] = self.state['optimizer'].state_dict()
        d['iterator'] = self.train_loader
        d['blueprint'] = f"{self.blueprint['name']}.pkl"
        torch.save(d, filename)
        pd.to_pickle(self.blueprint, d['blueprint'])

        if common.DEBUG_OPTIMIZER_VERBOSE:
            log(warning, "Dumped engine blueprint:")
            log(warning, "=====================")
            log(warning, "%s" % self.blueprint)
            log(warning, "=====================")

    @staticmethod
    def describe_default(prefix='EpochEngine', suffix='', parent=None,
                         net_blueprint=None, net_module=ScopedResNet,
                         skeleton=(16, 32, 64), group_depths=None, num_classes=10,
                         depth=28, width=1, block_depth=2,
                         block_module=ScopedResBlock, conv_module=ScopedConv2d,
                         bn_module=ScopedBatchNorm2d, linear_module=ScopedLinear,
                         act_module=ScopedReLU, kernel_size=3, padding=1,
                         input_shape=None, dilation=1, groups=1, bias=False,
                         drop_p=0.0, dropout_p=0.0, residual=True,
                         conv_kwargs=None, bn_kwargs=None, act_kwargs=None,
                         unit_module=ScopedConvUnit, group_module=ScopedResGroup,
                         fractal_depth=1, shortcut_index=-1,
                         dense_unit_module=ScopedConvUnit, no_weights=False,
                         head_kernel=7, head_stride=2, head_padding=3,
                         head_pool_kernel=3, head_pool_stride=2,
                         head_pool_padding=1, head_modules=('conv', 'bn', 'act', 'pool'),
                         optimizer_maker=ScopedOptimizerMaker,
                         optimizer_type=SGD, optimizer_parameter_picker=None,
                         max_epoch=200, batch_size=128,
                         learning_rate=0.1, lr_decay_ratio=0.2,
                         lr_drop_epochs=(60, 120, 160), logger=None,
                         data_loader=ScopedDataLoader, dataset="CIFAR10",
                         crop_size=32, num_thread=4, net_runner=ScopedNetRunner,
                         criterion=ScopedCrossEntropyLoss,
                         loss_func=ScopedCriterion, callback=all_to_none,
                         use_tqdm=False, momentum=0.9, weight_decay=0.0001,
                         validation_ratio=.1, test_mode=False):
        """Create a default ResBlock blueprint

        Args:
            prefix (str): Prefix from which the member scopes will be created
            suffix (str): Suffix to append the name of the scoped object
            parent (Blueprint): None or the instance of the parent blueprint
            net_blueprint (Blueprint): None or the blueprint of the network to be used
            net_module: Main network type (e.g. ScopedDenseNet, ScopedResNet etc.)
            skeleton (iterable): Smallest possible widths per group
            group_depths (iterable): Finer grained group depth description
            num_classes (int): Number of categories for supervised learning
            depth (int): Overall depth of the network
            width (int): Scalar to get the scaled width per group
            block_depth (int): Number of [conv/act/bn] units in the block
            block_module: Children modules used as block modules
            conv_module (type): CNN module to use in forward. e.g. ScopedConv2d
            bn_module: Batch normalization module. e.g. ScopedBatchNorm2d
            linear_module (type): Linear module for classification e.g. ScopedLinear
            act_module (type): Activation module e.g ScopedReLU
            kernel_size (int or tuple): Size of the convolving kernel.
            padding (int or tuple, optional): Padding for the first convolution
            input_shape (tuple): (N, C_{in}, H_{in}, W_{in})
            dilation (int): Spacing between kernel elements.
            groups: Number of blocked connections from input to output channels.
            bias: Add a learnable bias if True
            drop_p: Probability of vertical drop
            dropout_p: Probability of dropout in the blocks
            residual (bool): True if a shortcut connection will be used
            conv_kwargs: extra conv arguments to be used in children
            bn_kwargs: extra bn args, if bn module requires other than 'num_features'
            act_kwargs: extra act args, if act module requires other than defaults
            unit_module (type): basic building unit of resblock
            group_module (type): basic building group of resnet
            fractal_depth (int): recursion depth for fractal group module
            shortcut_index (int): Starting index for groups shortcuts to the linear layer
            dense_unit_module: Children modules that will be used in dense connections
            no_weights (bool): Weight sum and softmax the reused blocks or not
            head_kernel (int or tuple): Size of the kernel for head conv
            head_stride (int or tuple): Size of the stride for head conv
            head_padding (int or tuple): Size of the padding for head conv
            head_pool_kernel (int or tuple): Size of the first pool kernel
            head_pool_stride (int or tuple): Size of the first pool stride
            head_pool_padding (int or tuple): Size of the first pool padding
            head_modules (iterable): Key list of head modules
            optimizer_maker: Functor that will return an optimizer
            optimizer_type: Type of the optimizer that will be returned
            optimizer_parameter_picker: Function to pick the parameters to be optimized
            max_epoch (int): Maximum number of epochs for training
            batch_size (int): Batch size for training
            learning_rate (float): Initial learning rate for training
            lr_decay_ratio (float): Scalar that will be multiplied with the learning rate
            lr_drop_epochs (iterable): Epoch numbers where where lr *= lr_decay_ratio will occur
            logger: Functor that will print the training progress (None: only to stdout)
            data_loader: Loader that will load a dataset (and pad / augment it)
            dataset (str): Name of the dataset the loader can use
            crop_size (int): Size of the image samples for cropping after padding for data loader
            num_thread (int): Number of subprocesses for the data loader
            net_runner: Functor that runs the network and returns loss, output
            criterion: Loss criterion function to be used in net_runner
            loss_func: Module that can customize the loss criterion or use it as is
            callback: function to call after the output in forward is calculated
            use_tqdm (bool): use progress bar for each epoch during training
            momentum (float, optional): momentum factor (default: 0.9)
            weight_decay (float, optional): weight decay (L2 penalty) (default: 0.0005)
            validation_ratio (float, optional): ratio for the validation (default: .1)
            test_mode (bool, optional): when True, ignore the train/val loader and test once
        """
        suffix = f"{suffix}_test{test_mode}"
        default = Blueprint(prefix, suffix, parent, False, ScopedEpochEngine)

        default['learning_rate'] = learning_rate
        default['lr_decay_ratio'] = lr_decay_ratio
        default['lr_drop_epochs'] = lr_drop_epochs
        default['max_epoch'] = max_epoch
        default['validation_ratio'] = validation_ratio

        default['net_runner'] = net_runner.describe_default("%s/net_runner" % prefix,
                                                            suffix, default, criterion,
                                                            loss_func)

        default['train_loader'] = data_loader.describe_default("%s/train_loader" % prefix,
                                                               suffix, default, dataset,
                                                               True, batch_size, num_thread,
                                                               crop_size, is_validation=False,
                                                               validation_ratio=validation_ratio)

        default['validation_loader'] = data_loader.describe_default("%s/validation_loader" % prefix,
                                                                    suffix, default, dataset,
                                                                    False, batch_size, num_thread,
                                                                    crop_size, is_validation=True,
                                                                    validation_ratio=validation_ratio)

        default['test_loader'] = data_loader.describe_default("%s/test_loader" % prefix,
                                                              suffix, default, dataset,
                                                              False, batch_size, num_thread,
                                                              crop_size, is_validation=False)
        default['test_mode'] = test_mode
        default['use_tqdm'] = use_tqdm
        if logger is None:
            default['logger'] = Blueprint("%s/logger" % prefix, suffix, default)
        else:
            # make_module will return None
            default['logger'] = logger.describe_default("%s/logger" % prefix, suffix,
                                                        default)

        default['optimizer_maker'] = optimizer_maker.describe_default("%s/optimizer_maker" % prefix,
                                                                      suffix, default, optimizer_type,
                                                                      optimizer_parameter_picker,
                                                                      momentum, weight_decay)
        default['retain_graph'] = False
        if criterion != ScopedCrossEntropyLoss:
            default['retain_graph'] = True

        if net_blueprint is not None:
            default['net'] = net_blueprint
            default['net']['parent'] = default
        else:
            default['net'] = net_module.describe_default("%s/Network" % prefix, suffix, parent=parent,
                                                         skeleton=skeleton, group_depths=group_depths,
                                                         num_classes=num_classes, depth=depth,
                                                         width=width, block_depth=block_depth,
                                                         block_module=block_module, conv_module=conv_module,
                                                         bn_module=bn_module, linear_module=linear_module,
                                                         act_module=act_module, kernel_size=kernel_size,
                                                         padding=padding, input_shape=input_shape,
                                                         dilation=dilation, groups=groups, bias=bias,
                                                         callback=callback, drop_p=drop_p, dropout_p=dropout_p,
                                                         residual=residual, conv_kwargs=conv_kwargs,
                                                         bn_kwargs=bn_kwargs, act_kwargs=act_kwargs,
                                                         unit_module=unit_module, group_module=group_module,
                                                         fractal_depth=fractal_depth, shortcut_index=shortcut_index,
                                                         dense_unit_module=dense_unit_module,
                                                         no_weights=no_weights, head_kernel=head_kernel,
                                                         head_stride=head_stride, head_padding=head_padding,
                                                         head_pool_kernel=head_pool_kernel,
                                                         head_pool_stride=head_pool_stride,
                                                         head_pool_padding=head_pool_padding,
                                                         head_modules=head_modules)

        default['kwargs'] = {'blueprint': default}
        return default
