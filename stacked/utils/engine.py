import torch
import torchnet as tnt
from stacked.meta.scope import generate_random_scope
from stacked.utils import common
from logging import warning
import numpy as np
from tqdm import tqdm


def log(log_func, msg):
    if common.DEBUG_ENGINE:
        log_func("stacked.utils.engine: %s" % msg)


def get_num_parameters(net):
    return sum(p.numel() for p in net.parameters())


class EpochEngine(object):
    def __init__(self):
        self.hooks = {}
        self.state = {}
        self.retain_graph = False

    def hook(self, name, state):
        if name in self.hooks:
            self.hooks[name](state)

    def train_n_samples(self, n):
        torch.autograd.set_detect_anomaly(True)
        for i, sample in enumerate(self.state['iterator']):
            self.state['sample'] = sample
            self.hook('on_sample', self.state)
            retain_graph = self.retain_graph

            def closure():
                loss, output = self.state['network'](self.state['sample'])
                self.state['output'] = output
                self.state['loss'] = loss
                loss.backward(retain_graph=retain_graph)
                self.hook('on_forward', self.state)

                # to free memory in save_for_backward
                self.state['output'] = None
                self.state['loss'] = None
                return loss

            self.state['optimizer'].zero_grad()
            self.state['optimizer'].step(closure)
            self.hook('on_update', self.state)
            self.state['t'] += 1

            if i >= n:
                break

    def start_epoch(self):
        self.hook('on_start_epoch', self.state)
        common.TRAIN = True

    def end_epoch(self):
        self.state['epoch'] += 1
        common.CURRENT_EPOCH = self.state['epoch']
        self.hook('on_end_epoch', self.state)
        return self.state

    def train_one_epoch(self, n=np.inf):
        self.start_epoch()
        self.train_n_samples(n)
        return self.end_epoch()

    def set_state(self, network, iterator, maxepoch, optimizer,
                  epoch=0, t=0, train=True, score=np.inf):
        self.state = {
                'network': network,
                'iterator': iterator,
                'maxepoch': maxepoch,
                'optimizer': optimizer,
                'epoch': epoch,
                't': t,
                'train': train,
                'score': score
                }

    def train(self, network, iterator, maxepoch, optimizer):
        self.set_state(network, iterator, maxepoch, optimizer)
        self.hook('on_start', self.state)
        while self.state['epoch'] < self.state['maxepoch']:
            self.train_one_epoch()
        self.hook('on_end', self.state)
        return self.state

    def test(self, network, iterator):
        common.TRAIN = False
        state = {
            'network': network,
            'iterator': iterator,
            't': 0,
            'train': False,
            }

        self.hook('on_start', state)
        for sample in state['iterator']:
            state['sample'] = sample
            self.hook('on_sample', state)

            def closure():
                loss, output = state['network'](state['sample'])
                state['output'] = output
                state['loss'] = loss
                self.hook('on_forward', state)

                # to free memory in save_for_backward
                state['output'] = None
                state['loss'] = None

            closure()
            state['t'] += 1
        self.hook('on_end', state)
        return state


class EngineEventHooks(object):
    def __init__(self, engine=None, train_loader=None,
                 test_loader=None, net=None, net_runner=None,
                 make_optimizer=None, learning_rate=0.1,
                 lr_decay_ratio=0.2, lr_drop_epochs=None,
                 logger=None, train_id=None, epoch=0,
                 average_loss_meter=None, accuracy_meter=None,
                 train_timer=None, test_timer=None, use_tqdm=False):

        assert(engine is not None)
        assert (train_loader is not None)
        assert (test_loader is not None)
        assert (net is not None)
        assert (net_runner is not None)
        assert (make_optimizer is not None)

        num_parameters = get_num_parameters(net)

        log(warning, '\nTotal number of parameters in model with id: {} is: {}'
            .format(train_id, num_parameters))

        # defaults
        if average_loss_meter is None:
            average_loss_meter = tnt.meter.AverageValueMeter()

        if accuracy_meter is None:
            accuracy_meter = tnt.meter.ClassErrorMeter(accuracy=True)

        if train_timer is None:
            train_timer = tnt.meter.TimeMeter('s')

        if test_timer is None:
            test_timer = tnt.meter.TimeMeter('s')

        if train_id is None:
            train_id = generate_random_scope()

        if logger is None:
            def print_log(_state, _stats):
                log(warning, '==> id: %s (%d/%d), test_acc: \33[91m%.2f\033[0m' %
                    (train_id, _state['epoch'], _state['maxepoch'],
                     _stats['test_acc']))
                log(warning, _stats)

            logger = print_log

        # change epoch and lr in the class definitions
        g = {'epoch': epoch, 'lr': learning_rate}

        def on_sample(state):
            state['sample'].append(state['train'])

        def on_forward(state):
            state['sample'].append(state['train'])
            accuracy_meter.add(state['output'].data,
                               torch.LongTensor(state['sample'][1]))
            average_loss_meter.add(state['loss'].item())

        def on_start(state):
            state['epoch'] = g['epoch']

        def on_start_epoch(state):
            accuracy_meter.reset()
            average_loss_meter.reset()
            train_timer.reset()
            if use_tqdm:
                state['iterator'] = tqdm(train_loader)
            else:
                state['iterator'] = train_loader

            g['epoch'] = state['epoch'] + 1
            if g['epoch'] in lr_drop_epochs:
                g['lr'] = state['optimizer'].param_groups[0]['lr']
                state['optimizer'] = make_optimizer(net, g['lr'] * lr_decay_ratio)

        def on_end_epoch(state):
            train_loss = average_loss_meter.value()
            train_acc = accuracy_meter.value()
            train_time = train_timer.value()

            average_loss_meter.reset()
            accuracy_meter.reset()
            test_timer.reset()

            net.eval()
            engine.test(net_runner, test_loader)
            net.train()

            test_acc = accuracy_meter.value()[0]
            test_loss = average_loss_meter.value()[0]
            state['score'] = test_loss
            logger(state, {
                "train_loss": train_loss[0],
                "train_acc": train_acc[0],
                "test_loss": test_loss,
                "test_acc": test_acc,
                "epoch": state['epoch'],
                "train_time": train_time,
                "test_time": test_timer.value(),
            })

        self.on_start = on_start
        self.on_sample = on_sample
        self.on_forward = on_forward
        self.on_start_epoch = on_start_epoch
        self.on_end_epoch = on_end_epoch
