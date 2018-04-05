class EpochEngine(object):
    def __init__(self):
        self.hooks = {}
        self.state = {}

    def hook(self, name, state):
        if name in self.hooks:
            self.hooks[name](state)

    def train_one_epoch(self):
        self.hook('on_start_epoch', self.state)
        for sample in self.state['iterator']:
            self.state['sample'] = sample
            self.hook('on_sample', self.state)

            def closure():
                loss, output = self.state['network'](self.state['sample'])
                self.state['output'] = output
                self.state['loss'] = loss
                loss.backward()
                self.hook('on_forward', self.state)
                # to free memory in save_for_backward
                self.state['output'] = None
                self.state['loss'] = None
                return loss

            self.state['optimizer'].zero_grad()
            self.state['optimizer'].step(closure)
            self.hook('on_update', self.state)
            self.state['t'] += 1

        self.state['epoch'] += 1
        self.hook('on_end_epoch', self.state)
        return self.state

    def set_state(self, network, iterator, maxepoch, optimizer, epoch=0, t=0, train=True):
        self.state = {
                'network': network,
                'iterator': iterator,
                'maxepoch': maxepoch,
                'optimizer': optimizer,
                'epoch': epoch,
                't': t,
                'train': train,
                }

    def train(self, network, iterator, maxepoch, optimizer):
        self.set_state(network, iterator, maxepoch, optimizer)
        self.hook('on_start', self.state)
        while self.state['epoch'] < self.state['maxepoch']:
            self.train_one_epoch()
        self.hook('on_end', self.state)
        return self.state

    def test(self, network, iterator):
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
