import torch

class sharedAdam(torch.optim.Adam):

    '''
    Extends the PyTorch Optimizer to share gradients across processes
    '''

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        super(sharedAdam, self).__init__(params, lr, betas, eps, weight_decay)
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                state["shared_steps"], state["step"] = torch.zeros(1).share_memory_(), 0
                state['exp_avg'] = p.data.new().resize_as_(p.data).zero_().share_memory_()
                state['exp_avg_sq'] = p.data.new().resize_as_(p.data).zero_().share_memory_()

    def step(self, closure=None):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None: continue
                self.state[p]['shared_steps'] += 1
                self.state[p]['step'] = self.state[p]['shared_steps'][0] - 1

        super.step(closure)
    


