import torch
from torch.optim.optimizer import Optimizer
import math

class AdamHD(Optimizer):
    """Implements AdamW algorithm with hypergradient descent for online learning rate, weight decay,
    beta1 adaptation, and GrokFast gradient modification.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining parameter groups
        lr (float, optional): initial learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients for computing running averages of gradient
            and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) applied outside of the gradient update (default: 0)
        hypergrad_lr (float, optional): hypergradient learning rate for online tuning of the learning rate (default: 1e-8)
        hypergrad_beta_lr (float, optional): hypergradient learning rate for beta1 adaptation (default: 0)
        grokfast (bool, optional): whether to enable GrokFast gradient modification (default: True)
        grokfast_alpha (float, optional): EMA decay rate for GrokFast (default: 0.98)
        grokfast_lamb (float, optional): scaling factor for adding EMA gradient in GrokFast (default: 2.0)
        grokfast_after_step (int, optional): number of steps after which to start GrokFast (default: 0)
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0,
                 hypergrad_lr=1e-7, hypergrad_beta_lr=0,
                 grokfast=True, grokfast_alpha=0.98, grokfast_lamb=2.0, grokfast_after_step=0):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                        hypergrad_lr=hypergrad_lr, hypergrad_beta_lr=hypergrad_beta_lr,
                        grokfast=grokfast, grokfast_alpha=grokfast_alpha,
                        grokfast_lamb=grokfast_lamb, grokfast_after_step=grokfast_after_step)
        super(AdamHD, self).__init__(params, defaults)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            hypergrad_lr = group['hypergrad_lr']
            hypergrad_beta_lr = group['hypergrad_beta_lr']
            grokfast = group['grokfast']
            grokfast_alpha = group['grokfast_alpha']
            grokfast_lamb = group['grokfast_lamb']
            grokfast_after_step = group['grokfast_after_step']

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('AdamHD does not support sparse gradients')

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                # Hypergradient descent adjustment for learning rate and beta1 (after step 1)
                if state['step'] > 1:
                    prev_bias_correction1 = 1 - 0.99 ** (state['step'] - 1)
                    prev_bias_correction2 = 1 - 0.999 ** (state['step'] - 1)
                    # Hypergradient for learning rate
                    h_lr = torch.dot(grad.view(-1), torch.div(exp_avg, exp_avg_sq.sqrt().add(group['eps'])).view(-1))
                    h_lr *= math.sqrt(prev_bias_correction2) / prev_bias_correction1
                    # Update learning rate
                    group['lr'] += hypergrad_lr * (h_lr * (max(1, 100 - state['step'])))  # Scaling factor as per original implementation
                    group['lr'] = min(max(group['lr'], 0), 0.01)

                    # Hypergradient for beta1
                    if hypergrad_beta_lr > 0:
                        h_beta1 = torch.dot(exp_avg.view(-1), grad.view(-1))
                        beta1 += hypergrad_beta_lr * h_beta1
                        beta1 = max(min(beta1, 0.999), 0)  # Ensure beta1 is within [0, 0.999]
                        group['betas'] = (beta1, beta2)

                # Implement GrokFast
                should_grokfast = grokfast and state['step'] > grokfast_after_step and grokfast_lamb > 0
                if should_grokfast:
                    if 'grok_exp_avg' not in state:
                        # Initialize GrokFast EMA of gradient
                        state['grok_exp_avg'] = grad.clone()
                    grok_exp_avg = state['grok_exp_avg']
                    # Update GrokFast EMA
                    grok_exp_avg.lerp_(grad, 1 - grokfast_alpha)
                    # Modify gradient
                    grad.add_(grok_exp_avg, alpha=grokfast_lamb)

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                denom = exp_avg_sq.sqrt().add_(group['eps'])

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1

                # Apply weight decay
                if group['weight_decay'] != 0:
                    p.data.add_(p.data, alpha=-group['weight_decay'] * group['lr'])

                # Update parameter
                p.data.addcdiv_(exp_avg, denom, value=-step_size)

        return loss
