import torch as pt

from .geometric_algebra import custom_norm


class MomentumLayerNormalization(pt.nn.Module):
    """Exponential decay normalization.

    Calculates a running average of the L2 norm and scales inputs to
    have length (over the last axis) 1, on average.

    :param momentum: Momentum of moving average, from 0 to 1
    :param epsilon: Minimum norm for normalization scaling factor

    """

    def __init__(self, momentum=0.99, epsilon=1e-7):
        super().__init__()
        self.register_buffer('momentum', pt.as_tensor(momentum))
        self.register_buffer('epsilon', pt.as_tensor(epsilon))
        self.register_buffer('norm', pt.ones(1))

    def forward(self, x):
        if self.training:
            norm = custom_norm(x)
            norm = pt.mean(norm)
            self.norm[:] = norm.detach()

        return x/pt.maximum(self.norm, self.epsilon).detach()
