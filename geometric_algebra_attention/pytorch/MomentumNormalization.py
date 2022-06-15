import torch as pt

class MomentumNormalization(pt.nn.Module):
    """Exponential decay normalization.

    Computes the mean and standard deviation all axes but the last and
    normalizes values to have mean 0 and variance 1; suitable for
    normalizing a vector of real-valued quantities with differing
    units.

    :param n_dim: Last dimension of the layer input
    :param momentum: Momentum of moving average, from 0 to 1

    """

    def __init__(self, n_dim, momentum=.99):
        super().__init__()
        self.n_dim = n_dim
        self.register_buffer('momentum', pt.as_tensor(momentum))
        self.register_buffer('mu', pt.zeros(n_dim))
        self.register_buffer('sigma', pt.ones(n_dim))

    def forward(self, x):
        if self.training:
            axes = tuple(range(x.ndim - 1))
            mu_calc = pt.mean(x, axes, keepdim=False)
            sigma_calc = pt.std(x, axes, keepdim=False, unbiased=False)

            new_mu = self.momentum*self.mu + (1 - self.momentum)*mu_calc
            new_sigma = self.momentum*self.sigma + (1 - self.momentum)*sigma_calc

            self.mu[:] = new_mu.detach()
            self.sigma[:] = new_sigma.detach()

        sigma = pt.maximum(self.sigma, pt.as_tensor(1e-7))

        return (x - self.mu.detach())/sigma.detach()
