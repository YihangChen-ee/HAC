import torch
import torch.nn as nn
import torch.nn.functional as nnf
import numpy as np
from torch.distributions.uniform import Uniform
from utils.encodings import use_clamp

class Entropy_gaussian_clamp(nn.Module):
    def __init__(self, Q=1):
        super(Entropy_gaussian_clamp, self).__init__()
        self.Q = Q
    def forward(self, x, mean, scale, Q=None):
        if Q is None:
            Q = self.Q
        if use_clamp:
            x_mean = x.mean()
            x_min = x_mean - 15_000 * Q
            x_max = x_mean + 15_000 * Q
            x = torch.clamp(x, min=x_min.detach(), max=x_max.detach())
        scale = torch.clamp(scale, min=1e-9)
        m1 = torch.distributions.normal.Normal(mean, scale)
        lower = m1.cdf(x - 0.5*Q)
        upper = m1.cdf(x + 0.5*Q)
        likelihood = torch.abs(upper - lower)
        likelihood = Low_bound.apply(likelihood)
        bits = -torch.log2(likelihood)
        return bits


class Entropy_gaussian(nn.Module):
    def __init__(self, Q=1):
        super(Entropy_gaussian, self).__init__()
        self.Q = Q
    def forward(self, x, mean, scale, Q=None, x_mean=None):
        if Q is None:
            Q = self.Q
        if use_clamp:
            if x_mean is None:
                x_mean = x.mean()
            x_min = x_mean - 15_000 * Q
            x_max = x_mean + 15_000 * Q
            x = torch.clamp(x, min=x_min.detach(), max=x_max.detach())
        scale = torch.clamp(scale, min=1e-9)
        m1 = torch.distributions.normal.Normal(mean, scale)
        lower = m1.cdf(x - 0.5*Q)
        upper = m1.cdf(x + 0.5*Q)
        likelihood = torch.abs(upper - lower)
        likelihood = Low_bound.apply(likelihood)
        bits = -torch.log2(likelihood)
        return bits


class Entropy_bernoulli(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x, p):
        # p = torch.sigmoid(p)
        p = torch.clamp(p, min=1e-6, max=1 - 1e-6)
        pos_mask = (1 + x) / 2.0  # 1 -> 1, -1 -> 0
        neg_mask = (1 - x) / 2.0  # -1 -> 1, 1 -> 0
        pos_prob = p
        neg_prob = 1 - p
        param_bit = -torch.log2(pos_prob) * pos_mask + -torch.log2(neg_prob) * neg_mask
        return param_bit


class Entropy_factorized(nn.Module):
    def __init__(self, channel=32, init_scale=10, filters=(3, 3, 3), likelihood_bound=1e-6,
                 tail_mass=1e-9, optimize_integer_offset=True, Q=1):
        super(Entropy_factorized, self).__init__()
        self.filters = tuple(int(t) for t in filters)
        self.init_scale = float(init_scale)
        self.likelihood_bound = float(likelihood_bound)
        self.tail_mass = float(tail_mass)
        self.optimize_integer_offset = bool(optimize_integer_offset)
        self.Q = Q
        if not 0 < self.tail_mass < 1:
            raise ValueError(
                "`tail_mass` must be between 0 and 1")
        filters = (1,) + self.filters + (1,)
        scale = self.init_scale ** (1.0 / (len(self.filters) + 1))
        self._matrices = nn.ParameterList([])
        self._bias = nn.ParameterList([])
        self._factor = nn.ParameterList([])
        for i in range(len(self.filters) + 1):
            init = np.log(np.expm1(1.0 / scale / filters[i + 1]))
            self.matrix = nn.Parameter(torch.FloatTensor(
                channel, filters[i + 1], filters[i]))
            self.matrix.data.fill_(init)
            self._matrices.append(self.matrix)
            self.bias = nn.Parameter(
                torch.FloatTensor(channel, filters[i + 1], 1))
            noise = np.random.uniform(-0.5, 0.5, self.bias.size())
            noise = torch.FloatTensor(noise)
            self.bias.data.copy_(noise)
            self._bias.append(self.bias)
            if i < len(self.filters):
                self.factor = nn.Parameter(
                    torch.FloatTensor(channel, filters[i + 1], 1))
                self.factor.data.fill_(0.0)
                self._factor.append(self.factor)

    def _logits_cumulative(self, logits, stop_gradient):
        for i in range(len(self.filters) + 1):
            matrix = nnf.softplus(self._matrices[i])
            if stop_gradient:
                matrix = matrix.detach()
            # print('dqnwdnqwdqwdqwf:', matrix.shape, logits.shape)
            logits = torch.matmul(matrix, logits)
            bias = self._bias[i]
            if stop_gradient:
                bias = bias.detach()
            logits += bias
            if i < len(self._factor):
                factor = nnf.tanh(self._factor[i])
                if stop_gradient:
                    factor = factor.detach()
                logits += factor * nnf.tanh(logits)
        return logits

    def forward(self, x, Q=None):
        # x: [N, C], quantized
        if Q is None:
            Q = self.Q
        else:
            Q = Q.permute(1, 0).contiguous()
        x = x.permute(1, 0).contiguous()  # [C, N]
        # print('dqwdqwdqwdqwfqwf:', x.shape, Q.shape)
        lower = self._logits_cumulative(x - 0.5*(1/Q), stop_gradient=False)
        upper = self._logits_cumulative(x + 0.5*(1/Q), stop_gradient=False)
        sign = -torch.sign(torch.add(lower, upper))
        sign = sign.detach()
        likelihood = torch.abs(
            nnf.sigmoid(sign * upper) - nnf.sigmoid(sign * lower))
        likelihood = Low_bound.apply(likelihood)
        bits = -torch.log2(likelihood)  # [C, N]
        bits = bits.permute(1, 0).contiguous()
        return bits


class Low_bound(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        x = torch.clamp(x, min=1e-6)
        return x

    @staticmethod
    def backward(ctx, g):
        x, = ctx.saved_tensors
        grad1 = g.clone()
        grad1[x < 1e-6] = 0
        pass_through_if = np.logical_or(
            x.cpu().numpy() >= 1e-6, g.cpu().numpy() < 0.0)
        t = torch.Tensor(pass_through_if+0.0).cuda()
        return grad1 * t


class UniverseQuant(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        #b = np.random.uniform(-1,1)
        b = 0
        uniform_distribution = Uniform(-0.5*torch.ones(x.size())
                                       * (2**b), 0.5*torch.ones(x.size())*(2**b)).sample().cuda()
        return torch.round(x+uniform_distribution)-uniform_distribution

    @staticmethod
    def backward(ctx, g):

        return g
