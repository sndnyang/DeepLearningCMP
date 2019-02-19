import random
from math import *

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as functions

from ..ModelUtil import Flatten, UnFlatten


class GaussianNoise(nn.Module):
    """Gaussian noise regularizer.

    Args:
        sigma (float, optional): relative standard deviation used to generate the
            noise. Relative means that it will be multiplied by the magnitude of
            the value your are adding the noise to. This means that sigma can be
            the same regardless of the scale of the vector.
        is_relative_detach (bool, optional): whether to detach the variable before
            computing the scale of the noise. If `False` then the scale of the noise
            won't be seen as a constant but something to optimize: this will bias the
            network to generate vectors with smaller values.
    """

    def __init__(self, sigma=0.1, is_relative_detach=True):
        super().__init__()
        self.sigma = sigma
        self.is_relative_detach = is_relative_detach
        self.noise = torch.tensor(0)

    def forward(self, x):
        if self.training and self.sigma != 0:
            scale = self.sigma * x.detach() if self.is_relative_detach else self.sigma * x
            sampled_noise = self.noise.repeat(*x.size()).normal_() * scale
            x = x + sampled_noise
        return x


def gaussian_mixture(batchsize, ndim, num_labels, label_indices=None):
    if ndim % 2 != 0:
        raise Exception("ndim must be a multiple of 2.")

    def sample(x, y, label, num_labels):
        shift = 1.4
        r = 2.0 * np.pi / float(num_labels) * float(label)
        new_x = x * cos(r) - y * sin(r)
        new_y = x * sin(r) + y * cos(r)
        new_x += shift * cos(r)
        new_y += shift * sin(r)
        return np.array([new_x, new_y]).reshape((2,))

    x_var = 0.5
    y_var = 0.05
    x = np.random.normal(0, x_var, (batchsize, ndim // 2))
    y = np.random.normal(0, y_var, (batchsize, ndim // 2))
    z = np.empty((batchsize, ndim), dtype=np.float32)
    for batch in range(batchsize):
        for zi in range(ndim // 2):
            if label_indices is None:
                z[batch, zi * 2:zi * 2 + 2] = sample(x[batch, zi], y[batch, zi], random.randint(0, num_labels - 1), num_labels)
            else:
                z[batch, zi*2:zi*2+2] = sample(x[batch, zi], y[batch, zi], label_indices[batch], num_labels)
    return z


class AAE_semi_reg(nn.Module):
    def __init__(self, input_shape=(1, 28, 28), z_dim=2):
        super(AAE_semi_reg, self).__init__()
        x_dim = int(np.prod(input_shape))
        y_dim = 11
        h_dim = 1000

        self.dim_x = x_dim
        self.dim_y = y_dim
        self.dim_z = z_dim
        self.h_dim = h_dim

        self.encoder = nn.Sequential(
            Flatten(),
            # nn.BatchNormalization(h_dim),
            nn.Linear(x_dim, h_dim),
            nn.ReLU(),
            # nn.BatchNormalization(h_dim),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, z_dim),
        )

        self.decoder = nn.Sequential(
            # nn.BatchNormalization(h_dim),
            nn.Linear(z_dim, h_dim),
            nn.ReLU(),
            # nn.BatchNormalization(h_dim),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, x_dim),
            nn.Tanh(),
        )

        self.discriminator = nn.Sequential(
            # nn.BatchNormalization(nh_dim),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            # nn.BatchNormalization(nh_dim),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            # nn.BatchNormalization(nh_dim),
            nn.Linear(h_dim, 2),
        )

        self.merge_y = nn.Linear(y_dim, h_dim, bias=False)
        self.merge_z = nn.Linear(z_dim, h_dim, bias=False)
        self.merge_bias = nn.Parameter(torch.zeros((h_dim,)))

    def encode_x_z(self, x):
        return self.encoder(x)

    def discriminate(self, y, z, apply_softmax=False, device="cpu"):

        scale = 0.3 * z.detach()
        sampled_noise = torch.empty(z.shape).normal_().to(device) * scale
        z = z + sampled_noise
        merge = self.merge_y(y) + self.merge_z(z) + self.merge_bias

        logit = self.discriminator(merge)
        if apply_softmax:
            return functions.softmax(logit)
        return logit

    def decode_z_x(self, z):
        return self.decoder(z)


class AAE_back(nn.Module):
    def __init__(self, input_shape=(1, 28, 28), z_dim=2):
        super(AAE_back, self).__init__()
        x_dim = int(np.prod(input_shape))
        y_dim = 11
        h_dim = 1000

        self.dim_x = x_dim
        self.dim_y = y_dim
        self.dim_z = z_dim
        self.h_dim = h_dim

        self.encoder = nn.Sequential(
            Flatten(),
            # nn.BatchNormalization(h_dim),
            nn.Linear(x_dim, h_dim),
            nn.ReLU(),
            # nn.BatchNormalization(h_dim),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, z_dim),
        )

        self.decoder = nn.Sequential(
            # nn.BatchNormalization(h_dim),
            nn.Linear(z_dim, h_dim),
            nn.ReLU(),
            # nn.BatchNormalization(h_dim),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, x_dim),
            nn.Tanh(),
        )

        self.discriminator_cat = nn.Sequential(
            # nn.BatchNormalization(nh_dim),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            # nn.BatchNormalization(nh_dim),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            # nn.BatchNormalization(nh_dim),
            nn.Linear(h_dim, 1),
            nn.Sigmoid()
        )

        self.discriminator_gauss = nn.Sequential(
            # nn.BatchNormalization(nh_dim),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            # nn.BatchNormalization(nh_dim),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            # nn.BatchNormalization(nh_dim),
            nn.Linear(h_dim, 1),
            nn.Sigmoid()
        )
        self.merge_y = nn.Linear(y_dim, h_dim)
        self.merge_z = nn.Linear(z_dim, h_dim)

    def encode_x_z(self, x):
        return self.encoder(x)

    def discriminate(self, y, z, apply_softmax=False, device="cpu"):

        scale = 0.3 * z.detach()
        sampled_noise = torch.empty(z.shape).normal_().to(device) * scale
        z = z + sampled_noise
        categorical = self.discriminator_cat(self.merge_y(y))
        gaussian = self.discriminator_gauss(self.merge_z(z))
        # logit = self.discriminator(merge)
        logit = torch.cat([])
        if apply_softmax:
            return functions.softmax(logit)
        return logit

    def decode_z_x(self, z):
        return self.decoder(z)
