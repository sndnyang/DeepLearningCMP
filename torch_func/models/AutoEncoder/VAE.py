import torch
import numpy as np
import torch.nn as nn

from ..ModelUtil import Flatten, UnFlatten


class VAE(nn.Module):
    def __init__(self, input_shape=(1, 28, 28), z_dim=2):
        super(VAE, self).__init__()
        self.device = None
        self.encoder = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2),
            nn.ReLU(),
            Flatten()
        )
        input_size = self._get_conv_output_size(input_shape)
        self.dense = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )
        self.fc1 = nn.Linear(128, z_dim)
        self.fc2 = nn.Linear(128, z_dim)
        self.fc3 = nn.Linear(z_dim, 128)

        self.decoder = nn.Sequential(
            nn.Linear(128, input_size),
            nn.ReLU(),
            UnFlatten(),
            nn.ConvTranspose2d(input_size, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=6, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(32, input_shape[0], kernel_size=6, stride=2),
            nn.Sigmoid(),
        )

    def reparameterize(self, mu, log_var):
        std = log_var.mul(0.5).exp_()
        # return torch.normal(mu, std)
        esp = torch.randn(*mu.size()).to(self.device)
        z = mu + std * esp
        return z

    def bottleneck(self, h):
        mu, log_var = self.fc1(h), self.fc2(h)
        z = self.reparameterize(mu, log_var)
        return z, mu, log_var

    def encode(self, x):
        h = self.encoder(x)
        h = self.dense(h)
        z, mu, log_var = self.bottleneck(h)
        return z, mu, log_var

    def decode(self, z):
        z = self.fc3(z)
        z = self.decoder(z)
        return z

    def forward(self, x):
        z, mu, log_var = self.encode(x)
        z = self.decode(z)
        return z, mu, log_var

    def _get_conv_output_size(self, shape):
        inputs = torch.rand(1, *shape)
        output_feat = self.encoder(inputs)
        n_size = output_feat.data.view(1, -1).size(1)
        return n_size


class VAELinear(VAE):
    def __init__(self, input_shape=(1, 28, 28), z_dim=2):
        super(VAELinear, self).__init__()
        x_dim = int(np.prod(input_shape))
        y_dim = 11
        h_dim = 1000

        self.dim_x = x_dim
        self.dim_y = y_dim
        self.dim_z = z_dim
        self.dim_h = h_dim

        self.encoder = nn.Sequential(
            Flatten(),
            nn.BatchNormalization(h_dim),
            nn.Linear(x_dim, h_dim),
            nn.ReLU(),
            nn.BatchNormalization(h_dim),
            nn.Linear(h_dim, h_dim)
        )

        input_size = self._get_conv_output_size(input_shape)
        self.fc1 = nn.Linear(input_size, z_dim)
        self.fc2 = nn.Linear(input_size, z_dim)
        self.fc3 = nn.Linear(z_dim, input_size)

        self.decoder = nn.Sequential(
            nn.BatchNormalization(h_dim),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.BatchNormalization(h_dim),
            nn.Linear(h_dim, x_dim),
            UnFlatten()
        )
