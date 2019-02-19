import numpy as np
import torch.nn as nn
import torch.nn.functional as nfunc


class MLP(nn.Module):
    def __init__(self, layer_sizes):
        super(MLP, self).__init__()
        layers = []
        for i, (m, n) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
            affine_layer = nn.Linear(m, n)
            layers.append(affine_layer)
            if i < len(layer_sizes) - 2:
                bn = nn.BatchNorm1d(n, eps=2e-5)
                layers.append(bn)
                layers.append(nn.ReLU(inplace=True))
        self.net = nn.Sequential(*layers)

    def forward(self, x, update_batch_stats=True):
        x = x.view(x.shape[0], -1)
        for layer in self.net:
            if layer.__class__.__name__.find("BatchNorm") == 0:
                x = call_bn(layer, x, update_batch_stats)
            else:
                x = layer(x)
        output = x
        return output


def call_bn(bn, x, update_batch_stats=True):
    if bn.training is False:
        return bn(x)
    elif not update_batch_stats:
        return nfunc.batch_norm(x, None, None, bn.weight, bn.bias, True, bn.momentum, bn.eps)
    else:
        return bn(x)


class FNN_syn(nn.Module):
    def __init__(self):
        super(FNN_syn, self).__init__()
        self.l1 = nn.Linear(100, 100)
        self.l2 = nn.Linear(100, 2)

    def forward(self, h):
        h = self.l1(h)
        h = nfunc.relu(h)
        h = self.l2(h)
        return h


# Following this repository
# https://github.com/musyoku/vat
class FullyNet(nn.Module):
    def __init__(self, n_class, n_ch, res):
        super(FullyNet, self).__init__()
        self.input_len = n_ch * res * res
        self.fc1 = nn.Linear(self.input_len, 1200)
        self.fc2 = nn.Linear(1200, 600)
        self.fc3 = nn.Linear(600, n_class)

        self.bn_fc1 = nn.BatchNorm1d(1200, eps=2e-5)
        self.bn_fc2 = nn.BatchNorm1d(600, eps=2e-5)

    def forward(self, x, update_batch_stats=True):
        h = nfunc.relu(call_bn(self.bn_fc1, self.fc1(x.view(-1, self.input_len)), update_batch_stats))
        h = nfunc.relu(call_bn(self.bn_fc2, self.fc2(h), update_batch_stats))
        # h = nfunc.relu(self.bn_fc1(self.fc1(x.view(-1, self.input_len))))
        # h = nfunc.relu(self.bn_fc2(self.fc2(h)))
        # h = nfunc.relu(self.fc1(x.view(-1, self.input_len)))
        # h = nfunc.relu(self.fc2(h))
        return self.fc3(h)


def mlp(pretrained=False, input_shape=None, num_classes=10, **kwargs):
    layer_sizes = kwargs.get("layer_sizes", [1200, 600])
    # layer_sizes = kwargs.get("layer_sizes", [1000, 1000, 1000])
    if input_shape is None:
        input_shape = [1, 28, 28]
    layer_sizes = [int(np.prod(input_shape))] + layer_sizes + [num_classes]
    return MLP(layer_sizes)


def fullynet(pretrained=False, input_shape=None, num_classes=10, **kwargs):
    if input_shape is None:
        input_shape = [1, 28, 28]
    return FullyNet(num_classes, input_shape[0], input_shape[1])
