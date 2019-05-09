import torch.nn as nn
import torch.nn.functional as nfunc

from .utils import call_bn


class MLPSemi(nn.Module):
    # specific MLP layers and sizes for MNIST semi supervised learning
    def __init__(self, affine=False, top_bn=True):
        super(MLPSemi, self).__init__()
        self.top_bn = top_bn
        self.input_len = 1 * 28 * 28
        self.fc1 = nn.Linear(self.input_len, 1200, bias=False)
        self.fc2 = nn.Linear(1200, 1200, bias=False)
        self.fc3 = nn.Linear(1200, 10, bias=not self.top_bn)
        self.bn_fc1 = nn.BatchNorm1d(1200, affine=affine)
        self.bn_fc2 = nn.BatchNorm1d(1200, affine=affine)

        if self.top_bn:
            self.bn_fc3 = nn.BatchNorm1d(10, affine=affine)

    def forward(self, x, update_batch_stats=True):
        h = nfunc.relu(call_bn(self.bn_fc1, self.fc1(x.view(-1, self.input_len)), update_batch_stats))
        h = nfunc.relu(call_bn(self.bn_fc2, self.fc2(h), update_batch_stats))
        if self.top_bn:
            h = call_bn(self.bn_fc3, self.fc3(h), update_batch_stats)
        else:
            h = self.fc3(h)
        logits = h
        return logits


class MLP(nn.Module):
    # general MLP for VAT semi-supervised learning on MNIST
    def __init__(self, layer_sizes, top_bn=False):
        super(MLP, self).__init__()
        layers = []
        self.top_bn = top_bn
        for i, (m, n) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
            affine_layer = nn.Linear(m, n, bias=False if i < len(layer_sizes) - 1 or self.top_bn else True)
            layers.append(affine_layer)
            if i < len(layer_sizes) - 2:
                bn = nn.BatchNorm1d(n)
                layers.append(bn)
                layers.append(nn.ReLU(inplace=True))
        if self.top_bn:
            layers.append(nn.BatchNorm1d(layer_sizes[-1]))
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


class MLPSyn(nn.Module):
    def __init__(self):
        super(MLPSyn, self).__init__()
        self.l1 = nn.Linear(100, 100)
        self.l2 = nn.Linear(100, 2)

    def forward(self, h):
        h = self.l1(h)
        h = nfunc.relu(h)
        h = self.l2(h)
        return h


"""
[url](https://github.com/takerum/vat/blob/master/models/fnn_mnist_sup.py)

or

[url](https://github.com/takerum/vat/blob/master/models/fnn_mnist_semisup.py)

###VAT for semi-supervised learning on MNIST dataset (with 100 labeled samples)

python train_mnist_semisup.py --cost_type=VAT_finite_diff --epsilon=0.3 --layer_sizes=784-1200-1200-10 --num_labeled_samples=100

"""
