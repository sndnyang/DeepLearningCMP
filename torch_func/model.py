import torch.nn as nn
import torch.nn.functional as nfunc


def call_bn(bn, x, update_batch_stats=True):
    if bn.training is False:
        return bn(x)
    elif not update_batch_stats:
        return nfunc.batch_norm(x, None, None, bn.weight, bn.bias, True, bn.momentum, bn.eps)
    else:
        return bn(x)


class MLP(nn.Module):
    # specific MLP layers and sizes for MNIST semi supervised learning
    def __init__(self, affine=False, top_bn=True):
        super(MLP, self).__init__()
        self.input_len = 1 * 28 * 28
        self.fc1 = nn.Linear(self.input_len, 1200)
        self.fc2 = nn.Linear(1200, 1200)
        self.fc3 = nn.Linear(1200, 10)
        self.bn_fc1 = nn.BatchNorm1d(1200, affine=affine)
        self.bn_fc2 = nn.BatchNorm1d(1200, affine=affine)

        self.top_bn = top_bn
        if top_bn:
            self.bn_fc3 = nn.BatchNorm1d(10, affine=affine)

    def forward(self, x, update_batch_stats=True, return_h=False):
        # return_h is for hidden output h may be used in large margin
        endpoints = {}
        h = nfunc.relu(call_bn(self.bn_fc1, self.fc1(x.view(-1, self.input_len)), update_batch_stats))
        endpoints["fc_layer0"] = h
        h = nfunc.relu(call_bn(self.bn_fc2, self.fc2(h), update_batch_stats))
        endpoints["fc_layer1"] = h
        if self.top_bn:
            h = call_bn(self.bn_fc3, self.fc3(h), update_batch_stats)
        else:
            h = self.fc3(h)
        logits = h
        if return_h:
            return logits, endpoints
        else:
            return logits


"""
[url](https://github.com/takerum/vat/blob/master/models/fnn_mnist_sup.py)

or

[url](https://github.com/takerum/vat/blob/master/models/fnn_mnist_semisup.py)

###VAT for semi-supervised learning on MNIST dataset (with 100 labeled samples)

python train_mnist_semisup.py --cost_type=VAT_finite_diff --epsilon=0.3 --layer_sizes=784-1200-1200-10 --num_labeled_samples=100

"""