import torch.nn as nn
import torch.nn.functional as nfunc

from .utils import call_bn


class CNN9c(nn.Module):

    def __init__(self, args):
        super(CNN9c, self).__init__()
        input_shape = (3, 32, 32)
        num_conv = 128
        affine = args.affine
        self.top_bn = args.top_bn
        self.dropout = args.dropout_rate
        self.c1 = nn.Conv2d(input_shape[0], num_conv, 3, 1, 1, bias=False)
        self.c2 = nn.Conv2d(num_conv, num_conv, 3, 1, 1, bias=False)
        self.c3 = nn.Conv2d(num_conv, num_conv, 3, 1, 1, bias=False)
        self.c4 = nn.Conv2d(num_conv, num_conv * 2, 3, 1, 1, bias=False)
        self.c5 = nn.Conv2d(num_conv * 2, num_conv * 2, 3, 1, 1, bias=False)
        self.c6 = nn.Conv2d(num_conv * 2, num_conv * 2, 3, 1, 1, bias=False)
        self.c7 = nn.Conv2d(num_conv * 2, num_conv * 4, 3, 1, 0, bias=False)
        self.c8 = nn.Conv2d(num_conv * 4, num_conv * 2, 1, 1, 0, bias=False)
        self.c9 = nn.Conv2d(num_conv * 2, 128, 1, 1, 0, bias=False)
        self.bn1 = nn.BatchNorm2d(num_conv, affine=affine)
        self.bn2 = nn.BatchNorm2d(num_conv, affine=affine)
        self.bn3 = nn.BatchNorm2d(num_conv, affine=affine)
        self.bn4 = nn.BatchNorm2d(num_conv * 2, affine=affine)
        self.bn5 = nn.BatchNorm2d(num_conv * 2, affine=affine)
        self.bn6 = nn.BatchNorm2d(num_conv * 2, affine=affine)
        self.bn7 = nn.BatchNorm2d(num_conv * 4, affine=affine)
        self.bn8 = nn.BatchNorm2d(num_conv * 2, affine=affine)
        self.bn9 = nn.BatchNorm2d(num_conv, affine=affine)
        self.mp1 = nn.MaxPool2d(2, 2)
        self.mp2 = nn.MaxPool2d(2, 2)
        self.mp3 = nn.MaxPool2d(2, 2)
        self.aap = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(128, 10, bias=not self.top_bn)
        self.drop1 = nn.Dropout2d(self.dropout)
        self.drop2 = nn.Dropout2d(self.dropout)

        if self.top_bn:
            self.bnf = nn.BatchNorm1d(10, affine=affine)

    def forward(self, x, update_batch_stats=True, return_h=False):
        h = x
        endpoints = {}
        h = self.c1(h)
        h = nfunc.leaky_relu(call_bn(self.bn1, h, update_batch_stats=update_batch_stats), negative_slope=0.1)
        h = self.c2(h)
        h = nfunc.leaky_relu(call_bn(self.bn2, h, update_batch_stats=update_batch_stats), negative_slope=0.1)
        h = self.c3(h)
        h = nfunc.leaky_relu(call_bn(self.bn3, h, update_batch_stats=update_batch_stats), negative_slope=0.1)
        h = self.mp1(h)
        if self.dropout:
            h = self.drop1(h)
        endpoints["conv_layer0"] = h

        h = self.c4(h)
        h = nfunc.leaky_relu(call_bn(self.bn4, h, update_batch_stats=update_batch_stats), negative_slope=0.1)
        h = self.c5(h)
        h = nfunc.leaky_relu(call_bn(self.bn5, h, update_batch_stats=update_batch_stats), negative_slope=0.1)
        h = self.c6(h)
        h = nfunc.leaky_relu(call_bn(self.bn6, h, update_batch_stats=update_batch_stats), negative_slope=0.1)
        h = self.mp2(h)
        if self.dropout:
            h = self.drop2(h)
        endpoints["conv_layer1"] = h

        h = self.c7(h)
        h = nfunc.leaky_relu(call_bn(self.bn7, h, update_batch_stats=update_batch_stats), negative_slope=0.1)
        h = self.c8(h)
        h = nfunc.leaky_relu(call_bn(self.bn8, h, update_batch_stats=update_batch_stats), negative_slope=0.1)
        h = self.c9(h)
        h = nfunc.leaky_relu(call_bn(self.bn9, h, update_batch_stats=update_batch_stats), negative_slope=0.1)
        h = self.mp3(h)
        h = self.aap(h)
        endpoints["fc_layer0"] = h
        output = self.linear(h.view(-1, 128))
        # cifar10 don't use
        # svhn use top_bn
        if self.top_bn:
            output = call_bn(self.bnf, output, update_batch_stats=update_batch_stats)
        if return_h:
            return output, endpoints
        else:
            return output
