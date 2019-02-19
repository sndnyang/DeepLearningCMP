import torch
import torch.nn as nn
import torch.nn.functional as nfunc

from .ModelUtil import Flatten

from .MLP import call_bn


class CNN2(nn.Module):
    def __init__(self, input_shape=(1, 28, 28)):
        super(CNN2, self).__init__()
        # 1 input image channel, 6 output input_shape[0], 5x5 square convolution
        # kernel
        self.conv1 = nn.Conv2d(input_shape[0], 20, 5)
        self.bn1 = nn.BatchNorm2d(20, eps=2e-5)
        self.conv2 = nn.Conv2d(20, 50, 5)
        self.bn1 = nn.BatchNorm2d(50, eps=2e-5)
        # an affine operation: y = Wx + b
        input_size = self._get_conv_output_size(input_shape)
        self.fc1 = nn.Linear(input_size, 500)
        self.bn1 = nn.BatchNorm1d(500, eps=2e-5)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        conv1_out = self.conv1(x)
        pool1_out, pool1_ind = nfunc.max_pool2d(conv1_out, (2, 2), return_indices=True)
        conv2_out = self.conv2(pool1_out)
        pool2_out, pool2_ind = nfunc.max_pool2d(conv2_out, (2, 2), return_indices=True)
        flat_out = pool2_out.view(-1, self.num_flat_features(pool2_out))
        fc1_out = self.fc1(flat_out)

        relu1_out = nfunc.relu(self.bn1(fc1_out))
        fc2_out = self.fc2(relu1_out)
        return fc2_out

    def _get_conv_output_size(self, shape):
        bs = 1
        inputs = torch.rand(bs, *shape)
        outputs = nn.MaxPool2d((2, 2))(self.conv1(inputs))
        output_feat = nn.MaxPool2d((2, 2))(self.conv2(outputs))
        n_size = output_feat.data.view(bs, -1).size(1)
        return n_size

    @staticmethod
    def num_flat_features(x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


class CNN3(nn.Module):

    def __init__(self, input_shape=(3, 32, 32)):
        super(CNN3, self).__init__()
        self.conv1 = nn.Conv2d(input_shape[0], 64, kernel_size=5)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3)
        self.fc1 = nn.Linear(128, 10)

    def forward(self, x):
        x = nfunc.relu(nfunc.max_pool2d(self.conv1(x), 2))
        x = nfunc.relu(nfunc.max_pool2d(self.conv2(x), 2))
        x = nfunc.relu(nfunc.max_pool2d(self.conv3(x), 2))
        x = nfunc.adaptive_avg_pool2d(x, 1)
        x = x.view(-1, 128)
        x = self.fc1(x)
        return x


class CNN9(nn.Module):

    def __init__(self, input_shape=(3, 32, 32), num_conv=64):
        super(CNN9, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(input_shape[0], num_conv, 3, 1, 1, bias=False),
            nn.BatchNorm2d(num_conv, eps=2e-5),
            nn.LeakyReLU(0.1),

            nn.Conv2d(num_conv, num_conv, 3, 1, 1, bias=False),
            nn.BatchNorm2d(num_conv, eps=2e-5),
            nn.LeakyReLU(0.1),

            nn.Conv2d(num_conv, num_conv, 3, 1, 1, bias=False),
            nn.BatchNorm2d(num_conv, eps=2e-5),
            nn.LeakyReLU(0.1),

            nn.MaxPool2d(2, 2),
            nn.Dropout2d(),

            nn.Conv2d(num_conv, num_conv * 2, 3, 1, 1, bias=False),
            nn.BatchNorm2d(num_conv * 2, eps=2e-5),
            nn.LeakyReLU(0.1),

            nn.Conv2d(num_conv * 2, num_conv * 2, 3, 1, 1, bias=False),
            nn.BatchNorm2d(num_conv * 2, eps=2e-5),
            nn.LeakyReLU(0.1),

            nn.Conv2d(num_conv * 2, num_conv * 2, 3, 1, 1, bias=False),
            nn.BatchNorm2d(num_conv * 2, eps=2e-5),
            nn.LeakyReLU(0.1),

            nn.MaxPool2d(2, 2),
            nn.Dropout2d(),

            nn.Conv2d(num_conv * 2, num_conv * 2, 3, 1, 0, bias=False),
            nn.BatchNorm2d(num_conv * 2, eps=2e-5),
            nn.LeakyReLU(0.1),

            nn.Conv2d(num_conv * 2, num_conv * 2, 1, 1, 1, bias=False),
            nn.BatchNorm2d(num_conv * 2, eps=2e-5),
            nn.LeakyReLU(0.1),

            nn.Conv2d(num_conv * 2, 128, 1, 1, 1, bias=False),
            nn.BatchNorm2d(128, eps=2e-5),
            nn.LeakyReLU(0.1),

            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.linear = nn.Linear(128, 10)
        self.bn = nn.BatchNorm1d(1, eps=2e-50)

    def forward(self, input):
        output = self.main(input)
        output = self.linear(output.view(input.size()[0], -1))
        # if self.top_bn:
        #     output = self.bn(output)
        return output


class CNN9b(nn.Module):

    def __init__(self, input_shape=(3, 32, 32), num_conv=96):
        super(CNN9b, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(input_shape[0], num_conv, 3, 1, 1, bias=False),
            nn.BatchNorm2d(num_conv, eps=2e-5),
            nn.LeakyReLU(0.1),

            nn.Conv2d(num_conv, num_conv, 3, 1, 1, bias=False),
            nn.BatchNorm2d(num_conv, eps=2e-5),
            nn.LeakyReLU(0.1),

            nn.Conv2d(num_conv, num_conv, 3, 1, 1, bias=False),
            nn.BatchNorm2d(num_conv, eps=2e-5),
            nn.LeakyReLU(0.1),

            nn.MaxPool2d(2, 2, 1),
            nn.Dropout2d(),

            nn.Conv2d(num_conv, num_conv * 2, 3, 1, 1, bias=False),
            nn.BatchNorm2d(num_conv * 2, eps=2e-5),
            nn.LeakyReLU(0.1),

            nn.Conv2d(num_conv * 2, num_conv * 2, 3, 1, 1, bias=False),
            nn.BatchNorm2d(num_conv * 2, eps=2e-5),
            nn.LeakyReLU(0.1),

            nn.Conv2d(num_conv * 2, num_conv * 2, 3, 1, 1, bias=False),
            nn.BatchNorm2d(num_conv * 2, eps=2e-5),
            nn.LeakyReLU(0.1),

            nn.MaxPool2d(2, 2, 1),
            nn.Dropout2d(),

            nn.Conv2d(num_conv * 2, num_conv * 2, 3, 1, 0, bias=False),
            nn.BatchNorm2d(num_conv * 2, eps=2e-5),
            nn.LeakyReLU(0.1),

            nn.Conv2d(num_conv * 2, num_conv * 2, 1, 1, 1, bias=False),
            nn.BatchNorm2d(num_conv * 2, eps=2e-5),
            nn.LeakyReLU(0.1),

            nn.Conv2d(num_conv * 2, 192, 1, 1, 1, bias=False),
            nn.BatchNorm2d(192, eps=2e-5),
            nn.LeakyReLU(0.1),

            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.linear = nn.Linear(192, 10)
        self.bn = nn.BatchNorm1d(1, eps=2e-50)

    def forward(self, input):
        output = self.main(input)
        output = self.linear(output.view(input.size()[0], -1))
        # if self.top_bn:
        #     output = self.bn(output)
        return output


class CNN9c(nn.Module):

    def __init__(self, input_shape=(3, 32, 32), num_conv=128, top_bn=False):
        super(CNN9c, self).__init__()
        self.c1 = nn.Conv2d(input_shape[0], num_conv, 3, 1, 1)
        self.c2 = nn.Conv2d(num_conv, num_conv, 3, 1, 1)
        self.c3 = nn.Conv2d(num_conv, num_conv, 3, 1, 1)
        self.c4 = nn.Conv2d(num_conv, num_conv * 2, 3, 1, 1)
        self.c5 = nn.Conv2d(num_conv * 2, num_conv * 2, 3, 1, 1)
        self.c6 = nn.Conv2d(num_conv * 2, num_conv * 2, 3, 1, 1)
        self.c7 = nn.Conv2d(num_conv * 2, num_conv * 4, 3, 1, 0)
        self.c8 = nn.Conv2d(num_conv * 4, num_conv * 2, 1, 1, 0)
        self.c9 = nn.Conv2d(num_conv * 2, 128, 1, 1, 0)
        self.bn1 = nn.BatchNorm2d(num_conv, eps=2e-5)
        self.bn2 = nn.BatchNorm2d(num_conv, eps=2e-5)
        self.bn3 = nn.BatchNorm2d(num_conv, eps=2e-5)
        self.bn4 = nn.BatchNorm2d(num_conv * 2, eps=2e-5)
        self.bn5 = nn.BatchNorm2d(num_conv * 2, eps=2e-5)
        self.bn6 = nn.BatchNorm2d(num_conv * 2, eps=2e-5)
        self.bn7 = nn.BatchNorm2d(num_conv * 4, eps=2e-5)
        self.bn8 = nn.BatchNorm2d(num_conv * 2, eps=2e-5)
        self.bn9 = nn.BatchNorm2d(num_conv, eps=2e-5)
        self.bnf = nn.BatchNorm2d(128, eps=2e-5)
        self.mp1 = nn.MaxPool2d(2, 2)
        self.mp2 = nn.MaxPool2d(2, 2)
        self.drop1 = nn.Dropout2d()
        self.drop2 = nn.Dropout2d()
        self.aap = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(128, 10)

        self.top_bn = top_bn
        if top_bn:
            self.bn = nn.BatchNorm1d(10, eps=2e-5)

    def forward(self, x, update_batch_stats=True):
        h = x
        h = self.c1(h)
        h = nfunc.leaky_relu(call_bn(self.bn1, h, update_batch_stats=update_batch_stats), negative_slope=0.1)
        h = self.c2(h)
        h = nfunc.leaky_relu(call_bn(self.bn2, h, update_batch_stats=update_batch_stats), negative_slope=0.1)
        h = self.c3(h)
        h = nfunc.leaky_relu(call_bn(self.bn3, h, update_batch_stats=update_batch_stats), negative_slope=0.1)
        h = self.mp1(h)
        h = self.drop1(h)

        h = self.c4(h)
        h = nfunc.leaky_relu(call_bn(self.bn4, h, update_batch_stats=update_batch_stats), negative_slope=0.1)
        h = self.c5(h)
        h = nfunc.leaky_relu(call_bn(self.bn5, h, update_batch_stats=update_batch_stats), negative_slope=0.1)
        h = self.c6(h)
        h = nfunc.leaky_relu(call_bn(self.bn6, h, update_batch_stats=update_batch_stats), negative_slope=0.1)
        h = self.mp2(h)
        h = self.drop2(h)

        h = self.c7(h)
        h = nfunc.leaky_relu(call_bn(self.bn7, h, update_batch_stats=update_batch_stats), negative_slope=0.1)
        h = self.c8(h)
        h = nfunc.leaky_relu(call_bn(self.bn8, h, update_batch_stats=update_batch_stats), negative_slope=0.1)
        h = self.c9(h)
        h = nfunc.leaky_relu(call_bn(self.bn9, h, update_batch_stats=update_batch_stats), negative_slope=0.1)
        h = self.aap(h)
        output = self.linear(h.view(-1, 128))
        if self.top_bn:
            output = call_bn(self.bnf, output, update_batch_stats=update_batch_stats)
        return output


class ConvNet(nn.Module):
    def __init__(self, input_shape=(3, 32, 32)):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(input_shape[0], 32, 5, padding=2)
        self.conv2 = nn.Conv2d(32, 32, 5, padding=2)
        self.conv3 = nn.Conv2d(32, 64, 5, padding=2)
        self.conv4 = nn.Conv2d(64, 64, 4)
        self.conv5 = nn.Conv2d(64, 10, 1)

        for k in ['conv1', 'conv2', 'conv3', 'conv4', 'conv5']:
            w = self.__getattr__(k)
            nn.init.kaiming_normal_(w.weight.data)
            w.bias.data.fill_(0)

    def forward(self, x):
        x = self.conv1(x)
        x, pool1_ind = nfunc.max_pool2d(x, kernel_size=2, stride=2, return_indices=True)
        x = nfunc.relu(x)
        x = self.conv2(x)
        x = nfunc.relu(x)
        x = nfunc.avg_pool2d(x, kernel_size=2, stride=2)
        x = self.conv3(x)
        x = nfunc.relu(x)
        x = nfunc.avg_pool2d(x, kernel_size=2, stride=2)
        x = self.conv4(x)
        x = nfunc.relu(x)
        x = self.conv5(x)
        x = x.view(-1, 10)

        return x


class CNN5(nn.Module):
    def __init__(self, input_shape=(3, 32, 32)):
        super(CNN5, self).__init__()
        self.conv1 = nn.Conv2d(input_shape[0], 32, 5, padding=2)
        self.conv2 = nn.Conv2d(32, 32, 5, padding=2)
        self.conv3 = nn.Conv2d(32, 64, 5, padding=2)
        self.conv4 = nn.Conv2d(64, 64, 4)
        self.conv5 = nn.Conv2d(64, 10, 1)

        for k in ['conv1', 'conv2', 'conv3', 'conv4', 'conv5']:
            w = self.__getattr__(k)
            nn.init.kaiming_normal_(w.weight.data)
            w.bias.data.fill_(0)

        self.out = dict()

    def forward(self, x):
        x = self.conv1(x)
        x, pool1_ind = nfunc.max_pool2d(x, kernel_size=2, stride=2, return_indices=True)
        x = nfunc.relu(x)

        x = self.conv2(x)
        x = nfunc.relu(x)
        x = nfunc.avg_pool2d(x, kernel_size=2, stride=2)

        x = self.conv3(x)

        x = nfunc.relu(x)
        x = nfunc.avg_pool2d(x, kernel_size=2, stride=2)

        x = self.conv4(x)
        x = nfunc.relu(x)

        x = self.conv5(x)

        x = x.view(-1, 10)

        return x


def cnn2(pretrained=False, input_shape=None, num_classes=10,  **kwargs):
    return CNN2(input_shape)


def cnn3(pretrained=False, input_shape=None, num_classes=10,  **kwargs):
    return CNN3(input_shape)


def cnn5(pretrained=False, input_shape=None, num_classes=10,  **kwargs):
    return CNN5(input_shape)


def cnn9(pretrained=False, input_shape=None, num_classes=10, **kwargs):
    return CNN9(input_shape)


def cnn9b(pretrained=False, input_shape=None, num_classes=10, **kwargs):
    return CNN9b(input_shape)


def cnn9c(pretrained=False, input_shape=None, num_classes=10, **kwargs):
    return CNN9c(input_shape)


def convnet(pretrained=False, input_shape=None, num_classes=10, **kwargs):
    return ConvNet(input_shape)
