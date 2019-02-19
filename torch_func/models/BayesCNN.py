import torch
import torch.nn as nn
import torch.nn.functional as func
from torch.autograd import Variable

# number of convolutional filters to use
nb_filters = 32
# size of pooling area for max pooling
nb_pool = 2
# convolution kernel size
nb_conv = 4


class Flatten(nn.Module):
    """
    convert from keras, for the datasets
    keras: (num of data, height , width, channel)
    PyTorch: (num of data, channel, height, width)
    need permute and contiguous()
    """
    def forward(self, data):
        if len(data.shape) == 4:
            # k, x, y, z = data.shape
            data = data.permute(0, 2, 3, 1).contiguous()

        return data.view(data.size(0), -1)


class BayCNN(nn.Module):
    def __init__(self, pretrained=False, dropout=True, batchnorm=False, input_shape=(1, 28, 28), num_classes=10):
        super(BayCNN, self).__init__()
        self.batchnorm = batchnorm

        conv1 = nn.Sequential(
            nn.Conv2d(input_shape[0], nb_filters, kernel_size=nb_conv),
            nn.ReLU(),
            nn.Conv2d(nb_filters, nb_filters, kernel_size=nb_conv),
            nn.ReLU(),
            nn.MaxPool2d(nb_pool),
        )
        if dropout:
            conv1.add_module("drop1", nn.Dropout2d(0.25))

        conv2 = nn.Sequential(
            nn.Conv2d(nb_filters, nb_filters * 2, kernel_size=nb_conv),
            nn.ReLU(),
            nn.Conv2d(nb_filters * 2, nb_filters * 2, kernel_size=nb_conv),
            nn.ReLU(),
            nn.MaxPool2d(nb_pool),
        )
        if dropout:
            conv2.add_module("drop2", nn.Dropout2d(0.25))
        self.conv = nn.Sequential(
            conv1,
            conv2,
        )
        input_size = self._get_conv_output_size(input_shape)
        self.dense = nn.Sequential(
            Flatten(),
            nn.Linear(input_size, 128),
        )
        self.bn = nn.BatchNorm1d(128)
        if dropout:
            self.dense.add_module("drop3", nn.Dropout2d(0.5))

        self.fc = nn.Sequential(
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def _get_conv_output_size(self, shape):
        bs = 1
        inputs = Variable(torch.rand(bs, *shape))
        output_feat = self.conv(inputs)
        n_size = output_feat.data.view(bs, -1).size(1)
        return n_size

    def forward(self, x):
        x = self.conv(x)
        x = self.dense(x)
        if self.batchnorm:
            x = self.bn(x)
        x = self.fc(x)
        return x


class BayShallowCNN(nn.Module):
    def __init__(self, pretrained=False, dropout=True, batchnorm=False, input_shape=(1, 28, 28), num_classes=10):
        super(BayShallowCNN, self).__init__()
        self.dropout = dropout
        self.batchnorm = batchnorm
        self.drop_rate = [0.25, 0.5]

        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], nb_filters, kernel_size=nb_conv),
            nn.ReLU(),
            nn.Conv2d(nb_filters, nb_filters, kernel_size=nb_conv),
            nn.ReLU(),
            nn.MaxPool2d(nb_pool),
        )

        input_size = self._get_conv_output_size(input_shape)
        self.dense = nn.Sequential(
            Flatten(),
            nn.Linear(input_size, 128)
        )
        self.bn = nn.BatchNorm1d(128)
        self.fc = nn.Sequential(
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def _get_conv_output_size(self, shape):
        bs = 1
        input_variable = torch.rand(bs, *shape)
        output_feat = self.conv(input_variable)
        n_size = output_feat.data.view(bs, -1).size(1)
        return n_size

    def set_drop_rate(self, drop_rate=None):
        if drop_rate is None:
            drop_rate = [0.25, 0.5]
        self.drop_rate = drop_rate

    def forward(self, x):
        x = self.conv(x)
        if self.dropout:
            x = func.dropout2d(x, self.drop_rate[0], self.training)
        x = self.dense(x)
        if self.batchnorm:
            x = self.bn(x)
        if self.dropout:
            x = func.dropout2d(x, self.drop_rate[1], self.training)
        x = self.fc(x)
        return x


class BayDeepCNN(nn.Module):
    def __init__(self, pretrained=False, dropout=True, batchnorm=False, input_shape=(3, 32, 32), num_classes=10):
        super(BayDeepCNN, self).__init__()
        self.batchnorm = batchnorm

        conv1 = nn.Sequential(
            nn.Conv2d(input_shape[0], nb_filters, kernel_size=3, padding=1),
            nn.ELU(),
            nn.Conv2d(nb_filters, nb_filters, kernel_size=3, padding=1),
            nn.ELU(),
            nn.MaxPool2d(nb_pool),
        )
        if dropout:
            conv1.add_module("drop1", nn.Dropout2d(0.25))

        conv2 = nn.Sequential(
            nn.Conv2d(nb_filters, nb_filters * 2, kernel_size=3, padding=1),
            nn.ELU(),
            nn.Conv2d(nb_filters * 2, nb_filters * 2, kernel_size=3, padding=1),
            nn.ELU(),
            nn.MaxPool2d(nb_pool),
        )
        if dropout:
            conv2.add_module("drop2", nn.Dropout2d(0.25))
        conv3 = nn.Sequential(
            nn.Conv2d(nb_filters * 2, nb_filters * 4, kernel_size=3, padding=1),
            nn.ELU(),
            nn.Conv2d(nb_filters * 4, nb_filters * 4, kernel_size=3, padding=1),
            nn.ELU(),
            nn.MaxPool2d(nb_pool),
        )
        if dropout:
            conv3.add_module("drop3", nn.Dropout2d(0.25))
        self.conv = nn.Sequential(
            conv1,
            conv2,
            conv3,
        )
        input_size = self._get_conv_output_size(input_shape)
        self.dense = nn.Sequential(
            Flatten(),
            nn.Linear(input_size, 128),
        )
        self.bn = nn.BatchNorm1d(128)
        if dropout:
            self.dense.add_module("drop4", nn.Dropout2d(0.5))

        self.fc = nn.Sequential(
            nn.ELU(),
            nn.Linear(128, num_classes)
        )

    def _get_conv_output_size(self, shape):
        bs = 1
        inputs = Variable(torch.rand(bs, *shape))
        output_feat = self.conv(inputs)
        n_size = output_feat.data.view(bs, -1).size(1)
        return n_size

    def forward(self, x):
        x = self.conv(x)
        x = self.dense(x)
        if self.batchnorm:
            x = self.bn(x)
        x = self.fc(x)
        return x


def bay_cnn(pretrained=False,  input_shape=None, num_classes=10,  **kwargs):
    return BayCNN(**kwargs)


def bay_s_cnn(pretrained=False,  input_shape=None, num_classes=10,  **kwargs):
    return BayShallowCNN(**kwargs)


def bay_d_cnn(pretrained=False, input_shape=None, num_classes=10,  **kwargs):
    return BayDeepCNN(**kwargs)
