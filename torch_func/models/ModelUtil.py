import torch.nn as nn


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


class UnFlatten(nn.Module):
    def forward(self, data):
        size = data.view(data.size(0), -1)
        return data.view(data.size(0), size.shape[1], 1, 1)
