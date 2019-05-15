# coding=utf-8
# Copyright 2018 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Model for MNIST classification.

The model is a two layer convolutional network followed by a fully connected
layer. Changes to the model architecture can be made by modifying
mnist_config.py file.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
from absl import flags
import tensorflow as tf

FLAGS = flags.FLAGS

MOMENTUM = 0.9
EPS = 1e-5


def weight_init_tf(shape):
    fan_in = 0
    if len(shape) == 4:
        fan_in = shape[1] * shape[2] * shape[3]
    if len(shape) == 2:
        fan_in = shape[1]
    if fan_in:
        s = 1.0 * np.sqrt(6.0 / fan_in)
        transpose = np.random.uniform(-s, s, shape).astype("float32")
    if len(shape) == 2:
        transpose = transpose.T
    if len(shape) == 4:
        transpose = np.transpose(transpose, axes=(2, 3, 1, 0))
    print(shape, transpose.sum())
    return transpose


def pool2d_layer(inputs, pool_type, pool_size=2, pool_stride=2):
    """Pooling layer.

    Args:
      inputs: Tensor of size [batch, H, W, channels].
      pool_type: String ("max", or "average"), specifying pooling type.
      pool_size: Integer > 1 pooling size.
      pool_stride: Integer > 1 pooling stride.

    Returns:
      Pooling result.
    """
    if pool_type == "max":
        # Max pooling layer
        return tf.layers.max_pooling2d(
            inputs, pool_size=[pool_size] * 2, strides=pool_stride)

    elif pool_type == "average":
        # Average pooling layer
        return tf.layers.average_pooling2d(
            inputs, pool_size=[pool_size] * 2, strides=pool_stride)


class MNISTNetwork(tf.keras.Model):
    """MNIST model. """

    def __init__(self, config):
        self.num_classes = config.num_classes
        self.var_list = []
        self.init_ops = None
        self.regularizer = config.regularizer
        self.activation = config.activation
        self.filter_sizes_conv_layers = config.filter_sizes_conv_layers
        self.num_units_fc_layers = config.num_units_fc_layers
        self.pool_params = config.pool_params
        self.dropout_rate = config.dropout_rate
        self.batch_norm = config.batch_norm
        self.conv_layers = []
        in_channel = 1
        for i, filter_size in enumerate(self.filter_sizes_conv_layers):
            f_size = filter_size[0]
            conv_layer = tf.layers.Conv2D(kernel_size=filter_size[0], filters=filter_size[1], strides=(1, 1), padding="same",
                                          activation=self.activation, use_bias=self.batch_norm,
                                          kernel_initializer=tf.constant_initializer((weight_init_tf((filter_size[1], in_channel, f_size, f_size)))))
            self.conv_layers.append(conv_layer)
            in_channel = filter_size[1]
        self.fc_layers = []
        in_shape = 64 * 7 * 7
        for i, num_units in enumerate(self.num_units_fc_layers):
            fc_layer = tf.layers.Dense(num_units, activation=self.activation,
                                       kernel_initializer=tf.constant_initializer((weight_init_tf((num_units, in_shape)))),)
            self.fc_layers.append(fc_layer)
            in_shape = num_units
        self.output_layer = tf.layers.Dense(self.num_classes, activation=None,
                                            kernel_initializer=tf.constant_initializer((weight_init_tf((self.num_classes, in_shape)))),)

    def __call__(self, images, is_training=False):
        """Builds model."""
        endpoints = {}
        net = images
        for i in range(len(self.filter_sizes_conv_layers)):
            layer_suffix = "layer%d" % i
            net = self.conv_layers[i](net)
            if self.pool_params:
                net = pool2d_layer(net, pool_type=self.pool_params["type"], pool_size=self.pool_params["size"]
                                   , pool_stride=self.pool_params["stride"])

            if self.dropout_rate > 0:
                net = tf.layers.dropout(net, rate=self.dropout_rate, training=is_training)

            if self.batch_norm:
                net = tf.layers.batch_normalization(
                    net, training=is_training, momentum=MOMENTUM, epsilon=EPS)

            endpoints["conv_" + layer_suffix] = net
            print("conv %s" % layer_suffix, "%.4f" % net.numpy().sum())

        net = tf.layers.flatten(net)

        for i in range(len(self.num_units_fc_layers)):
            layer_suffix = "layer%d" % i
            net = self.fc_layers[i](net)
            endpoints["fc_" + layer_suffix] = net
            print("fully %s" % layer_suffix, "%.4f" % net.numpy().sum())

        logits = self.output_layer(net)
        print("logits", "%.4f" % logits.numpy().sum())
        endpoints["logits"] = logits

        return logits, endpoints
