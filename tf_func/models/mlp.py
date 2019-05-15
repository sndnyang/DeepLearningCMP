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

import os
import numpy as np
import tensorflow as tf

from .. import layers as L


MOMENTUM = 0.9
EPS = 1e-5


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


class MLP(object):
    """MNIST model. """

    def __init__(self, config):
        self.num_classes = config.num_classes
        self.var_list = []
        self.init_ops = None
        self.activation = 'relu'
        self.filter_sizes_conv_layers = []
        self.num_units_fc_layers = [1200, 1200]
        self.pool_params = config.pool_params
        self.dropout_rate = config.dropout_rate
        self.batch_norm = config.batch_norm
        self.top_bn = config.top_bn

    def __call__(self, images, is_training=False, update_batch_stats=True, stochastic=True, seed=1234):
        """Builds model."""
        endpoints = {}
        net = images
        reuse = tf.AUTO_REUSE
        net = tf.layers.flatten(net)

        for i, num_units in enumerate(self.num_units_fc_layers):
            layer_suffix = "layer%d" % i
            with tf.variable_scope(os.path.join("mnist_network", "fc_" + layer_suffix), reuse=reuse):
                net = tf.layers.dense(
                    net,
                    num_units,
                    activation=self.activation,
                    use_bias=True)
                # net = tf.layers.batch_normalization(net, training=is_training)
                net = L.bn(net, num_units, is_training=is_training, update_batch_stats=update_batch_stats, name='b1')

            endpoints["fc_" + layer_suffix] = net

        with tf.variable_scope( os.path.join("mnist_network", "output_layer"), reuse=reuse):
            logits = tf.layers.dense(
                net,
                self.num_classes,
                activation=None)
            if FLAGS.top_bn:
                logits = L.bn(logits, self.num_classes, is_training=is_training, update_batch_stats=update_batch_stats, name='b1')
        endpoints["logits"] = net

        return logits, endpoints


def logit(x, is_training=True, update_batch_stats=True, stochastic=True, seed=1234, top_bn=True):
    h = x
    endpoints = {}
    rng = np.random.RandomState(seed)
    h = tf.layers.flatten(h)
    # h = tf.reduce_mean(h, reduction_indices=[1, 2])  # Global average pooling
    h = L.fc(h, 784, 1200, seed=rng.randint(123456), name='fc1')
    h = L.relu(L.bn(h, 1200, is_training=is_training, update_batch_stats=update_batch_stats, name='b1'))
    endpoints["fc1"] = h
    h = L.fc(h, 1200, 1200, seed=rng.randint(123456), name='fc2')
    h = L.relu(L.bn(h, 1200, is_training=is_training, update_batch_stats=update_batch_stats, name='b2'))
    endpoints["fc2"] = h
    h = L.fc(h, 1200, 10, seed=rng.randint(123456), name='fc')

    if top_bn:
        h = L.bn(h, 10, is_training=is_training, update_batch_stats=update_batch_stats, name='bfc')

    return h, endpoints


def mlp3(config):
    return logit
