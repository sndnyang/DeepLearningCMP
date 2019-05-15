# Copyright 2015 Google Inc. All Rights Reserved.
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
# ==============================================================================
"""Routine for decoding the CIFAR-10 binary file format."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import tarfile

import numpy as np
from scipy import linalg
import glob
import pickle

from six.moves import xrange  # pylint: disable=redefined-builtin
from six.moves import urllib

import tensorflow as tf

from .dataset_utils import *

DATA_URL = 'http://www.cs.toronto.edu/~kriz/mnist-python.tar.gz'

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('size', 100, "The number of labeled examples")

# Process images of this size. Note that this differs from the original CIFAR
# image size of 28 x 28. If one alters this number, then the entire model
# architecture will change and any model would need to be retrained.
IMAGE_SIZE = 28

# Global constants describing the MNIST data set.
NUM_CLASSES = 10
NUM_EXAMPLES_TRAIN = 60000
NUM_EXAMPLES_TEST = 10000


def load_mnist_dataset():
    if sys.version_info.major == 3:
        dataset = pickle.load(open('dataset/mnist.pkl', 'rb'), encoding="bytes")
    else:
        dataset = pickle.load(open('dataset/mnist.pkl', 'rb'))
    train_set_x = np.concatenate((dataset[0][0], dataset[1][0]), axis=0)
    train_set_y = np.concatenate((dataset[0][1], dataset[1][1]), axis=0)
    return (train_set_x, train_set_y), (dataset[2][0], dataset[2][1])


def load_mnist():
    """Download and extract the tarball from Alex's website."""
    dest_directory = FLAGS.data_dir
    if not os.path.exists(dest_directory):
        os.makedirs(dest_directory)
    filepath = 'dataset/mnist.pkl'
    if not os.path.exists(filepath):
        print("please download it firstly")
        sys.exit(-1)

    # Training set
    print("Loading training data...")

    if sys.version_info.major == 3:
        dataset = pickle.load(open('dataset/mnist.pkl', 'rb'), encoding="bytes")
    else:
        dataset = pickle.load(open('dataset/mnist.pkl', 'rb'))
    train_set_x = np.concatenate((dataset[0][0], dataset[1][0]), axis=0)
    train_set_y = np.concatenate((dataset[0][1], dataset[1][1]), axis=0)
    return (train_set_x, train_set_y), (dataset[2][0], dataset[2][1])


def prepare_dataset():
    (train_images, train_labels), (test_images, test_labels) = load_mnist()
    dirpath = os.path.join(FLAGS.data_dir, 'seed1')
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)

    rng = np.random.RandomState(1)
    rand_ix = rng.permutation(NUM_EXAMPLES_TRAIN)
    _train_images, _train_labels = train_images[rand_ix], train_labels[rand_ix]

    examples_per_class = int(FLAGS.size / 10)
    labeled_train_images = np.zeros((FLAGS.size, 784), dtype=np.float32)
    labeled_train_labels = np.zeros(FLAGS.size, dtype=np.int64)
    for i in range(10):
        ind = np.where(_train_labels == i)[0]
        labeled_train_images[i * examples_per_class:(i + 1) * examples_per_class] \
            = _train_images[ind[0:examples_per_class]]
        labeled_train_labels[i * examples_per_class:(i + 1) * examples_per_class] \
            = _train_labels[ind[0:examples_per_class]]
        _train_images = np.delete(_train_images, ind[0:examples_per_class], 0)
        _train_labels = np.delete(_train_labels, ind[0:examples_per_class])

    rand_ix_labeled = rng.permutation(FLAGS.size)
    labeled_train_images, labeled_train_labels = labeled_train_images[rand_ix_labeled], labeled_train_labels[rand_ix_labeled]

    convert_images_and_labels(labeled_train_images, labeled_train_labels, os.path.join(dirpath, 'labeled_train.tfrecords'))
    convert_images_and_labels(train_images, train_labels, os.path.join(dirpath, 'unlabeled_train.tfrecords'))
    convert_images_and_labels(test_images, test_labels, os.path.join(dirpath, 'test.tfrecords'))

    # Construct dataset for validation
    if 1000 > FLAGS.size:
        train_images_valid, train_labels_valid = labeled_train_images[:], labeled_train_labels[:]
    else:
        train_images_valid, train_labels_valid = labeled_train_images[1000:], labeled_train_labels[1000:]
    test_images_valid, test_labels_valid = labeled_train_images[:1000], labeled_train_labels[:1000]
    unlabeled_train_images_valid = np.concatenate((train_images_valid, _train_images), axis=0)
    unlabeled_train_labels_valid = np.concatenate((train_labels_valid, _train_labels), axis=0)
    convert_images_and_labels(train_images_valid, train_labels_valid, os.path.join(dirpath, 'labeled_train_val.tfrecords'))
    convert_images_and_labels(unlabeled_train_images_valid, unlabeled_train_labels_valid, os.path.join(dirpath, 'unlabeled_train_val.tfrecords'))
    convert_images_and_labels(test_images_valid, test_labels_valid, os.path.join(dirpath, 'test_val.tfrecords'))


def inputs(size=100, batch_size=100,
           train=True, validation=False,
           shuffle=True, num_epochs=None):
    if validation:
        if train:
            filenames = ['labeled_train_val.tfrecords']
            num_examples = size - 1000
        else:
            filenames = ['test_val.tfrecords']
            num_examples = 1000
    else:
        if train:
            filenames = ['labeled_train.tfrecords']
            num_examples = size
        else:
            filenames = ['test.tfrecords']
            num_examples = NUM_EXAMPLES_TEST

    filenames = [os.path.join('seed1', filename) for filename in filenames]

    filename_queue = generate_filename_queue(filenames, '~/project/data/dataset/mnist', num_epochs)
    image, label = read(filename_queue, dataset="mnist")
    image = transform(tf.cast(image, tf.float32), dataset="mnist") if train else image
    return generate_batch([image, label], num_examples, batch_size, shuffle)


def unlabeled_inputs(batch_size=100,
                     validation=False,
                     shuffle=True):
    if validation:
        filenames = ['unlabeled_train_val.tfrecords']
        num_examples = NUM_EXAMPLES_TRAIN - 1000
    else:
        filenames = ['unlabeled_train.tfrecords']
        num_examples = NUM_EXAMPLES_TRAIN

    filenames = [os.path.join('seed1', filename) for filename in filenames]
    filename_queue = generate_filename_queue(filenames, '~/project/data/dataset/mnist')
    image, label = read(filename_queue, dataset="mnist")
    image = transform(tf.cast(image, tf.float32), dataset="mnist")
    return generate_batch([image], num_examples, batch_size, shuffle)


def main(argv):
    # prepare_dataset()
    images, labels = inputs(batch_size=128,
                            train=True,
                            validation=False,
                            shuffle=True)
    print(labels)
    for i in range(100):
        print(0)


if __name__ == "__main__":
    tf.app.flags.DEFINE_string('data_dir', os.path.join(os.environ['HOME'], 'project/data/dataset/mnist'), 'where to store the dataset')
    tf.app.run()
