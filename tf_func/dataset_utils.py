import tensorflow as tf
import os, sys, pickle
import numpy as np
from scipy import linalg


def data_input_fn(filenames, batch_size=1000, shuffle=False):
    
    def _parser(record):
        features={
            'label': tf.FixedLenFeature([], tf.int64),
            'image': tf.FixedLenFeature([], tf.string)
        }
        parsed_record = tf.parse_single_example(record, features)
        image = tf.decode_raw(parsed_record['image'], tf.float32)

        label = tf.cast(parsed_record['label'], tf.int32)

        return image, tf.one_hot(label, depth=10)
    
    def _iter():
        dataset = (tf.data.TFRecordDataset(filenames)
            .map(_parser))
        if shuffle:
            dataset = dataset.shuffle(buffer_size=10_000)

        dataset = dataset.repeat(None) # Infinite iterations: let experiment determine num_epochs
        dataset = dataset.batch(batch_size)
        
        iterator = dataset.make_one_shot_iterator()
        return iterator
    
    def _input_fn():        
        iterator = _iter()
        features, labels = iterator.get_next()
        
        return features, labels
    return _iter


def unpickle(file):
    fp = open(file, 'rb')
    if sys.version_info.major == 2:
        data = pickle.load(fp)
    elif sys.version_info.major == 3:
        data = pickle.load(fp, encoding='latin-1')
    fp.close()
    return data


def ZCA(data, reg=1e-6):
    mean = np.mean(data, axis=0)
    mdata = data - mean
    sigma = np.dot(mdata.T, mdata) / mdata.shape[0]
    U, S, V = linalg.svd(sigma)
    components = np.dot(np.dot(U, np.diag(1 / np.sqrt(S) + reg)), U.T)
    whiten = np.dot(data - mean, components.T)
    return components, mean, whiten


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def convert_images_and_labels(images, labels, filepath):
    num_examples = labels.shape[0]
    if images.shape[0] != num_examples:
        raise ValueError("Images size %d does not match label size %d." %
                         (images.shape[0], num_examples))
    print('Writing', filepath)
    writer = tf.python_io.TFRecordWriter(filepath)
    for index in range(num_examples):
        image = images[index].tolist()
        image_feature = tf.train.Feature(float_list=tf.train.FloatList(value=image))
        example = tf.train.Example(features=tf.train.Features(feature={
            'height': _int64_feature(32),
            'width': _int64_feature(32),
            'depth': _int64_feature(3),
            'label': _int64_feature(int(labels[index])),
            'image': image_feature}))
        writer.write(example.SerializeToString())
    writer.close()


def read(filename_queue, dataset="cifar10"):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    shape = 3072
    if dataset == "mnist":
        shape = 784
    features = tf.parse_single_example(
        serialized_example,
        # Defaults are not specified since both keys are required.
        features={
            'image': tf.FixedLenFeature([shape], tf.float32),
            'label': tf.FixedLenFeature([], tf.int64),
        })

    # Convert label from a scalar uint8 tensor to an int32 scalar.
    image = features['image']
    shape = [32, 32, 3]
    if dataset == "mnist":
        shape = [28, 28, 1]
    image = tf.reshape(image, shape)
    label = tf.one_hot(tf.cast(features['label'], tf.int32), 10)
    return image, label


def generate_batch(
        example,
        min_queue_examples,
        batch_size, shuffle):
    """
    Arg:
        list of tensors.
    """
    num_preprocess_threads = 1

    if shuffle:
        ret = tf.train.shuffle_batch(
            example,
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            capacity=min_queue_examples + 3 * batch_size,
            min_after_dequeue=min_queue_examples)
    else:
        ret = tf.train.batch(
            example,
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            allow_smaller_final_batch=True,
            capacity=min_queue_examples + 3 * batch_size)

    return ret


def transform(image, dataset="cifar10", aug_trans=False, aug_flip=False):
    shape = [32, 32, 3]
    if dataset == "mnist":
        shape = [28, 28, 1]
    image = tf.reshape(image, shape)
    if aug_trans or aug_flip:
        print("augmentation")
        if aug_trans:
            image = tf.pad(image, [[2, 2], [2, 2], [0, 0]])
            image = tf.random_crop(image, shape)
        if aug_flip:
            image = tf.image.random_flip_left_right(image)
    return image


def generate_filename_queue(filenames, data_dir, num_epochs=None):
    # print("filenames in queue:", filenames)
    for i in range(len(filenames)):
        filenames[i] = os.path.join(data_dir, filenames[i])
    return tf.train.string_input_producer(filenames, num_epochs=num_epochs)


